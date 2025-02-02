import sys
import time
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
import chex

from sims.AlphaTrade.gymnax_exchange.jaxob.jorderbook import *
from sims.AlphaTrade.gymnax_exchange.jaxob.JaxOrderBookArrays import get_data_messages


@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    l2_asks: chex.Array
    l2_bids: chex.Array
    init_time: int
    time: int
    custom_ID_counter: int
    window_index: int
    init_price: int
    cash: float  # Agent's cash balance
    positions: chex.Array  # Array of open positions: shape (M, 2)
                            # Each position: [price, quantity]
    active_orders: chex.Array  # Shape: (M, 3), where M is the number
                                # of pending orders
    ema_mid_price: chex.Array  # ema of mid-prices
    profit_history: chex.Array  # Historical returns for reward calculation
    cash_utilisation_history: chex.Array # step-wise raw cash used over lookback window
    step_counter: int
    max_steps_in_episode: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    state_array_list: chex.Array
    episode_time: int = 60 * 30  # 1800 seconds
    time_per_step: int = 1e6  # 1000000 ns (1 ms) time delay
    initial_cash: float = 100000.0  # Initial cash for the agent
    risk_free_rate: float = 0.04  # Risk-free rate for Sortino ratio
    target_return: float = 0.0  # Target return for Sortino ratio
    max_position_limit: int = 10  # Maximum number of open positions allowed
# reward func = sortino ratio (returns - risk-free rate)/downside deviation - {abs(chosen ideal exposure - cash balance/portfolio mark-to-market) - maker_fee*number of limit orders opened or closed}

class L2BaseLOBEnv(environment.Environment):
    def __init__(
        self,
        data_path: str,
        n_actions: int = 5,
        agent_active_order_limit: int = 5,
        book_depth: int = 10,
        ema_alpha: float = 0.8,
        n_trades_logged: int = 100,
        n_orders_per_side: int = 100,
        seconds_per_window: int = 1800, # counted by seconds, 1800s=0.5h
        start_time_in_seconds: int = 34200, # 09:30
        end_time_in_seconds: int = 57600, # 16:00
        step_lines: int = 100,
        lookback_steps: int = 20,
        window_index: int = -1
    ):
        super().__init__()
        self.slice_time_window = seconds_per_window
        self.step_lines = step_lines
        self.message_path = f'{data_path}/trades/'
        self.ob_path = f'{data_path}/lobs/'
        self.start_time = start_time_in_seconds
        self.end_time = end_time_in_seconds
        self.n_orders_per_side = n_orders_per_side
        self.n_trades_logged = n_trades_logged
        self.book_depth = book_depth # Number of levels in the order book
                                    # to observe
        self.ema_alpha = ema_alpha
        self.n_actions = n_actions
        self.agent_active_order_limit = agent_active_order_limit
        self.custom_id_counter = 0
        self.trader_unique_id = -9000+1
        self.tick_size = 100
        self.lookback_window = lookback_steps  # Number of steps for
                                        # reward calculation
        self.window_index = window_index
        self.ob = OrderBook(
            nOrders=self.n_orders_per_side,
            nTrades=self.n_trades_logged
        )

        self.action_space_ = spaces.Box(
            low=jnp.array([0.0, 0.0, -1], dtype=jnp.float32),
            high=jnp.array([jnp.inf, jnp.inf, 1], dtype=jnp.float32),
            shape=(self.n_actions, 3),
            dtype=jnp.float32
        )

        self.observation_space_ = spaces.Dict({
            'best_bids': spaces.Box(
                low=0.0,
                high=jnp.inf,
                shape=(self.book_depth, 2),
                dtype=jnp.float32
            ),
            'best_asks': spaces.Box(
                low=0.0,
                high=jnp.inf,
                shape=(self.book_depth, 2),
                dtype=jnp.float32
            ),
            'inventory': spaces.Box(
                low=0,
                high=jnp.inf,
                shape=(1,),
                dtype=jnp.float32
            ),
            'mark_to_market': spaces.Box(
                low=0,
                high=jnp.inf,
                shape=(1,),
                dtype=jnp.float32
            ),
            # Add any other observations you deem appropriate
        })

        # Load the LOB data from source (e.g. crypto, LOBSTER)
        def load_LOBSTER(slice_time_window, step_lines, message_path, orderbook_path, start_time, end_time):
            """
            Load and preprocess data from LOBSTER files.
            """
            # Define internal functions for loading and processing data
            def load_files():
                from os import listdir
                from os.path import isfile, join
                import pandas as pd

                def read_from_path(data_path):
                    return sorted(
                        [f for f in listdir(data_path) if isfile(join(data_path, f))]
                    )

                message_files = read_from_path(message_path)
                orderbook_files = read_from_path(orderbook_path)

                dtype = {0: float, 1: int, 2: int, 3: int, 4: int, 5: int}
                message_csvs = [
                    pd.read_csv(
                        message_path + file, usecols=range(6), dtype=dtype, header=None
                    )
                    for file in message_files
                    if file.endswith("csv")
                ]
                orderbook_csvs = [
                    pd.read_csv(orderbook_path + file, header=None)
                    for file in orderbook_files
                    if file.endswith("csv")
                ]
                return message_csvs, orderbook_csvs

            messages, orderbooks = load_files()

            # Preprocess messages and orderbooks
            def preprocess_message_ob(message, orderbook):
                def split_timestamp(m):
                    m[6] = m[0].apply(lambda x: int(x))
                    m[7] = ((m[0] - m[6]) * int(1e9)).astype(int)
                    m.columns = [
                        "time",
                        "type",
                        "order_id",
                        "qty",
                        "price",
                        "direction",
                        "time_s",
                        "time_ns",
                    ]
                    return m

                message = split_timestamp(message)

                def filter_valid(m):
                    m = m[m.type.isin([1, 2, 3, 4])]
                    valid_index = m.index.to_numpy()
                    m.reset_index(inplace=True, drop=True)
                    return m, valid_index

                message, valid_index = filter_valid(message)

                def adjust_executions(m):
                    m.loc[m["type"] == 4, "direction"] *= -1
                    m.loc[m["type"] == 4, "type"] = 1
                    return m

                message = adjust_executions(message)

                def remove_deletes(m):
                    m.loc[m["type"] == 3, "type"] = 2
                    return m

                message = remove_deletes(message)

                def add_trader_id(m):
                    m["trader_id"] = m["order_id"]
                    return m

                message = add_trader_id(message)
                orderbook = orderbook.iloc[valid_index, :].reset_index(drop=True)
                return message, orderbook

            pairs = [
                preprocess_message_ob(message, orderbook)
                for message, orderbook in zip(messages, orderbooks)
            ]
            messages, orderbooks = zip(*pairs)

            # Slice data without overlap
            def index_of_slice_without_overlap(start, end, interval):
                return list(range(start, end, interval))

            indices = index_of_slice_without_overlap(start_time, end_time, slice_time_window)

            def slice_without_overlap(message, orderbook):
                import numpy as np

                def split_message(m, ob):
                    sliced_parts = []
                    init_obs = []
                    for i in range(len(indices) - 1):
                        start_idx = indices[i]
                        end_idx = indices[i + 1]
                        index_s, index_e = m[
                            (m["time"] >= start_idx) & (m["time"] < end_idx)
                        ].index[[0, -1]].tolist()
                        index_e = (
                            (index_e // step_lines + 10) * step_lines + index_s % step_lines
                        )
                        assert (
                            (index_e - index_s) % step_lines == 0
                        ), "Wrong code 31"
                        sliced_part = m.loc[np.arange(index_s, index_e)]
                        sliced_parts.append(sliced_part)
                        init_obs.append(ob.iloc[index_s, :])

                    # Last sliced part
                    start_idx = indices[-2]
                    end_idx = indices[-1]
                    index_s, index_e = m[
                        (m["time"] >= start_idx) & (m["time"] < end_idx)
                    ].index[[0, -1]].tolist()
                    index_s = (
                        (index_s // step_lines - 10) * step_lines + index_e % step_lines
                    )
                    assert (
                        (index_e - index_s) % step_lines == 0
                    ), "Wrong code 32"
                    last_sliced_part = m.loc[np.arange(index_s, index_e)]
                    sliced_parts.append(last_sliced_part)
                    init_obs.append(ob.iloc[index_s, :])

                    for part in sliced_parts:
                        assert (
                            part.time_s.iloc[-1] - part.time_s.iloc[0] >= slice_time_window
                        ), f"Wrong code 33, {part.time_s.iloc[-1] - part.time_s.iloc[0]}, {slice_time_window}"
                        assert part.shape[0] % step_lines == 0, "Wrong code 34"
                    return sliced_parts, init_obs

                sliced_parts, init_obs = split_message(message, orderbook)

                def sliced_to_cube(sliced):
                    columns = [
                        "type",
                        "direction",
                        "qty",
                        "price",
                        "trader_id",
                        "order_id",
                        "time_s",
                        "time_ns",
                    ]
                    cube = sliced[columns].to_numpy()
                    cube = cube.reshape((-1, step_lines, 8))
                    return cube

                sliced_cubes = [sliced_to_cube(part) for part in sliced_parts]
                sliced_cubes_with_ob = zip(sliced_cubes, init_obs)
                return sliced_cubes_with_ob

            sliced_cubes_with_ob_list = [
                slice_without_overlap(message, orderbook)
                for message, orderbook in zip(messages, orderbooks)
            ]

            # Flatten the list of cubes
            def nestlist_to_flatlist(nested_list):
                import itertools

                return list(itertools.chain.from_iterable(nested_list))

            cubes_with_OB = nestlist_to_flatlist(sliced_cubes_with_ob_list)
            max_steps_in_episode_arr = jnp.array(
                [m.shape[0] for m, _ in cubes_with_OB], dtype=jnp.int32
            )

            # Pad Cubes to have the same shape
            def cubes_with_ob_padding(cubes_with_ob):
                max_m = max(m.shape[0] for m, _ in cubes_with_ob)
                new_cubes_with_ob = []
                for cube, ob in cubes_with_ob:
                    def padding(c, target_shape):
                        padding = [(0, target_shape - c.shape[0]), (0, 0), (0, 0)]
                        padded_cube = np.pad(c, padding, mode="constant", constant_values=0)
                        return padded_cube

                    cube = padding(cube, max_m)
                    new_cubes_with_ob.append((cube, ob))
                return new_cubes_with_ob

            cubes_with_OB = cubes_with_ob_padding(cubes_with_OB)
            return cubes_with_OB, max_steps_in_episode_arr

        # Load the LOB data from source (e.g. crypto, LOBSTER)
        cubes_with_OB, max_steps_in_episode_arr = load_LOBSTER(
            self.slice_time_window,
            self.step_lines,
            self.message_path,
            self.ob_path,
            self.start_time,
            self.end_time
        )

        self.max_steps_in_episode_arr = max_steps_in_episode_arr

        # Extract messages and books from the loaded data
        # 4D Array: (n_windows x n_steps (max) x n_messages x n_features)
        self.messages = jnp.array([jnp.array(cube) for cube, _ in cubes_with_OB])
        # 2D Array: (n_windows x [4*n_depth])
        self.books = jnp.array([jnp.array(book) for _, book in cubes_with_OB])

        self.n_windows=len(self.books)

    @property
    def action_space(self) -> spaces.Box:
        return self.action_space_

    @property
    def observation_space(self) -> spaces.Dict:
        return self.observation_space_

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams
    ) -> Tuple[Dict, EnvState]:
        idx_data_window = (
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())
            if self.window_index == -1 else
            jnp.array(self.window_index, dtype=jnp.int32)
        )
        # state_array = params.state_array_list[idx_data_window]

        state = self.ob.reset(self.books[idx_data_window])
        l2_state = self.ob.get_L2_state(state, self.book_depth)
        l2_ask_prices, l2_ask_quants = l2_state[::2], l2_state[1::2]
        l2_bid_prices, l2_bid_quants = l2_state[2::2], l2_state[3::2]
        best_ask = [l2_ask_prices[0], l2_ask_quants[0]]
        best_bid = [l2_bid_prices[0], l2_bid_quants[0]]

        best_asks = jnp.array([[0, 0]]*self.lookback_window-1 + best_ask)
        best_bids = jnp.array([[0, 0]]*self.lookback_window-1 + best_bid)
        # Initialize the environment state
        state = EnvState(
            trades=state.trades,
            best_asks=best_asks,
            best_bids=best_bids,
            l2_asks=jnp.hstack([l2_ask_prices, l2_ask_quants]).flatten(),
            l2_bids=jnp.hstack([l2_bid_prices, l2_bid_quants]).flatten(),
            init_time=0,
            time=0,
            customIDcounter=0,
            window_index=idx_data_window,
            cash=params.initial_cash,
            positions=jnp.zeros((params.max_position_limit, 2)),  # Initialize positions to zeros
            mid_price=jnp.zeros(self.lookback_window),
            profit_history=jnp.zeros(self.lookback_window - 1),
            step_counter=0,
            max_steps_in_episode=self.max_steps_in_episode_arr[idx_data_window],
        )
        obs = self.get_obs(state)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams
    ) -> Tuple[Dict, EnvState, float, bool, dict]:
        """Performs one step in the environment."""
        # each action is a tuple (price, quantity, side)
        # side: -1 for sell, 1 for buy, 0 for hold

        # initialize variables
        cash = state.cash

        # get next set of data messages
        data_msgs = get_data_messages(
            params.message_data,
            state.window_index,
            state.step_counter
        )

        # process agent actions
        action_msgs = self._get_action_msgs(action, state, params)

        # deal with cancel messages
        cancel_msgs = self._get_cancel_msgs(action, state)

        # submit latest trade data & actions to the orderbook
        all_msgs = jnp.concatenate(
            [
                cancel_msgs,
                action_msgs,
                data_msgs
            ], axis=0)
        
        # get latest time for internal state
        time = all_msgs[-1][0][-2:]

        # if trades_reinit is set as true, only consider trades from the last
        # step so residual order book levels are cleared (levels that aren't
        # in original data feed), else carry last-state trades forward
        trades = (
            jnp.ones((self.n_trades_logged, 6), dtype=jnp.int32)*-1
            if self.trades_reinit else state.trades
        )

        state_ = LobState(
            state.ask_raw_orders,
            state.bid_raw_orders,
            trades
        )

        # process new lob state
        state_, l2_state = self.ob.process_orders_array_l2(
            state_,
            all_msgs,
            self.book_depth
        )

        #### UPDATE STATE
        l2_ask_prices, l2_ask_quants = l2_state[::2], l2_state[1::2]
        l2_bid_prices, l2_bid_quants = l2_state[2::2], l2_state[3::2]

        l2_asks = jnp.hstack([l2_ask_prices, l2_ask_quants]).flatten()
        l2_bids = jnp.hstack([l2_bid_prices, l2_bid_quants]).flatten()

        best_asks = jnp.concatenate(
            [state.best_asks[:self.lookback_window-1], l2_asks[:2]]
        )
        best_bids = jnp.concatenate(
            [state.best_bids[:self.lookback_window-1], l2_bids[:2]]
        )

        # get all non-empty trades
        trades  = state_.trades
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)

        # filter for training agent's trades
        mask = (
            (-9000 < executed[:, 2]) & (executed[:, 2] < 0)) \
                | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0)
        )
        agent_trades = jnp.where(mask[:, jnp.newaxis], executed, 0)

        # get new cash, positions & mark-to-market
        profit = self._get_trade_profits(agent_trades, state_)
        executed_trade_sides = jnp.where(
            mask,
            jnp.where(

            ),
            0
        )
        cash_used = (agent_trades[:, 0] * agent_trades[:, 1] * -1 * agent_trades[:, ]).sum()
        cash = state.cash - cash_uilised + profit
        positions = self._update_positions(agent_trades, state_)

        # get active ask orders
        asks = state_.asks
        mask = ((-9000 < asks[:, 2]) & (asks[:, 2] < 0)) | ((-9000 < asks[:, 3]) & (asks[:, 3] < 0))
        agent_active_asks = jnp.where(mask[:, jnp.newaxis], asks, 0)

        # fill non-agent asks with [0, 0, 0]
        agent_active_asks = jnp.where(
            agent_active_asks,
            jnp.hstack([[-1]*self.book_depth, asks[:, 0], asks[:, 1]]),
            jnp.array([0, 0, 0])
        )

        # separate asks from empty arrays
        agent_active_asks = jax.lax.sort_key_val(
            agent_active_asks[:, 0],
            agent_active_asks,
            dimension=0
        )[1][:self.agent_active_order_limit] # limit obs size to max
        # active order limit


        # get active bid orders
        bids = state_.bids
        mask = ((-9000 < bids[:, 2]) & (bids[:, 2] < 0)) | ((-9000 < bids[:, 3]) & (bids[:, 3] < 0))
        agent_active_bids = jnp.where(mask[:, jnp.newaxis], bids, 0)

        # fill non-agent bids with [0, 0, 0]
        agent_active_bids = jnp.where(
            agent_active_bids,
            jnp.hstack([[-1]*self.book_depth, bids[:, 0], bids[:, 1]]),
            jnp.array([0, 0, 0])
        )

        # separate bids from empty arrays
        agent_active_bids = jax.lax.sort_key_val(
            agent_active_bids[:, 0],
            agent_active_bids,
            dimension=0
        )[1][:self.agent_active_order_limit] # limit obs size to max
        # active order limit

        # Update prices history
        mid_price = (state.best_bids[0] + state.best_asks[0]) / 2
        ema_mid_price = self.ema_alpha*mid_price + (1-self.ema_alpha)*state.ema_mid_price
        state = state.replace(ema_mid_price=ema_mid_price)

        # Calculate returns history
        profit_history = jnp.concatenate([state.returns[:self.lookback_window-1], jnp.array(profit)])

        state = state.replace(
            ask_raw_orders=asks,
            bid_raw_orders=bids,
            trades=trades,
            l2_asks=l2_asks,
            l2_bids=l2_bids,
            best_asks=best_asks,
            best_bids=best_bids,
            cash=cash,
            active_orders=active_orders,
            positions=positions,
            profit_history=profit_history,
            time=time,
            custom_ID_counter=mask.sum(),
            step_counter=state.step_counter+1
        )

        # Compute reward (Sortino ratio)
        if state.step_counter >= self.lookback_window:
            # Calculate downside deviation
            # change returns history to be cash profit/cash_usage_history
            returns_history = jnp.diff(profit_history) / profit_history[:-1]
            negative_returns = jnp.minimum(profit_history - params.target_return, 0)
            downside_deviation = jnp.sqrt(jnp.mean(negative_returns ** 2) + 1e-8)

            # Calculate expected return
            expected_return = jnp.mean(profit_history)

            # Calculate Sortino ratio
            sortino_ratio = (expected_return - params.risk_free_rate) / downside_deviation
            reward = sortino_ratio
        else:
            reward = 0.0

        # Check if episode is done
        done = self.is_terminal(state, params)

        # Get the new observation
        obs = self.get_obs(state)

        info = {
            'cash': state.cash,
            'positions': state.positions,
            'profit': profit,
            'reward': reward,
            'time': state.time,
        }

        return obs, state, reward, done, info

    def get_obs(self, state: EnvState) -> Dict:
        """Generates the observation from the state."""

        # Extract Level-2 order book data up to K levels
        best_bids = state.l2_bids[:, 0]
        best_asks = state.l2_asks[:, 0]

        # Market features
        mid_prices = (
            ((best_asks + best_bids) // 2 // self.tick_size) * self.tick_size
        )
        spreads = best_asks - best_bids
        # ema = ...
        # sma = ...
        # std = ... # standard deviation
        # bb = ... # bollinger bands
        # lob_skew = ...
        # wmid_price_diff = ... # diff between mid-price and wmid-price
        imbalance = (
            state.best_asks[:, 1] - state.best_bids[:, 1]
        )

        # Internal features
        cash = state.cash
        inventory = state.positions
        mark_to_market = cash + inventory * best_bids[0]
        active_orders = state.active_orders

        # Other features
        time_of_day = state.time
        delta_t = state.time - state.init_time
        init_price = state.init_price

        obs = jnp.concatenate([
            best_bids,
            best_asks,
            mid_prices,
            spreads,
            imbalance,
            cash,
            inventory,
            mark_to_market,
            active_orders,
            time_of_day,
            delta_t,
            jnp.array([init_price]),
            jnp.array([state.step_counter]),
            jnp.array([state.max_steps_in_episode])
        ])

        # Normalize observations
        def obs_norm(obs):
            return jnp.concatenate(
                (
                    obs[:500] / 3.5e7,  # Price levels
                    obs[500:600] / 100000,  # Spreads
                    obs[600:700] / 100,  # Shallow imbalance
                    obs[700:701] / 100000, # Cash
                    obs[701:702] / 10000, # Inventory
                    obs[702:703] / 100000, # Mark-To-Market
                    obs[703:703+self.n_actions] / 3.5e7, # Active Order Prices
                    obs[703+self.n_actions:703+2*self.n_actions] / 3.5e7, # Active Order Quantities
                    obs[703+2*self.n_actions:704+2*self.n_actions] / 1e9,  # Time of day (nanoseconds)
                    obs[704+2*self.n_actions:705+2*self.n_actions] / 1e9,  # Delta T (nanoseconds)
                    obs[705+2*self.n_actions:706+2*self.n_actions] / 300,  # Step counter
                    obs[706+2*self.n_actions:707+2*self.n_actions] / 300,  # Max steps in episode
                )
            )

        obs = obs_norm(obs)

        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Checks whether the episode is done."""
        time_exceeded = (state.time - state.init_time)[0] > params.episode_time
        out_of_cash = state.cash <= 0
        return jnp.logical_or(time_exceeded, out_of_cash)
    
    def _get_action_msgs(action: chex.Array, state: EnvState, params: EnvParams) -> chex.Array:
        types = jnp.where(action[:,0])
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)

    @property
    def name(self) -> str:
        """Returns the name of the environment."""
        return "BaseL2LOBEnv-v0"

    # Additional methods for order matching, state updates, etc., would be implemented here.

# Usage example:
if __name__ == "__main__":
    data_path = "./data"  # Path to your data
    env = BaseL2LOBEnv(
        data_path,
        N_orders=5,
        K_levels=10,
        L_steps=20,
        max_steps_in_episode=1000
    )
    # Assuming you have the necessary data loaded into message_data, book_data, and state_array_list
    env_params = EnvParams(
        message_data=...,
        book_data=...,
        state_array_list=...,
        initial_cash=100000.0,
        initial_positions=0.0,
        risk_free_rate=0.01,
        target_return=0.0,
    )
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)
    done = False
    while not done:
        # Sample a random action
        rng, action_rng = jax.random.split(rng)
        action = env.action_space.sample(action_rng)
        obs, state, reward, done, info = env.step(rng, state, action, env_params)
        print(f"Step: {state.step_counter}, Reward: {reward}, cash: {state.cash}")
