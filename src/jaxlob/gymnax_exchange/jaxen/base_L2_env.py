import sys
import time
from typing import Callable, List, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
import chex

from jaxlob.gymnax_exchange.jaxob.jorderbook import *
from jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays import get_data_messages

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
    cash: float  # Agent's cash balance
    holdings: chex.Array  # Number representing amount of the asset
    active_orders: chex.Array  # Shape: (M, 3), where M is the number
                                # of pending orders
    orders_state: dict[str, chex.Array]
    mid_price: chex.Array  # array ofmid-prices
    profit_history: chex.Array  # Historical returns for reward calculation
    step_counter: int
    max_steps_in_episode: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    episode_time: int = 60 * 30  # 1800 seconds
    time_per_step: int = 1e8  # 100000000 ns (1 ms) time delay
    initial_cash: float = 100000.0  # Initial cash for the agent
    initial_holdings: float = 0.0  # Initial holdings for the agent
    risk_free_rate: float = 0.04  # Risk-free rate for Sortino ratio
    target_return: float = 0.0  # Target return for Sortino ratio
    max_active_orders: int = 10  # Maximum number of open positions allowed
    reward_functions: List[Callable] = None
    reward_coefficients: List[chex.Array] = None
    maker_fee: float = 0.0001  # Maker fee for limit orders
    taker_fee: float = 0.0005  # Taker fee for market orders
    max_limit_orders: int = 10  # Maximum number of limit orders allowed
# reward func = sortino ratio (returns - risk-free rate)/downside deviation - {abs(chosen ideal exposure - cash balance/portfolio mark-to-market) - maker_fee*number of limit orders opened or closed}

class L2BaseLOBEnv(environment.Environment):
    def __init__(
        self,
        data_path: str,
        n_actions: int = 5,
        max_active_orders: int = 10,
        agent_active_order_limit: int = 5,
        book_depth: int = 10,
        ema_alpha: float = 0.8,
        n_trades_logged: int = 100,
        trades_reinit: bool = True,
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
                                    # to observe
        self.book_depth = book_depth
        self.trades_reinit = trades_reinit
        self.ema_alpha = ema_alpha
        self.n_actions = n_actions
        self.max_active_orders = max_active_orders
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

        # self.action_space_ = spaces.Box(
        #     low=jnp.array([0.0, 0.0, 0.0, 0], dtype=jnp.float32),
        #     high=jnp.array([1, 1, jnp.inf, jnp.inf], dtype=jnp.float32),
        #     shape=(self.n_actions, 4),
        #     dtype=jnp.float32
        # )
        self.action_space_ = spaces.Tuple([
            spaces.Dict({
                'action_type': spaces.Discrete(2),
                'side': spaces.Discrete(2),
                'price': spaces.Box(low=0.0, high=1E9, shape=(1,), dtype=jnp.float32),
                'quantity': spaces.Box(low=0.0, high=1E9, shape=(1,), dtype=jnp.float32)
            }) for _ in range(self.n_actions)
        ])

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
            import numpy as np

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

            cubes_with_ob = nestlist_to_flatlist(sliced_cubes_with_ob_list)
            max_steps_in_episode_arr = jnp.array(
                [m.shape[0] for m, _ in cubes_with_ob], dtype=jnp.int32
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

            cubes_with_ob = cubes_with_ob_padding(cubes_with_ob)
            return cubes_with_ob, max_steps_in_episode_arr

        # Load the LOB data from source (e.g. crypto, LOBSTER)
        cubes_with_ob, max_steps_in_episode_arr = load_LOBSTER(
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
        self.messages = jnp.array([jnp.array(cube) for cube, _ in cubes_with_ob])
        # 2D Array: (n_windows x [4*n_depth])
        self.books = jnp.array([jnp.array(book) for _, book in cubes_with_ob])

        self.n_windows=len(self.books)

    def action_space(self, params: EnvParams) -> spaces.Box:
        return self.action_space_

    def observation_space(self, params: EnvParams) -> spaces.Dict:
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
        l2_ask_prices, l2_ask_quants = l2_state[:4*self.book_depth:4], l2_state[1:4*self.book_depth:4]
        l2_bid_prices, l2_bid_quants = l2_state[2:4*self.book_depth:4], l2_state[3:4*self.book_depth:4]

        l2_asks = jnp.zeros(2 * len(l2_ask_prices))
        l2_asks = l2_asks.at[::2].set(l2_ask_prices)
        l2_asks = l2_asks.at[1::2].set(l2_ask_quants)
        
        l2_bids = jnp.zeros(2 * len(l2_bid_prices))
        l2_bids = l2_bids.at[::2].set(l2_bid_prices)
        l2_bids = l2_bids.at[1::2].set(l2_bid_quants)

        best_ask = [l2_ask_prices[0], l2_ask_quants[0]]
        best_bid = [l2_bid_prices[0], l2_bid_quants[0]]

        best_asks = jnp.array([[0, 0]]*(self.lookback_window-1) + [best_ask])
        best_bids = jnp.array([[0, 0]]*(self.lookback_window-1) + [best_bid])
        # Initialize the environment state
        state = EnvState(
            ask_raw_orders=state.asks,
            bid_raw_orders=state.bids,
            trades=state.trades,
            best_asks=best_asks,
            best_bids=best_bids,
            l2_asks=jnp.hstack([l2_ask_prices, l2_ask_quants]).flatten(),
            l2_bids=jnp.hstack([l2_bid_prices, l2_bid_quants]).flatten(),
            init_time=0,
            time=jnp.array([0, 0]),
            custom_ID_counter=0,
            window_index=idx_data_window,
            cash=params.initial_cash,
            holdings=params.initial_holdings,
            active_orders=jnp.ones((self.max_active_orders, 8)) * -1,
            orders_state={
                "active_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "executed_limit_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "executed_market_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "failed_delete_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "invalid_delete_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "successful_delete_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "successfully_deleted_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "failed_cancel_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "invalid_cancel_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "successful_cancel_orders": jnp.ones((self.max_active_orders, 8)) * -1,
                "successfully_cancelled_orders": jnp.ones((self.max_active_orders, 8)) * -1,
            },
            mid_price=jnp.zeros(self.lookback_window),
            profit_history=jnp.zeros(self.lookback_window),
            step_counter=0,
            max_steps_in_episode=self.max_steps_in_episode_arr[idx_data_window],
        )

        obs = self.get_obs(state, params)
        # print(state)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams
    ) -> Tuple[Dict, EnvState, float, bool, dict]:        
        """Performs one step in the environment."""
        # each action is a tuple (type, side, price, quantity)
        # type: {0: Hold, 1: Limit, 2: Cancel, 3: Delete, 4: Market}
        # side: {-1: Ask, 1: Bid, 0 Hold}
        # price: must be 0 if it's a hold, or else it won't be
        #   converted to an empty slot in the orderbook by jaxlob

        # initialize variables
        cash = state.cash

        # get next set of data messages
        data_msgs = get_data_messages(
            params.message_data,
            state.window_index,
            state.step_counter
        )

        # process agent actions
        action_msgs, custom_id_counter_increment = self._get_action_msgs(action, state, params)

        # # deal with cancel (partial deletion) & delete (full deletion) messages
        # cancel_msgs, delete_msgs, action_msgs = self._get_cancel_delete_msgs(action, state)

        # submit latest trade data & actions to the orderbook
        all_msgs = jnp.concatenate(
            [
                # cancel_msgs,
                # delete_msgs,
                action_msgs,
                data_msgs
            ],
            axis=0
        )

        # get latest time for internal state
        latest_time = all_msgs[-1][-2:]
        state = state.replace(time=latest_time)

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
        l2_state = l2_state[-1]

        #### UPDATE STATE
        new_custom_id = state.custom_ID_counter + custom_id_counter_increment
        state = state.replace(custom_ID_counter=new_custom_id)
        l2_ask_prices, l2_ask_quants = l2_state[:4*self.book_depth:4], l2_state[1:4*self.book_depth:4]
        l2_bid_prices, l2_bid_quants = l2_state[2:4*self.book_depth:4], l2_state[3:4*self.book_depth:4]

        # Create arrays with alternating price and quantity elements
        l2_asks = jnp.zeros(2 * len(l2_ask_prices))
        l2_asks = l2_asks.at[::2].set(l2_ask_prices)
        l2_asks = l2_asks.at[1::2].set(l2_ask_quants)
        
        l2_bids = jnp.zeros(2 * len(l2_bid_prices))
        l2_bids = l2_bids.at[::2].set(l2_bid_prices)
        l2_bids = l2_bids.at[1::2].set(l2_bid_quants)
        state = state.replace(
            ask_raw_orders=state_.asks,
            bid_raw_orders=state_.bids,
            l2_asks=l2_asks,
            l2_bids=l2_bids,
            trades=state_.trades
        )

        best_asks = jnp.concatenate(
            [state.best_asks[1:], jnp.expand_dims(l2_asks[:2], 0)]
        )
        best_bids = jnp.concatenate(
            [state.best_bids[1:], jnp.expand_dims(l2_bids[:2], 0)]
        )
        state = state.replace(best_asks=best_asks, best_bids=best_bids)

        # get all non-empty trades (non [-1, -1, -1, -1, -1, -1] trades) 
        trades  = state.trades

        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)

        # filter for training agent's trades
        mask = (
            (
                (self.trader_unique_id < executed[:, 2])
                & (executed[:, 2] < (self.trader_unique_id + new_custom_id))
            )
            | (
                (self.trader_unique_id < executed[:, 3])
                & (executed[:, 3] < (self.trader_unique_id + new_custom_id))
            )
        )
        agent_trades = jnp.where(mask[:, jnp.newaxis], executed, 0)

        # update currently active orders, executed orders, failed orders, etc.
        orders_state = self._get_orders_state(state, action_msgs, agent_trades)
        state = state.replace(orders_state=orders_state)

        # get new cash, positions & mark-to-market
        profit = L2BaseLOBEnv._get_trade_profits(orders_state, agent_trades)

        cash = state.cash + profit
        holdings_delta_from_limit_executions = (
            orders_state['executed_limit_orders'][:, 2]
            * orders_state['executed_limit_orders'][:, 1]
        ).sum()
        holdings_delta_from_market_executions = (
            orders_state['executed_market_orders'][:, 2]
            * orders_state['executed_market_orders'][:, 1]
        ).sum()
        holdings_delta_from_executions = holdings_delta_from_limit_executions + holdings_delta_from_market_executions
        holdings = state.holdings + holdings_delta_from_executions
        state = state.replace(cash=cash, holdings=holdings)

        # Update prices history
        mid_price = (state.best_bids[0][0] + state.best_asks[0][0]) / 2
        ema_mid_price = self.ema_alpha*mid_price + (1-self.ema_alpha)*state.mid_price[-1]
        mid_price_history = jnp.concatenate([state.mid_price[1:], jnp.expand_dims(ema_mid_price, 0)])
        state = state.replace(mid_price=mid_price_history)

        # Calculate returns history
        profit_history = jnp.concatenate([state.profit_history[1:], jnp.expand_dims(profit, 0)])

        state = state.replace(
            profit_history=profit_history,
            step_counter=state.step_counter+1
        )

        reward = self.get_reward(state, params)

        # Check if episode is done
        done = self.is_terminal(state, params)

        # Get the new observation
        obs = self.get_obs(state, params)

        info = {
            'cash': state.cash,
            'holdings': state.holdings,
            'profit': profit,
            'reward': reward,
            'time': state.time,
            'step': state.step_counter,
        }
        # print(state)
        return obs, state, reward, done, info

    def get_reward(self, state: EnvState, params: EnvParams) -> float:
        """Calculates the reward for the current state."""
        if params.reward_functions is None:
            return self._default_reward_function(state, params)

        rewards = []
        for f, c in zip(params.reward_functions, params.reward_coefficients):
            rewards.append(f(state, params) * c)
        return jnp.array(rewards).sum()

    def _default_reward_function(self, state: EnvState, params: EnvParams) -> float:
        def sortino_reward(state: EnvState, params: EnvParams) -> float:
            # Calculate downside deviation
            # change returns history to be cash profit/cash_usage_history
            returns_history = jnp.diff(state.profit_history) / state.profit_history[:-1]
            negative_returns = jnp.minimum(returns_history - params.target_return, 0)
            downside_deviation = jnp.sqrt(jnp.mean(negative_returns ** 2) + 1e-8)

            # Calculate expected return
            expected_return = jnp.mean(returns_history)

            # Calculate Sortino ratio
            sortino_ratio = (expected_return - params.risk_free_rate) / downside_deviation
            reward = sortino_ratio
            return reward

        # Compute reward (Sortino ratio)
        reward = jax.lax.cond(
            state.step_counter >= self.lookback_window,
            lambda: sortino_reward(state, params),
            lambda: jnp.array(0.0)
        )
        return reward

    def get_obs(self, state: EnvState, params: EnvParams) -> Dict:
        # TODO: FIX THIS - SO MUCH WRONG, JUST WANNA GET IT STARTED
        """Generates the observation from the state."""

        # Extract Level-2 order book data up to K levels
        best_bids = state.l2_bids[:self.book_depth]
        best_asks = state.l2_asks[:self.book_depth]

        # Market features
        mid_prices = (
            ((best_asks + best_bids) // 2 // self.tick_size) * self.tick_size
        )
        spreads = state.l2_asks[self.book_depth:] - state.l2_bids[self.book_depth:]
        # ema = ...
        # sma = ...
        # std = ... # standard deviation
        # bb = ... # bollinger bands
        # lob_skew = ...
        # wmid_price_diff = ... # diff between mid-price and wmid-price
        imbalance = (
            state.l2_asks[:self.book_depth] - state.l2_bids[:self.book_depth]
        )

        # Internal features
        cash = state.cash
        inventory = state.holdings
        mark_to_market = cash + inventory * state.mid_price[-1]
        active_orders = state.active_orders

        # Other features
        time_of_day = state.time[0]
        delta_t = state.time[0] - state.init_time

        # Concatenate all features into a flat 1D array
        obs = jnp.concatenate([
            state.l2_bids,
            state.l2_asks,
            mid_prices,
            spreads,
            imbalance,
            jnp.array([cash]),
            jnp.array([inventory]),
            jnp.array([mark_to_market]),
            active_orders[:, 2],
            active_orders[:, 3],
            jnp.array([time_of_day]),
            jnp.array([delta_t]),
            jnp.array([state.step_counter/state.max_steps_in_episode]),
        ])

        # Normalize observations
        def obs_norm(obs):
            return jnp.concatenate([
                obs[:5*self.book_depth] / 3.5e7,  # Price levels
                obs[5*self.book_depth:6*self.book_depth] / 100000,  # Spreads
                obs[6*self.book_depth:7*self.book_depth] / 100,  # Shallow imbalance
                jnp.array([obs[7*self.book_depth] / 100000]),  # Cash
                jnp.array([obs[7*self.book_depth+1] / 10000]),  # Inventory
                jnp.array([obs[7*self.book_depth+2] / 100000]),  # Mark-To-Market
                obs[7*self.book_depth+3:-self.max_active_orders-3] / 3.5e7,  # Active Order Prices
                obs[-self.max_active_orders-3:-3] / 10000,  # Active Order Quantities
                jnp.array([obs[-3] / 1e9]),  # Time of day (nanoseconds)
                jnp.array([obs[-2] / 1e9]),  # Delta T (nanoseconds)
                jnp.array([obs[-1]]),  # Episode completion fraction
            ])

        obs = obs_norm(obs)

        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Checks whether the episode is done."""
        time_exceeded = (state.time - state.init_time)[0] > params.episode_time
        out_of_cash = state.cash <= 0
        return jnp.logical_or(time_exceeded, out_of_cash)
    
    def _get_action_msgs(self, action: chex.Array, state: EnvState, params: EnvParams) -> tuple[chex.Array, chex.Array]:
        """Returns action messages from the action."""
        # TODO: MAKE SURE TO CHECK THAT THE IDS OF CANCEL AND DELETE ORDERS ARE IDS OF ALREADY PENDING ORDERS
        (types, sides, prices, quants) = (
            jnp.array([a['action_type'] for a in action]),
            jnp.array([a['side'] for a in action]),
            jnp.concat([a['price'] for a in action]),
            jnp.concat([a['quantity'] for a in action])
        )

        types_cond = types != 0
        sorted_keys, sorted_types = jax.lax.sort_key_val(types_cond, types)

        # Prepare action messages
        num_non_hold_actions = jnp.sum(sorted_keys, -1) 

        types = sorted_types.astype(jnp.int32)
        sides = sides.astype(jnp.int32)
        trader_ids = jnp.ones((self.n_actions,), dtype=jnp.int32) * self.trader_unique_id
        order_ids = (
            jnp.ones((self.n_actions,), dtype=jnp.int32) * (self.trader_unique_id + state.custom_ID_counter)
            + jnp.arange(0, self.n_actions)
        )
        order_ids = jnp.where(sorted_keys, order_ids, self.trader_unique_id) # hold "orders" OID equal to TID
        times = jnp.resize(
            state.time + params.time_per_step, (self.n_actions, 2)
        )
        # Stack action message components
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1, dtype=jnp.int32)
        return action_msgs, num_non_hold_actions

    @staticmethod
    def _get_trade_profits(orders_state: Dict[str, chex.Array], agent_executed_trades: chex.Array) -> chex.Array:
        """Calculates the profit from executed limit and market orders."""
        executed_limit_orders = orders_state['executed_limit_orders']
        executed_market_orders = orders_state['executed_market_orders']

        # calculate profits from executed limit orders
        executed_limit_orders_pnl = executed_limit_orders[:, 2] * executed_limit_orders[:, 3] * -1
        limit_orders_profit = jnp.where(
            executed_limit_orders[:, 1] == -1,
            -executed_limit_orders_pnl,
            executed_limit_orders_pnl
        )
        limit_orders_profit = jnp.sum(limit_orders_profit)

        # calculate profits from executed market orders
        market_orders_profit = jax.vmap(
            lambda x: jnp.where(
                agent_executed_trades[:, 2] == x[4],
                agent_executed_trades[:, 0] * agent_executed_trades[:, 1],
                0.
            ).sum() * x[1] * -1
        )(executed_market_orders).sum()

        return limit_orders_profit + market_orders_profit

    def _get_orders_state(
        self,
        state: EnvState,
        action_msgs: chex.Array,
        executed_trades: chex.Array
    ) -> Dict[str, chex.Array]:
        """Updates agent's internal order state, including pending (active),
        partially-executed, and fully executed trades, returning a 
        
        Args:
            -
        """
        ########## HELPER FUNCTIONS ##########
        def _get_trade_quantities(active_order_id, executed_trades):
            # check if order ID matches either aggressive or standing side
            is_aggressor = executed_trades[:, 2] == active_order_id  # OIDa
            is_standing = executed_trades[:, 3] == active_order_id   # OIDs
            
            # get the executed quantity where there was a match
            # if no match found, return 0 or some sentinel value
            matched_quantity = jnp.where(is_aggressor | is_standing, 
                                        executed_trades[:, 1],  # quantity column
                                        0.).sum()
            return matched_quantity

        def _match_order_ids(
            active_order: chex.Array,
            executed_trades: chex.Array,
            order_type: int=None
        ) -> chex.Array:
            """Returns array containing bools for whether active_order_id
            was the aggressive or standing order in any of the given trades,
            in shape (2, executed_trades)
            """
            active_order_id = active_order[4]
            type_match = True
            if order_type is not None:
                active_order_type = active_order[0]
                type_match = active_order_type == order_type

            # check if order ID matches either aggressive or standing side
            is_aggressor = executed_trades[:, 2] == active_order_id  # OIDa
            is_standing = executed_trades[:, 3] == active_order_id   # OIDs
            
            return jnp.array([jnp.logical_and(is_aggressor, type_match), jnp.logical_and(is_standing, type_match)])

        def _remove_orders(mask: chex.Array, orders: chex.Array) -> chex.Array:
            # idxs = jnp.nonzero(mask, size=mask.sum(), fill_value=-1)[0] # TODO: fix this so we guarantee that we never truncate the idx array
            # return jnp.delete(orders, idxs, assume_unique_indices=True)
            return jnp.where(mask[:, None], orders, jnp.ones_like(orders) * -1)

        orders_state = {}

        # combine first, so that any limit orders that executed
        # instantaneously are accounted for

        # vectorize for all active orders
        match_quantities = jax.vmap(_get_trade_quantities, 
                                        in_axes=(0, None))
        match_trades = jax.vmap(_match_order_ids, in_axes=(0, None, None))

        ########## MARKET ORDERS ##########
        # market orders are (fully or partially) filled immediately or discarded,
        # so do not persist to time step t+1
        matched_trades = match_trades(action_msgs, executed_trades, 4) # shape (action_msgs, 2, executed_trades)
        mask_executed_market_orders = jnp.any(matched_trades, axis=(1, 2))
        executed_market_orders = _remove_orders(~mask_executed_market_orders, action_msgs)

        idxs = jnp.where(state.orders_state['executed_market_orders'][:, 0] == -1, size=self.n_actions, fill_value=-1)[0]
        orders_state['executed_market_orders'] = state.orders_state['executed_market_orders'].at[idxs].set(executed_market_orders)

        action_msgs = _remove_orders(mask_executed_market_orders, action_msgs)
        # active_orders = jnp.concat([state.active_orders, action_msgs])
        idxs = jnp.where(state.active_orders[:, 0] == -1, size=self.n_actions, fill_value=-1)[0]
        active_orders = state.active_orders.at[idxs].set(action_msgs)
        # account for:
        #   - partial executions and/or cancellations of limit orders (which
        #   means the order is still active)
        #   - market order execution (which should be deleted whether it is
        #   full match or not)

        ########## LIMIT ORDERS ##########
        matched_trades = match_trades(active_orders, executed_trades, 1)
        mask_executed_limit_orders = jnp.any(matched_trades, axis=(1,2))
        matched_quantities = jnp.where(mask_executed_limit_orders, 
                                    match_quantities(active_orders[:, 4], executed_trades),
                                    0.).sum()

        mask_fully_executed_limits = matched_quantities == active_orders[:, 2]
        #mask_partially_executed_limits = mask_executed_limit_orders & ~mask_fully_executed_limits

        # updated active orders with new quantities
        active_orders = active_orders.at[:, 2].set(active_orders[:, 2] - matched_quantities)

        # obtain executed limited order amounts via active orders dummy
        active_orders_ = active_orders.at[:, 2].set(matched_quantities)
        executed_limit_orders = _remove_orders(~mask_executed_limit_orders, active_orders_)

        orders_state['executed_limit_orders'] = executed_limit_orders

        # filter out fully executed limit orders
        active_orders = _remove_orders(mask_fully_executed_limits, active_orders)

        ########## DELETE ORDERS ##########
        # remove failed delete orders
        mask_deletes = active_orders[:, 0] == 3
        mask_deletes_of_executed_trades = jnp.any(
            match_trades(active_orders, executed_trades, 3),
            axis=(1,2)
        )
        # delete failed if present in executed_trades but not in current active orders
        # (i.e. the referenced trade fully executed), since our previous
        # logic removes fully executed orders
        mask_failed_and_invalid_delete_orders = jnp.logical_and(
            mask_deletes,
            jax.vmap(
                lambda x: (active_orders[:, 4] == x[4]).sum() == 1  # TODO: this currently returns true even if the
                # second order is another cancel or delete -> include condition to match only limit orders, i.e.
                # (if a limit order exists with this id, then...)
            )(active_orders)
        )

        mask_failed_delete_orders = jnp.logical_and(
            mask_deletes_of_executed_trades,
            mask_failed_and_invalid_delete_orders
        )

        mask_invalid_delete_orders = jnp.logical_and(
            ~mask_deletes_of_executed_trades,
            mask_failed_and_invalid_delete_orders
        )

        mask_successful_delete_orders = jnp.logical_and(
            mask_deletes,
            jax.vmap(
                lambda x: jnp.logical_and(
                    active_orders[:, 4] == x[4],
                    active_orders[:, 0] == 1
                ).any()
            )(active_orders)
        )

        mask_successfully_deleted_orders = jax.vmap(
            lambda a: jnp.where(
                a[0] == 1,
                jnp.logical_and(
                    active_orders[:, 4] == a[4],
                    mask_successful_delete_orders
                ).any(),
                False
            )
        )(active_orders)

        mask_all_delete_orders = (
            mask_failed_delete_orders
            | mask_invalid_delete_orders
            | mask_successful_delete_orders
            | mask_successfully_deleted_orders
        )


        failed_delete_orders = _remove_orders(~mask_failed_delete_orders, active_orders)
        invalid_delete_orders = _remove_orders(~mask_invalid_delete_orders, active_orders)        
        successful_delete_orders = _remove_orders(~mask_successful_delete_orders, active_orders)
        successfully_deleted_orders = _remove_orders(~mask_successfully_deleted_orders, active_orders)
        active_orders = _remove_orders(mask_all_delete_orders, active_orders)

        orders_state['failed_delete_orders'] = failed_delete_orders
        orders_state['invalid_delete_orders'] = invalid_delete_orders
        orders_state['successful_delete_orders'] = successful_delete_orders
        orders_state['successfully_deleted_orders'] = successfully_deleted_orders
        ########## CANCELLATION ORDERS ##########
        mask_cancels = active_orders[:, 0] == 2
        mask_cancels_of_executed_trades = jnp.any(
            match_trades(active_orders, executed_trades, 2),
            axis=(1,2)
        )

        mask_failed_and_invalid_cancel_orders = jnp.logical_and(
            mask_cancels,
            jax.vmap(
                lambda x: (active_orders[:, 4] == x[4]).sum() == 1  # TODO: this currently returns true even if the
                # second order is another cancel or delete -> include condition to match only limit orders, i.e. 
                # (if a limit order exists with this id, then...)
            )(active_orders)
        )

        mask_failed_cancel_orders = jnp.logical_and(
            mask_cancels_of_executed_trades,
            mask_failed_and_invalid_cancel_orders
        )

        mask_invalid_cancel_orders = jnp.logical_and(
            ~mask_cancels_of_executed_trades,
            mask_failed_and_invalid_cancel_orders
        )

        mask_successful_cancel_orders = jnp.logical_and(
            mask_cancels,
            jax.vmap(
                lambda x: jnp.logical_and(
                    active_orders[:, 4] == x[4],
                    active_orders[:, 0] == 1
                ).any()
            )(active_orders)
        )

        mask_successfully_cancelled_orders = jax.vmap(
            lambda a: jnp.where(
                a[0] == 1,
                jnp.logical_and(
                    active_orders[:, 4] == a[4],
                    mask_successful_cancel_orders
                ).any(),
                False
            )
        )(active_orders)

        mask_all_cancels = (
            mask_failed_cancel_orders
            | mask_invalid_cancel_orders
            | mask_successful_cancel_orders
            | mask_successfully_cancelled_orders
        )

        failed_cancel_orders = _remove_orders(~mask_failed_cancel_orders, active_orders)
        invalid_cancel_orders = _remove_orders(~mask_invalid_cancel_orders, active_orders)
        successful_cancel_orders = _remove_orders(~mask_successful_cancel_orders, active_orders)
        successfully_cancelled_orders = _remove_orders(~mask_successfully_cancelled_orders, active_orders)
        active_orders = _remove_orders(mask_all_cancels, active_orders)

        orders_state['failed_cancel_orders'] = failed_cancel_orders
        orders_state['invalid_cancel_orders'] = invalid_cancel_orders
        orders_state['successful_cancel_orders'] = successful_cancel_orders
        orders_state['successfully_cancelled_orders'] = successfully_cancelled_orders

        orders_state['active_orders'] = active_orders
        return orders_state

    @property
    def name(self) -> str:
        """Returns the name of the environment."""
        return "BaseL2LOBEnv-v0"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            self.messages,
            self.books,
        )
    # Additional methods for order matching, state updates, etc., would be implemented here.

# Usage example:
if __name__ == "__main__":
    # jax.config.update('jax_platform_name', 'cpu')
    
    # Create environment
    rng = jax.random.PRNGKey(0)
    data_path = "/Users/alishihab/projects/trading/research/reinforcement-learning/data/lobster/AAPL"
    env = L2BaseLOBEnv(
        data_path,
        n_actions=5,
        agent_active_order_limit=5,
        book_depth=10,
        ema_alpha=0.8,
        n_trades_logged=100,
        trades_reinit=True,
        n_orders_per_side=100,
        seconds_per_window=1800,
        start_time_in_seconds=34200,
        end_time_in_seconds=57600,
        step_lines=100,
        lookback_steps=20,
        window_index=-1
    )
    
    # Define a wrapper function that doesn't take env_params as input
    def make_run(env, env_params):
        def _run(rng):
            obs, state = env.reset(rng, env_params)
            for i in range(1):
                start = time.time()
                rng, action_rng = jax.random.split(rng)
                action = env.action_space(env_params).sample(action_rng)
                obs, state, reward, done, info = env.step(rng, state, action, env_params)
                jax.debug.print("Step: {step_counter}, time: {time}, Action: {action}", step_counter=state.step_counter, time=(time.time() - start), action=action)
            return state
        return _run
    
    # Get the parameterized function and JIT it
    run_jit = jax.jit(make_run(env, env.default_params))
    
    # Run the JIT-compiled function
    state = run_jit(rng)

