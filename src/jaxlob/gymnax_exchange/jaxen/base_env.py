import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
from flax import struct
from sims.AlphaTrade.gymnax_exchange.jaxob import JaxOrderBookArrays as job


@struct.dataclass
class EnvState:
    """
    Environment state dataclass for the Limit Order Book (LOB) environment.
    """
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    time: chex.Array
    custom_id_counter: int
    window_index: int
    step_counter: int
    max_steps_in_episode: int

@struct.dataclass
class EnvParams:
    """
    Environment parameters dataclass for the LOB environment.
    """
    message_data: chex.Array
    book_data: chex.Array
    episode_time: int = 60 * 30  # 60 seconds times 30 minutes = 1800 seconds
    time_per_step: int = 0  # 0 implies not to use time step
    time_delay_obs_act: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0, 0])
    )  # 0 ns time delay

class BaseLOBEnv(environment.Environment):
    """
    Base Limit Order Book Environment for simulating market microstructure dynamics.
    """
    def __init__(self, data_path):
        super().__init__()
        self.slice_time_window = 1800 # counted by seconds, 1800s=0.5h
        self.step_lines = 100
        self.message_path = f'{data_path}/trades/'
        self.ob_path = f'{data_path}/lobs/'
        self.start_time = 34200  # 09:30
        self.end_time = 57600  # 16:00
        self.n_orders_per_side = 100
        self.n_trades_logged = 100
        self.book_depth = 10
        self.n_actions = 3
        self.custom_id_counter =0
        self.trader_unique_id = -9000+1
        self.tick_size = 100



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

        # ==================================================================
        # ================= CAUTION NOT BELONG TO BASE ENV =================
        # ================= EPECIALLY SUPPORT FOR EXEC ENV =================
        print("START:  pre-reset in the initialization")

        n_orders_per_side = self.n_orders_per_side
        n_trades_logged = self.n_trades_logged
        tick_size = self.tick_size
        step_lines = self.step_lines
        task_size = 200
        n_ticks_in_book = 20

        def get_state(message_data, book_data,max_steps_in_episode):
            """
            Constructs the initial state of the environment from message and book data.

            Args:
                message_data: Message data array.
                book_data: Book data array.
                max_steps_in_episode: Maximum steps in the episode.

            Returns:
                The initial state tuple.
            """
            time=jnp.array(message_data[0, 0, -2:])
            # Get initial orders ( 2 x Ndepth) x 6 based on the initial L2 orderbook for this window
            def get_initial_orders(book_data, time):
                orderbook_levels = 10
                init_id = -9000
                data = jnp.array(book_data).reshape(int(10 * 2), 2)
                new_arr = jnp.zeros((int(orderbook_levels * 2), 8), dtype=jnp.int32)
                init_ob = (
                    new_arr.at[:, 3]
                    .set(data[:, 0])
                    .at[:, 2]
                    .set(data[:, 1])
                    .at[:, 0]
                    .set(1)
                    .at[0 : orderbook_levels * 4 : 2, 1]
                    .set(-1)
                    .at[1 : orderbook_levels * 4 : 2, 1]
                    .set(1)
                    .at[:, 4]
                    .set(init_id)
                    .at[:, 5]
                    .set(init_id - jnp.arange(0, orderbook_levels * 2))
                    .at[:, 6]
                    .set(time[0])
                    .at[:, 7]
                    .set(time[1])
                )
                return init_ob

            init_orders = get_initial_orders(book_data, time)

            # Initialize both sides of the book as empty
            asks_raw = job.init_orderside(n_orders_per_side)
            bids_raw = job.init_orderside(n_orders_per_side)
            trades_init = (jnp.ones((n_trades_logged, 6)) * -1).astype(jnp.int32)

            # Process the initial messages through the order book
            ordersides = job.scan_through_entire_array(
                init_orders, (asks_raw, bids_raw, trades_init)
            )

            # Mid Price after init added to env state as the initial price
            best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(
                ordersides[0], ordersides[1]
            )
            mid_price = (best_bid[0] + best_ask[0]) // 2 // tick_size * tick_size

            state = (
                ordersides[0],
                ordersides[1],
                ordersides[2],
                jnp.resize(best_ask, (step_lines, 2)),
                jnp.resize(best_bid, (step_lines, 2)),
                time,
                time,
                0,
                -1,
                mid_price,
                task_size,
                0,
                0,
                0,
                0,
                max_steps_in_episode,
            )

            return state

        def get_obs(state):
            """
            Generates observations for both sell and buy sides.

            Args:
                state: Current state tuple.

            Returns:
                A tuple containing normalized observations for sell and buy sides.
            """
            # Extract best asks and bids
            best_asks = state[3][:, 0]
            best_bids = state[4][:, 0]
            mid_prices = (best_asks + best_bids) // 2 // tick_size * tick_size
            second_passives_sell_task = best_asks + tick_size * n_ticks_in_book
            second_passives_buy_task = best_bids - tick_size * n_ticks_in_book
            spreads = best_asks - best_bids

            # Time information
            time_of_day = state[6]
            delta_t = time_of_day - state[5]

            # Price drift
            init_price = state[9]
            price_drift = mid_prices[-1] - init_price

            # Task execution
            task_size = state[10]
            executed_quant = state[11]

            # Shallow imbalance
            best_asks_qtys = state[3][:, 1]
            best_bids_qtys = state[4][:, 1]
            shallow_imbalance = best_asks_qtys - best_bids_qtys

            # Step counters
            step_counter = 0
            max_steps_in_episode = state[-1]

            # Combine observations
            obs_sell = jnp.concatenate(
                (
                    best_bids,
                    best_asks,
                    mid_prices,
                    second_passives_sell_task,
                    spreads,
                    time_of_day,
                    delta_t,
                    jnp.array([init_price]),
                    jnp.array([price_drift]),
                    jnp.array([task_size]),
                    jnp.array([executed_quant]),
                    shallow_imbalance,
                    jnp.array([step_counter]),
                    jnp.array([max_steps_in_episode]),
                )
            )
            obs_buy = jnp.concatenate(
                (
                    best_bids,
                    best_asks,
                    mid_prices,
                    second_passives_buy_task,
                    spreads,
                    time_of_day,
                    delta_t,
                    jnp.array([init_price]),
                    jnp.array([price_drift]),
                    jnp.array([task_size]),
                    jnp.array([executed_quant]),
                    shallow_imbalance,
                    jnp.array([step_counter]),
                    jnp.array([max_steps_in_episode]),
                )
            )

            # Normalize observations
            def obs_norm(obs):
                return jnp.concatenate(
                    (
                        obs[:400] / 3.5e7,  # Price levels
                        obs[400:500] / 100000,  # Spreads
                        obs[500:501] / 100000,  # Time of day (seconds)
                        obs[501:502] / 1e9,  # Time of day (nanoseconds)
                        obs[502:503] / 10,  # Delta T (seconds)
                        obs[503:504] / 1e9,  # Delta T (nanoseconds)
                        obs[504:505] / 3.5e7,  # Initial price
                        obs[505:506] / 100000,  # Price drift
                        obs[506:507] / 500,  # Task size
                        obs[507:508] / 500,  # Executed quantity
                        obs[508:608] / 100,  # Shallow imbalance
                        obs[608:609] / 300,  # Step counter
                        obs[609:610] / 300,  # Max steps in episode
                    )
                )

            return obs_norm(obs_sell), obs_norm(obs_buy)

        # Function to get the state and observation
        def get_state_obs(message_data, book_data, max_steps_in_episode):
            state = get_state(message_data, book_data, max_steps_in_episode)
            obs_sell, obs_buy = get_obs(state)
            return state, obs_sell, obs_buy

        state_obs = [
            get_state_obs(
                cubes_with_OB[i][0],
                cubes_with_OB[i][1],
                max_steps_in_episode_arr[i]
            ) for i in range(len(max_steps_in_episode_arr))
        ]

        # Convert states to arrays
        def state_to_array(state):
            state_5 = jnp.hstack((state[5], state[6], state[9], state[15]))
            padded_state = jnp.pad(
                state_5, (0, 100 - state_5.shape[0]), constant_values=-1
            )[:, jnp.newaxis]
            state_array = jnp.hstack(
                (state[0], state[1], state[2], state[3], state[4], padded_state)
            )
            return state_array

        self.state_array_list = jnp.array(
            [state_to_array(state) for state, _, _ in state_obs]
        )
        self.obs_sell_list = jnp.array([obs_sell for _, obs_sell, _ in state_obs])
        self.obs_buy_list = jnp.array([obs_buy for _, _, obs_buy in state_obs])

        print("Finished pre-reset in the initialization")


        print(f"Num of data_window: {self.n_windows}")

    @property
    def default_params(self) -> EnvParams:
        """
        Returns the default parameters for the environment.
        """
        return EnvParams(self.messages, self.books)


    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Dict,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
        Performs one step in the environment using the given action.

        Args:
            key: JAX random key. <- Not currently used.
            state: Current state of the environment.
            action: Action taken by the agent.
            params: Environment parameters.

        Returns:
            A tuple containing the new observation, updated state, reward, done flag, and info dictionary.
        """
        # Obtain the messages for the step from the message data
        data_messages = job.get_data_messages(
            params.message_data, state.window_index, state.step_counter
        )

        # Prepare action messages
        types = jnp.ones((self.n_actions,), dtype=jnp.int32)
        sides = ((action["sides"] + 1) / 2).astype(jnp.int32)
        prices = action["prices"]
        quants = action["quantities"]
        trader_ids = jnp.ones((self.n_actions,), dtype=jnp.int32) * self.trader_unique_id
        order_ids = (
            jnp.ones((self.n_actions,), dtype=jnp.int32) * (self.trader_unique_id + state.custom_id_counter)
            + jnp.arange(0, self.n_actions)
        )
        times = jnp.resize(
            state.time + params.time_delay_obs_act, (self.n_actions, 2)
        )
        # Stack action message components
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        # Combine action messages with data messages
        total_messages = jnp.concatenate([action_msgs, data_messages], axis=0)

        # Save time of final message to add to state
        time = total_messages[-1][6:8]

        # Process messages through the order book
        ordersides = job.scan_through_entire_array(
            total_messages, (state.ask_raw_orders, state.bid_raw_orders, state.trades)
        )

        # Update state
        state = EnvState(
            ask_raw_orders=ordersides[0],
            bid_raw_orders=ordersides[1],
            trades=ordersides[2],
            init_time=state.init_time,
            time=time,
            custom_id_counter=state.custom_id_counter + self.n_actions,
            window_index=state.window_index,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
        )
        done = self.is_terminal(state, params)
        reward = 0

        return self.get_obs(state), state, reward, done, {"info": 0}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """
        Resets the environment to an initial state.

        Args:
            key: JAX random key.
            params: Environment parameters.

        Returns:
            A tuple containing the initial observation and the initial state.
        """
        idx_data_window = jax.random.randint(
            key, minval=0, maxval=self.n_windows, shape=()
        )

        # Get the initial time
        time = job.get_initial_time(params.message_data, idx_data_window)

        # Get initial orders based on the initial L2 orderbook for this window
        init_orders = job.get_initial_orders(params.book_data, idx_data_window, time)

        # Initialize both sides of the book as empty
        asks_raw = job.init_orderside(self.n_orders_per_side)
        bids_raw = job.init_orderside(self.n_orders_per_side)
        trades_init = (jnp.ones((self.n_trades_logged, 6)) * -1).astype(jnp.int32)

        # Process the initial messages through the order book
        ordersides = job.scan_through_entire_array(
            init_orders, (asks_raw, bids_raw, trades_init)
        )

        # Create the initial state
        state = EnvState(
            ask_raw_orders=ordersides[0],
            bid_raw_orders=ordersides[1],
            trades=ordersides[2],
            init_time=time,
            time=time,
            custom_id_counter=0,
            window_index=idx_data_window,
            step_counter=0,
            max_steps_in_episode=self.max_steps_in_episode_arr[idx_data_window],
        )

        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Checks whether the current state is terminal.

        Args:
            state: Current state of the environment.
            params: Environment parameters.

        Returns:
            A boolean indicating whether the state is terminal.
        """
        return (state.time - state.init_time)[0] > params.episode_time

    def get_obs(self, state: EnvState) -> chex.Array:
        """
        Generates an observation from the current state.

        Args:
            state: Current state of the environment.

        Returns:
            An array representing the observation.
        """
        return job.get_L2_state(
            self.book_depth, state.ask_raw_orders, state.bid_raw_orders
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the environment.
        """
        return "alphatradeBase-v0"

    @property
    def num_actions(self) -> int:
        """
        Returns the number of possible actions in the environment.
        """
        return self.n_actions

    def action_space(self) -> spaces.Dict:
        """
        Defines the action space of the environment.

        Args:
            params: Environment parameters (optional).

        Returns:
            A gymnax space object representing the action space.
        """
        return spaces.Dict(
            {
                "sides": spaces.Box(
                    low=0, high=2, shape=(self.n_actions,), dtype=jnp.int32
                ),
                "quantities": spaces.Box(
                    low=0, high=100, shape=(self.n_actions,), dtype=jnp.int32
                ),
                "prices": spaces.Box(
                    low=0, high=99999999, shape=(self.n_actions,), dtype=jnp.int32
                ),
            }
        )

    # TODO: define obs space ( 4 x Ndepth) array of quants & prices.
    # Not that important right now.
    def observation_space(self, params: EnvParams):
        """
        Defines the observation space of the environment.

        Args:
            params: Environment parameters.

        Returns:
            The observation space of the environment.
        """
        raise NotImplementedError("Observation space is not defined.")

    # FIXME: Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5)
    # fields in the bid/ask arrays to return something of value. Not sure if actually needed.
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """
        Defines the state space of the environment.

        Args:
            params: Environment parameters.

        Returns:
            A gymnax space object representing the state space.
        """
        return spaces.Dict(
            {
                "bids": spaces.Box(
                    low=-1,
                    high=999999999,
                    shape=(6, self.n_orders_per_side),
                    dtype=jnp.int32,
                ),
                "asks": spaces.Box(
                    low=-1,
                    high=999999999,
                    shape=(6, self.n_orders_per_side),
                    dtype=jnp.int32,
                ),
                "trades": spaces.Box(
                    low=-1, high=999999999, shape=(6, self.n_trades_logged), dtype=jnp.int32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
