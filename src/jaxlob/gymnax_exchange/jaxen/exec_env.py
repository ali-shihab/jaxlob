import sys
import time
import random
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
import chex

from sims.AlphaTrade.gymnax_exchange.jaxob import JaxOrderBookArrays as job
from .base_env import BaseLOBEnv


@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index: int
    init_price: int
    task_to_execute: int
    quant_executed: int
    total_revenue: int
    step_counter: int
    max_steps_in_episode: int
    slippage_rm: int
    price_adv_rm: int
    price_drift_rm: int
    vwap_rm: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    state_array_list: chex.Array
    obs_sell_list: chex.Array
    obs_buy_list: chex.Array
    episode_time: int = 60 * 30  # 1800 seconds
    time_per_step: int = 0  # 0 ns time delay
    time_delay_obs_act: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0, 0])
    )
    avg_twap_list: chex.Array = struct.field(
        default_factory=lambda: jnp.array([
            312747.47, 312674.06, 313180.38, 312813.25, 312763.78,
            313094.1, 313663.97, 313376.72, 313533.25, 313578.9,
            314559.1, 315201.1, 315190.2
        ])
    )


class ExecutionEnv(BaseLOBEnv):
    def __init__(
        self,
        data_path: str,
        task: str,
        window_index: int,
        action_type: str,
        task_size: int = 500,
        reward_lambda: float = 0.0,
        gamma: float = 0.00
    ):
        super().__init__(data_path)
        self.n_actions = 4  # [FT, M, NT, PP]
        self.task = task
        self.window_index = window_index
        self.action_type = action_type
        self.reward_lambda = reward_lambda
        self.gamma = gamma
        self.task_size = task_size
        self.n_fragment_max = 2
        self.n_ticks_in_book = 2

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            self.messages,
            self.books,
            self.state_array_list,
            self.obs_sell_list,
            self.obs_buy_list
        )

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        delta: Dict,
        params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs one step in the environment."""
        action = self._reshape_action(key, delta, state, params)
        data_messages = job.get_data_messages(
            params.message_data, state.window_index, state.step_counter
        )
        action_msgs = self._get_action_msgs(action, state, params)
        cnl_msgs = job.getCancelMsgs(
            state.ask_raw_orders if self.task == 'sell' else state.bid_raw_orders,
            -8999,
            self.n_actions,
            -1 if self.task == 'sell' else 1
        )
        total_messages = jnp.concatenate(
            [cnl_msgs, action_msgs, data_messages], axis=0
        )
        time_ = total_messages[-1][-2:]
        trades_reinit = (
            jnp.ones((self.n_trades_logged, 6)) * -1
        ).astype(jnp.int32)
        asks, bids, trades, bestasks, bestbids = (
            job.scan_through_entire_array_save_bidask(
                total_messages,
                (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
                self.step_lines
            )
        )
        reward, state = self._calculate_reward_and_update_state(
            state, asks, bids, trades, bestasks, bestbids, params
        )
        done = self.is_terminal(state, params)
        info = {
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": state.total_revenue / state.quant_executed,
            "current_step": state.step_counter,
            'done': done,
            'slippage_rm': state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": reward * 10000,  # Denormalize for logging
        }
        return self.get_obs(state, params), state, reward, done, info

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Resets the environment to an initial state."""
        idx_data_window = (
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())
            if self.window_index == -1 else
            jnp.array(self.window_index, dtype=jnp.int32)
        )
        state_array = params.state_array_list[idx_data_window]
        state = EnvState(
            *self._state_array_to_state(state_array, idx_data_window)
        )
        obs_sell = params.obs_sell_list[idx_data_window]
        obs_buy = params.obs_buy_list[idx_data_window]
        obs = obs_sell if self.task == "sell" else obs_buy
        return obs, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Checks whether the episode is done."""
        time_exceeded = (state.time - state.init_time)[0] > params.episode_time
        task_completed = state.task_to_execute - state.quant_executed <= 0
        return jnp.logical_or(time_exceeded, task_completed)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Returns the action space of the environment."""
        if self.action_type == 'delta':
            return spaces.Box(
                -5, 5, (self.n_actions,), dtype=jnp.int32
            )
        else:
            return spaces.Box(
                0, 100, (self.n_actions,), dtype=jnp.int32
            )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Returns the observation space of the environment."""
        return spaces.Box(-10, 10, (610,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Returns the state space of the environment."""
        return spaces.Dict({
            "bids": spaces.Box(
                -1, job.MAXPRICE, shape=(6, self.n_orders_per_side),
                dtype=jnp.int32
            ),
            "asks": spaces.Box(
                -1, job.MAXPRICE, shape=(6, self.n_orders_per_side),
                dtype=jnp.int32
            ),
            "trades": spaces.Box(
                -1, job.MAXPRICE, shape=(6, self.n_trades_logged),
                dtype=jnp.int32
            ),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })

    @property
    def name(self) -> str:
        """Returns the name of the environment."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Returns the number of actions possible in the environment."""
        return self.n_actions

    # Internal helper methods

    def _reshape_action(
        self,
        key: chex.PRNGKey,
        delta: Dict,
        state: EnvState,
        params: EnvParams
    ) -> jnp.ndarray:
        action_space_clipping = (
            lambda action_, task_size: (
                jnp.round((action_ - 0.5) * task_size).astype(jnp.int32)
                if self.action_type == 'delta' else
                jnp.round(action_ * task_size).astype(jnp.int32).clip(0, task_size)
            )
        )

        def twapV3(state_, env_params):
            remaining_time = (
                env_params.episode_time - (state_.time - state_.init_time)[0]
            )
            market_order_time = 60  # Last minute
            if_market_order = remaining_time <= market_order_time
            remained_quant = (
                state_.task_to_execute - state_.quant_executed
            )
            remained_steps = (
                state_.max_steps_in_episode - state_.step_counter
            )
            step_quant = jnp.ceil(
                remained_quant / remained_steps
            ).astype(jnp.int32)
            limit_quants = jax.random.permutation(
                key, jnp.array([step_quant - step_quant // 2,
                                step_quant // 2]),
                independent=True
            )
            market_quants = jnp.array([step_quant, step_quant])
            quants = jnp.where(if_market_order, market_quants, limit_quants)
            return quants

        get_base_action = lambda state_, params_: twapV3(state_, params_)

        def truncate_action(action_, remain_quant):
            action_ = jnp.round(action_).astype(jnp.int32).clip(0, self.task_size)
            scaled_action = jnp.where(
                action_.sum() > remain_quant,
                (action_ * remain_quant / action_.sum()).astype(jnp.int32),
                action_
            )
            return scaled_action

        base_action = get_base_action(state, params)
        action_ = (
            base_action + action_space_clipping(delta, state.task_to_execute)
            if self.action_type == 'delta' else
            action_space_clipping(delta, state.task_to_execute)
        )
        action = truncate_action(
            action_, state.task_to_execute - state.quant_executed
        )
        return action

    def _get_action_msgs(
        self,
        action: Dict,
        state: EnvState,
        params: EnvParams
    ) -> chex.Array:
        types = jnp.ones((self.n_actions,), jnp.int32)
        sides = (
            -1 * jnp.ones((self.n_actions,), jnp.int32)
            if self.task == 'sell' else
            jnp.ones((self.n_actions,), jnp.int32)
        )
        trader_ids = (
            jnp.ones((self.n_actions,), jnp.int32) * self.trader_unique_id
        )
        order_ids = (
            jnp.ones((self.n_actions,), jnp.int32) *
            (self.trader_unique_id + state.customIDcounter) +
            jnp.arange(0, self.n_actions)
        )
        times = jnp.resize(
            state.time + params.time_delay_obs_act, (self.n_actions, 2)
        )

        # Determine prices
        best_ask, best_bid = (
            state.best_asks[-1, 0], state.best_bids[-1, 0]
        )
        FT = best_bid if self.task == 'sell' else best_ask
        M = (
            ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size
        )
        NT = best_ask if self.task == 'sell' else best_bid
        PP = (
            best_ask + self.tick_size * self.n_ticks_in_book
            if self.task == 'sell' else
            best_bid - self.tick_size * self.n_ticks_in_book
        )
        MKT = 0 if self.task == 'sell' else job.MAX_INT

        # Determine if market order
        remaining_time = (
            params.episode_time - (state.time - state.init_time)[0]
        )
        market_order_time = 60
        if_market_order = remaining_time <= market_order_time

        def normal_order_logic(state_, action_):
            quants = action_.astype(jnp.int32)
            prices = jnp.array([FT, M, NT, PP], jnp.int32)
            return quants, prices

        def market_order_logic(state_):
            quant = state_.task_to_execute - state_.quant_executed
            quants = jnp.array([quant, 0, 0, 0], jnp.int32)
            prices = jnp.array([MKT, M, M, M], jnp.int32)
            return quants, prices

        market_quants, market_prices = market_order_logic(state)
        normal_quants, normal_prices = normal_order_logic(state, action)
        quants = jnp.where(if_market_order, market_quants, normal_quants)
        prices = jnp.where(if_market_order, market_prices, normal_prices)

        action_msgs = jnp.stack(
            [types, sides, quants, prices, trader_ids, order_ids],
            axis=1
        )
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        return action_msgs

    def _calculate_reward_and_update_state(
        self,
        state: EnvState,
        asks: chex.Array,
        bids: chex.Array,
        trades: chex.Array,
        bestasks: chex.Array,
        bestbids: chex.Array,
        params: EnvParams
    ) -> Tuple[float, EnvState]:
        executed = jnp.where(
            (trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0
        )
        mask = (
            ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) |
            ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        )
        agent_trades = jnp.where(mask[:, jnp.newaxis], executed, 0)
        agent_trades = self._truncate_agent_trades(
            agent_trades, state.task_to_execute - state.quant_executed
        )
        new_execution = agent_trades[:, 1].sum()
        revenue = (
            (agent_trades[:, 0] // self.tick_size * agent_trades[:, 1]).sum()
        )
        agent_quant = agent_trades[:, 1].sum()
        vwap = self._calculate_vwap(executed)
        vwap_rm = self._rolling_mean_value(
            state.vwap_rm, vwap, state.step_counter
        )
        advantage = revenue - vwap_rm * agent_quant
        reward = jnp.sign(agent_quant) * advantage / 10000  # Normalize reward

        # Update state
        bestasks, bestbids = (
            self._impute_best_prices(
                bestasks[-self.step_lines:], state.best_asks[-1, 0]
            ),
            self._impute_best_prices(
                bestbids[-self.step_lines:], state.best_bids[-1, 0]
            )
        )

        # Compute price_adv_value using jnp.where
        price_adv_value = jnp.where(
            agent_quant > 0,
            revenue // agent_quant - vwap,
            state.price_adv_rm
        )

        # Update rolling mean values
        price_adv_rm = self._rolling_mean_value(
            state.price_adv_rm,
            price_adv_value,
            state.step_counter
        )

        slippage_rm = self._rolling_mean_value(
            state.slippage_rm,
            revenue - state.init_price // self.tick_size * agent_quant,
            state.step_counter
        )

        price_drift_rm = self._rolling_mean_value(
            state.price_drift_rm,
            vwap - state.init_price // self.tick_size,
            state.step_counter
        )

        state = EnvState(
            ask_raw_orders=asks,
            bid_raw_orders=bids,
            trades=trades,
            best_asks=bestasks,
            best_bids=bestbids,
            init_time=state.init_time,
            time=state.time,
            customIDcounter=state.customIDcounter + self.n_actions,
            window_index=state.window_index,
            init_price=state.init_price,
            task_to_execute=state.task_to_execute,
            quant_executed=state.quant_executed + new_execution,
            total_revenue=state.total_revenue + revenue,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
            slippage_rm=slippage_rm,
            price_adv_rm=price_adv_rm,
            price_drift_rm=price_drift_rm,
            vwap_rm=vwap_rm
        )
        return reward, state

    def _truncate_agent_trades(
        self,
        agent_trades: chex.Array,
        remain_quant: int
    ) -> chex.Array:
        quantities = agent_trades[:, 1]
        cumsum_quantities = jnp.cumsum(quantities)
        cut_idx = jnp.argmax(cumsum_quantities >= remain_quant)
        truncated_trades = jnp.where(
            jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx,
            jnp.zeros_like(agent_trades[0]),
            agent_trades.at[:, 1].set(
                jnp.where(
                    jnp.arange(len(quantities)) < cut_idx,
                    quantities,
                    jnp.where(
                        jnp.arange(len(quantities)) == cut_idx,
                        remain_quant - cumsum_quantities[cut_idx - 1],
                        0
                    )
                )
            )
        )
        result = jnp.where(
            remain_quant >= jnp.sum(quantities),
            agent_trades,
            jnp.where(
                remain_quant <= quantities[0],
                jnp.zeros_like(agent_trades).at[0, :].set(
                    agent_trades[0]
                ).at[0, 1].set(remain_quant),
                truncated_trades
            )
        )
        return result

    def _calculate_vwap(self, executed: chex.Array) -> int:
        return (
            (executed[:, 0] // self.tick_size * executed[:, 1]).sum() //
            executed[:, 1].sum()
        )

    def _rolling_mean_value(
        self, avg_price: int, new_price: int, step_counter: int
    ) -> int:
        return (
            (avg_price * step_counter + new_price) / (step_counter + 1)
        ).astype(jnp.int32)

    def _impute_best_prices(
        self, best_prices: chex.Array, last_best_price: int
    ) -> chex.Array:
        def replace_values(prev, curr):
            last_valid = jnp.where(curr != 999999999, curr, prev)
            replaced_curr = jnp.where(curr == 999999999, last_valid, curr)
            return last_valid, replaced_curr

        def forward_fill(arr):
            index = jnp.argmax(arr[:, 0] != 999999999)
            arr = arr.at[0, 0].set(
                jnp.where(index == 0, arr[0, 0], arr[index][0])
            )
            last_valid, replaced = jax.lax.scan(
                replace_values, arr[0], arr[1:]
            )
            return jnp.concatenate([arr[:1], replaced])

        back_fill = lambda arr: jnp.flip(
            forward_fill(jnp.flip(arr, axis=0)), axis=0
        )
        mean_fill = lambda arr: (
            forward_fill(arr) + back_fill(arr)
        ) // 2

        return jnp.where(
            (best_prices[:, 0] == 999999999).all(),
            jnp.tile(
                jnp.array([last_best_price, 0]), (best_prices.shape[0], 1)
            ),
            mean_fill(best_prices)
        )

    def _state_array_to_state(
        self,
        state_array: chex.Array,
        idx_data_window: int
    ) -> Tuple:
        state0 = state_array[:, 0:6]
        state1 = state_array[:, 6:12]
        state2 = state_array[:, 12:18]
        state3 = state_array[:, 18:20]
        state4 = state_array[:, 20:22]
        state5 = state_array[0:2, 22:23].squeeze(axis=-1)
        state6 = state_array[2:4, 22:23].squeeze(axis=-1)
        state9 = state_array[4:5, 22:23][0].squeeze(axis=-1)
        return (
            state0,
            state1,
            state2,
            state3,
            state4,
            state5,
            state6,
            0,
            idx_data_window,
            state9,
            self.task_size,
            0,
            0,
            0,
            self.max_steps_in_episode_arr[idx_data_window],
            0,
            0,
            0,
            0
        )

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Generates the observation from the state."""
        best_asks = state.best_asks[:, 0]
        best_bids = state.best_bids[:, 0]
        mid_prices = (
            ((best_asks + best_bids) // 2 // self.tick_size) * self.tick_size
        )
        second_passives = (
            best_asks + self.tick_size * self.n_ticks_in_book
            if self.task == 'sell' else
            best_bids - self.tick_size * self.n_ticks_in_book
        )
        spreads = best_asks - best_bids
        time_of_day = state.time
        delta_t = state.time - state.init_time
        init_price = state.init_price
        price_drift = mid_prices[-1] - state.init_price
        task_size = state.task_to_execute
        executed_quant = state.quant_executed
        shallow_imbalance = (
            state.best_asks[:, 1] - state.best_bids[:, 1]
        )

        obs = jnp.concatenate([
            best_bids,
            best_asks,
            mid_prices,
            second_passives,
            spreads,
            time_of_day,
            delta_t,
            jnp.array([init_price]),
            jnp.array([price_drift]),
            jnp.array([task_size]),
            jnp.array([executed_quant]),
            shallow_imbalance,
            jnp.array([state.step_counter]),
            jnp.array([state.max_steps_in_episode])
        ])

        obs_norm = self._normalize_obs(obs)
        return obs_norm

    def _normalize_obs(self, obs: chex.Array) -> chex.Array:
        return jnp.concatenate([
            obs[:400] / 3.5e7,         # Normalize prices
            obs[400:500] / 100000,     # Normalize spreads
            obs[500:501] / 100000,     # Normalize timeOfDay
            obs[501:502] / 1e9,        # Normalize timeOfDay
            obs[502:503] / 10,         # Normalize deltaT
            obs[503:504] / 1e9,        # Normalize deltaT
            obs[504:505] / 3.5e7,      # Normalize initPrice
            obs[505:506] / 100000,     # Normalize priceDrift
            obs[506:507] / 500,        # Normalize taskSize
            obs[507:508] / 500,        # Normalize executed_quant
            obs[508:608] / 100,        # Normalize shallowImbalance
            obs[608:609] / 300,        # Normalize step_counter
            obs[609:610] / 300,        # Normalize max_steps_in_episode
        ])


# Main execution for testing
if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except IndexError:
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        "TASK_SIZE": 500,
        "WINDOW_INDEX": 0,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 0.0,
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env = ExecutionEnv(
        config["ATFOLDER"],
        config["TASKSIDE"],
        config["WINDOW_INDEX"],
        config["ACTION_TYPE"],
        config["TASK_SIZE"],
        config["REWARD_LAMBDA"]
    )
    env_params = env.default_params

    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print("Time for reset:", time.time() - start)

    for i in range(1, 100000):
        key_policy, _ = jax.random.split(key_policy, 2)
        test_action = env.action_space().sample(
            key_policy
        ) * random.randint(1, 10)
        start = time.time()
        obs, state, reward, done, info = env.step(
            key_step, state, test_action, env_params
        )
        if done:
            print("=" * 60)

    # Testing vmap capabilities
    enable_vmap = True
    if enable_vmap:
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample = jax.vmap(
            env.action_space().sample, in_axes=(0)
        )

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions = vmap_act_sample(vmap_keys)
        print(test_actions)

        start = time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print(
            "Time for vmap reset with", num_envs,
            "environments:", time.time() - start
        )

        start = time.time()
        n_obs, n_state, reward, done, _ = vmap_step(
            vmap_keys, state, test_actions, env_params
        )
        print(
            "Time for vmap step with", num_envs,
            "environments:", time.time() - start
        )
