import gym
from gym import spaces
import numpy as np

class PowerManagementEnv(gym.Env):
    def __init__(self):
        super(PowerManagementEnv, self).__init__()

        self.max_soc = 100
        self.min_soc = 0
        self.max_res = 100
        self.max_demand = 50
        self.max_soh = 1.0
        self.capacity = 50 
        self.grid_price = 0.3  
        self.battery_price = 200  
        self.cycle_life = 5000
        self.alpha = 0.5  
        self.lambda_ = 0.3 

        self.action_space = spaces.MultiDiscrete([2, 2])  # main and support actions

        self.main_obs_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.max_soc, self.max_demand, self.max_res], dtype=np.float32)
        )

        self.support_obs_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0], dtype=np.float32),
            high=np.array([self.max_soc, self.max_demand, self.max_res, self.max_soh], dtype=np.float32)
        )

        self.max_steps = 100
        self.reset()

    def reset(self):
        self.main_soc = 50
        self.support_soc = 50
        self.support_soh = 1.0
        self.energy_demand = np.random.randint(10, 30)
        self.res_amount = 50

        self.prev_main_soc = self.main_soc
        self.prev_support_soc = self.support_soc

        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            'main': np.array([self.main_soc, self.energy_demand, self.res_amount], dtype=np.float32),
            'support': np.array([self.support_soc, self.energy_demand, self.res_amount, self.support_soh], dtype=np.float32)
        }

    def step(self, actions):
        main_action, support_action = actions
        self.prev_main_soc = self.main_soc
        self.prev_support_soc = self.support_soc

        # Assume renewable energy usage is fixed at 10 for simplicity
        renewable_usage = 10
        main_grid_usage = 0
        support_grid_usage = 0

        # === Main Agent ===
        if main_action == 1:  # discharge
            discharge_amount = min(10, self.main_soc)
            self.main_soc -= discharge_amount
            self.energy_demand -= discharge_amount
        else:  # charge
            charge_amount = min(5, self.res_amount)
            self.main_soc = min(self.max_soc, self.main_soc + charge_amount)
            self.res_amount -= charge_amount
            main_grid_usage += charge_amount

        # === Support Agent ===
        if support_action == 1:  # discharge
            discharge_amount = min(10, self.support_soc)
            self.support_soc -= discharge_amount
            self.energy_demand -= discharge_amount
            self.support_soh -= 0.001  # discharge affects battery health
        else:  # charge
            charge_amount = min(3, self.res_amount)
            self.support_soc = min(self.max_soc, self.support_soc + charge_amount)
            self.res_amount -= charge_amount
            support_grid_usage += charge_amount

        self.energy_demand = max(0, self.energy_demand)
        self.res_amount = max(0, self.res_amount)
        self.support_soh = max(0.0, self.support_soh)

        # === Compute Rewards ===
        reward_main = self.compute_reward(
            soc=self.main_soc,
            soc_prev=self.prev_main_soc,
            capacity=self.capacity,
            grid_usage=main_grid_usage,
            renewable_usage=renewable_usage
        )

        reward_support = self.compute_reward(
            soc=self.support_soc,
            soc_prev=self.prev_support_soc,
            capacity=self.capacity,
            grid_usage=support_grid_usage,
            renewable_usage=renewable_usage
        )

        reward_support -= (1.0 - self.support_soh) * 2  # extra penalty for SoH degradation

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_obs(), {'main': reward_main, 'support': reward_support}, done, {}

    def compute_reward(self, soc, soc_prev, capacity, grid_usage, renewable_usage):
        # Cost reward
        r_cost = -self.alpha * self.grid_price * (grid_usage - renewable_usage)

        # Battery health penalty
        delta_soc = abs(soc - soc_prev)
        aging_penalty = (delta_soc * self.battery_price) / self.cycle_life
        r_health = -self.lambda_ * aging_penalty

        # SOC penalty
        if soc > 90:
            r_soc = (90 - soc) * capacity
        elif soc < 20:
            r_soc = (soc - 20) * capacity
        else:
            r_soc = 0

        return r_cost + r_health + r_soc

    def render(self, mode='human'):
        print(f"\nStep {self.step_count}")
        print(f"Main SoC: {self.main_soc:.2f}, Support SoC: {self.support_soc:.2f}, SoH: {self.support_soh:.3f}")
        print(f"Energy Demand: {self.energy_demand}, Reserve: {self.res_amount}")
