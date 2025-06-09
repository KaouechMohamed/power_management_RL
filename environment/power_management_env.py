import numpy as np
import pandas as pd
import gym
from gym import spaces

class BatteryClusterEnv(gym.Env):
    def __init__(self, df, reward_function):
        super(BatteryClusterEnv, self).__init__()
        
        # Load dataframe containing state information
        self.df = df
        self.current_step = 0
        
        # Define action space (4 discrete actions)
        self.action_space = spaces.Discrete(4)
        """
        Action Mapping:
        0: Charge Main Cluster
        1: Discharge Main Cluster
        2: Charge Support Cluster
        3: Discharge Support Cluster
        """
        
        # Define state space (using dataframe columns)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=len(self.df.columns),  
            dtype=np.float32
        )
        
        # External reward function
        self.reward_function = reward_function        
        self.main_cluster_soc = 0.5  
        self.support_cluster_soc = 0.5
        self.max_charge_rate = 0.1 
        
    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs
        
    def reset(self):
        self.current_step = 0
        self.main_cluster_soc = 0.5
        self.support_cluster_soc = 0.5
        return self._next_observation()
    
    def step(self, action):
        current_state = self._next_observation()
        
        # Apply action
        if action == 0:  # Charge Main
            self.main_cluster_soc = min(1.0, self.main_cluster_soc + self.max_charge_rate)
        elif action == 1:  # Discharge Main
            self.main_cluster_soc = max(0.0, self.main_cluster_soc - self.max_charge_rate)
        elif action == 2:  # Charge Support
            self.support_cluster_soc = min(1.0, self.support_cluster_soc + self.max_charge_rate)
        elif action == 3:  # Discharge Support
            self.support_cluster_soc = max(0.0, self.support_cluster_soc - self.max_charge_rate)
            
        # Calculate reward using external function
        reward = self.reward_function(
            current_state,
            action,
            main_soc=self.main_cluster_soc,
            support_soc=self.support_cluster_soc
        )
        
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Get next state
        next_state = self._next_observation() if not done else None
        
        return next_state, reward, done, {}

