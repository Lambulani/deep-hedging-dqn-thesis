from src.dqn_per import DQN
from src.envs import TradingEnv
import numpy as np


if __name__ == "__main__":
    # Initialize the environment and DQN agent
    env = TradingEnv(cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag=False,
                     continuous_action_flag=False, spread=0, init_ttm=10, trade_freq=0.2, num_contract=1, kappa=0.0,	
                     mu=0, vol=0.01 * np.sqrt(250), S=100, K=100, r=0, q=0)
    
    dqn = DQN(env)
    
    # Load the trained model
    dqn.load("dqn_model.h5")
    
    # Test the model
    episode_rewards, reinf_actions = dqn.test(total_episode=100)  # Adjust total_episode as needed

