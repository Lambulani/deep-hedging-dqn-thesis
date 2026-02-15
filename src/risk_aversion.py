import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.dqn_per import DQN
from src.envs import TradingEnv


def cost(delta_h, multiplier):
    """Calculate transaction costs based on delta changes."""
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)


# Parameters for training and testing
train_num = [25, 31]
model = ["282000", "348000"]
kappa_values = [1/10, 1/6]
cost_multiplier = 0 
oos = 10000

# Dictionary to store results for each kappa value
results = {}

dqn_actions_all = {}

for i in range(len(train_num)):
    kappa = kappa_values[i]
    
    # Initialize the environment
    env = TradingEnv(
        continuous_action_flag=False,
        sabr_flag=False,
        dg_random_seed=1,
        spread=0.0,
        num_contract=1,
        init_ttm=10,
        trade_freq=1/5,
        num_sim=oos,
        kappa=kappa,
        cost_multiplier=cost_multiplier,
        mu=0,
        vol=0.01 * np.sqrt(250),
        S=100,
        K=100,
        r=0,
        q=0
    )
    
    # Load the trained DQN model
    dqn = DQN(env)
    dqn.load(path=f"model/dqn/Training{train_num[i]}/dqn_model_{model[i]}.h5")
    
    # Test the DQN model
    episode_rewards, dqn_actions, q_values = dqn.test(total_episode=oos)
    dqn_actions = np.array(dqn_actions)
    dqn_actions = np.insert(dqn_actions, 0, 0, axis=1)
    action = np.diff(dqn_actions, axis=1)
    
    # Calculate transaction costs for DQN policy
    transaction_costs_dqn = cost(action, multiplier=cost_multiplier)
    total_costs_dqn = np.sum(transaction_costs_dqn, axis=1)
    
    # Calculate transaction costs for delta policy
    delta_path = env.delta_path * 100
    delta_path_append = np.insert(delta_path, 0, 0, axis=1)
    delta_h = np.diff(delta_path_append, axis=1)[:, :-1]
    transaction_costs_delta = cost(delta_h, multiplier=cost_multiplier)
    total_costs_delta = np.sum(transaction_costs_delta, axis=1)
    
    # Calculate PnL for delta policy
    v_t = env.option_price_path * 100
    v_t_diff = np.diff(v_t, axis=1)
    s_t = env.path
    s_t_diff = np.diff(s_t, axis=1)
    a_t_delta = delta_path[:, :-1]
    h_t_delta = a_t_delta * s_t_diff
    pi_t_delta = v_t_diff - h_t_delta - transaction_costs_delta
    total_pnl_delta = np.sum(pi_t_delta[:, 1:], axis=1)
    
    # Calculate PnL for DQN policy
    a_t_dqn = dqn_actions[:, :-1]
    h_t_dqn = a_t_dqn * s_t_diff
    pi_t_dqn = v_t_diff - h_t_dqn - transaction_costs_dqn
    total_pnl_dqn = np.sum(pi_t_dqn[:, 1:], axis=1)
    
    # Store results in the dictionary
    results[kappa] = {
        "std_pnl_dqn": np.std(pi_t_dqn, axis=1),
        "std_pnl_delta": np.std(pi_t_delta, axis=1)
    }

    dqn_actions_all[kappa] = { 
        "dqn_actions": dqn_actions,
        "delta_path": delta_path
    }


# Plot the pnl 
plt.figure(figsize=(10, 5))
for kappa, result in results.items():
    sns.kdeplot(result["std_pnl_dqn"], label=f"DQN kappa {kappa}", bw_adjust=1.2)

sns.kdeplot(results[kappa_values[0]]["std_pnl_delta"], label="Delta Hedge", color="orange", bw_adjust=1.2)
plt.xlabel("Realized Volatilty of Total PnL")
plt.ylabel("Density")
plt.title("Distribution of Realized Volatilty of Total PnL within Episode")
plt.legend()
plt.savefig(f"history/dqn/risk_aversion/realized_volatility_pnl_cost{cost_multiplier}.png")
plt.show()



# Time steps
eps = 73
time_steps = np.arange(len(dqn_actions_all[kappa_values[0]]["dqn_actions"][eps]))
colours = ["green", "blue"]

# Plot the hedging positions
plt.figure(figsize=(10, 6))
for kappa, dqn_actions_all in dqn_actions_all.items():
    dqn_positions = dqn_actions_all["dqn_actions"][eps,1:]
    plt.plot(time_steps[1:], dqn_positions, label=f"DQN kappa {kappa}", color=colours.pop(0))
plt.plot(time_steps[1:], dqn_actions_all["delta_path"][eps, 1:], label="Delta Hedging Position", color="orange")
plt.xlabel("Time Step")
plt.ylabel("Hedging Position")
plt.title("Hedging Positions: DQN vs Delta Hedge")
plt.legend()
plt.grid()
plt.savefig(f"history/dqn/risk_aversion/hedging_positions_cost{cost_multiplier}.png")
plt.show()

