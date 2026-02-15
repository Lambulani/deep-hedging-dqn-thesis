import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# put things common to different algorithms here
class DRL:
    def __init__(self):
        if not os.path.exists('model'):
            os.mkdir('model')

        if not os.path.exists('history'):
            os.mkdir('history')
    
    
        
    def test(self, total_episode, delta_flag=False, bartlett_flag=False):
        """hedge with model.
        """
        print('testing...')

        def cost(delta_h, multiplier):
            TickSize = 0.1
            return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)
        
        def student_t_statistic(data):
            n = len(data)
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            t_statistic = sample_mean / (sample_std / np.sqrt(n))
            return t_statistic


        def get_total_pnl_function(asset_price_path, option_price_path, action_path):
            total_pnl_path= []
            num_steps = len(action_path)
            for t in range(1, num_steps):
                option_value_t = 100*(option_price_path[t] - option_price_path[t-1])
                hedge_value_t = action_path[t-1]* (asset_price_path[t] - asset_price_path[t-1])
                cost_t = cost(action_path[t]-action_path[t-1], 5)
                total_pnl_t= option_value_t - hedge_value_t - cost_t
                total_pnl_path.append(total_pnl_t)

            total_pnl = np.sum(total_pnl_path)
            total_pnl_vol = np.std(total_pnl_path)
            total_pnl_studentT = student_t_statistic(total_pnl_path)

            return total_pnl, total_pnl_vol, total_pnl_studentT

        self.epsilon = -1
        reinf_actions = []

        

        for i in range(total_episode):
            observation = self.env.reset()
            done = False
            action_store = []
            

            while not done:

                # prepare state
                x = np.array(observation).reshape(1, -1)

                if delta_flag:
                    action = self.env.delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                elif bartlett_flag:
                    action = self.env.bartlett_delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                else:
                    # choose action from epsilon-greedy; epsilon has been set to -1
                    action, _, _ = self.egreedy_action(x)

                # store action to take a look
                action_store.append(action)
                observation, reward, done, info = self.env.step(action)
            
            reinf_actions.append(action_store)
            
        #Calculating Total Cost of Hedging
        option_delta_scaled = self.env.delta_path* 100
        option_delta_scaled = np.insert(option_delta_scaled, 0, 0, axis=1)
        # Calculate the change in delta
        bs_delta_change = np.diff(option_delta_scaled, axis=1)
        bs_cost_path = cost(bs_delta_change, 5)
        bs_total_cost = np.sum(bs_cost_path, axis=1)

        reinf_actions = np.insert(reinf_actions, 0, 0, axis=1)
        reinf_delta_change = np.diff(reinf_actions, axis=1)
        reinf_cost_path = cost(reinf_delta_change, 5)
        reinf_total_cost = np.sum(reinf_cost_path, axis=1)

        #calculate the total pnl delta 
        bs_total_pnl = []
        reinf_total_pnl = []

        bs_total_pnl_vol = []
        reinf_total_pnl_vol = []
        
        bs_total_pnl_studentT = []
        reinf_total_pnl_studentT= []

        self.env.path = np.insert(self.env.path, 0, 0, axis=1)
        self.env.option_price_path = np.insert(self.env.option_price_path, 0, 0, axis=1)

        for i in range(total_episode):
            bs_total_pnl.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i, :], option_delta_scaled[i,:])[0])
            reinf_total_pnl.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i,:], reinf_actions[i,:])[0])
           
            bs_total_pnl_vol.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i, :], option_delta_scaled[i,:])[1])
            reinf_total_pnl_vol.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i,:], reinf_actions[i,:])[1])
            
            bs_total_pnl_studentT.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i, :], option_delta_scaled[i,:])[2])
            reinf_total_pnl_studentT.append(get_total_pnl_function(self.env.path[i, :], self.env.option_price_path[i,:], reinf_actions[i,:])[2])

    
        
        

        
        # Kernel density plot showing the total cost of delta hedging
        plt.figure(figsize=(10, 5))
        sns.kdeplot(bs_total_cost, shade=True, label='BS delta Total Cost', color = "orange", bw_adjust=1.5)
        # sns.kdeplot(reinf_total_cost, shade=True, label='DDPG Total Cost',
        #             color = "blue", bw_adjust=1.5)
        plt.xlabel("Total Cost")
        plt.ylabel("Density")
        plt.title('Kernel Density Plot of Total Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig("Kernel_Density_Plot_Total_Cost.png")
        plt.show()

        # Kernel density plot showing the total cost of delta hedging
        plt.figure(figsize=(10, 5))
        sns.kdeplot(bs_total_pnl_vol, shade=True, label='BS delta Total PnL Volatility', color = "orange", bw_adjust=1.5)
        # sns.kdeplot(reinf_total_pnl_vol, shade=True, label='DDPG Total PnL Volatility ',
        #             color = "blue", bw_adjust=1.5)
        plt.xlabel("Total PnL Volatility")
        plt.ylabel("Density")
        plt.title('Kernel Density Plot of Total PnL Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig("Kernel_Density_Plot_of_Total_PnL_Volatility.png")
        plt.show()

        # Kernel density plot showing =the student T statistic of the total PnL
        plt.figure(figsize=(10, 5))
        sns.kdeplot(bs_total_pnl_studentT, shade=True, label='BS delta Total PnL Student t statistic', color = "orange",bw_adjust=1.5, )
        # sns.kdeplot(reinf_total_pnl_vol, shade=True, label='DDPG Total PnL Student t statistic ',
        #             color = "blue", bw_adjust=1.5)
        plt.xlabel("Studennt t statistic Total PnL")
        plt.ylabel("Density")
        plt.title('Kernel Density Plot of Student T statistic Total PnL')
        plt.legend()
        plt.grid(True)
        plt.savefig("Kernel_Density_Plot_of_StudentT_Total PnL_plot.png")
        plt.show()
    

    def plot(self):
        print("Ploting out of sample simulation")

        self.epsilon = -1 

        observation = self.env.reset()
        done = False
        action_store = []

        while not done:
            # prepare state
            x = np.array(observation).reshape(1, -1)
            action, _, _ = self.egreedy_action(x)
            action_store.append(action)
            observation, reward, done, info = self.env.step(action)
            path_row = info["path_row"]

        plt.figure(figsize=(10,5))
        plt.plot(action_store, label = "RL actions taken")
        plt.plot(self.env.delta_path[path_row]* 100, label = "Delta Path" )
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.title("Out of Sample Simulation ")
        plt.legend()
        plt.savefig("Out_of_Sample_Simulation.png")
        plt.show()




    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')



    # def test(self, total_episode, delta_flag=False, bartlett_flag=False):
    #     """hedge with model.
    #     """
    #     print('testing...')

    #     self.epsilon = -1

    #     w_T_store = []

    #     for i in range(total_episode):
    #         observation = self.env.reset()
    #         done = False
    #         action_store = []
    #         reward_store = []

    #         while not done:

    #             # prepare state
    #             x = np.array(observation).reshape(1, -1)

    #             if delta_flag:
    #                 action = self.env.delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
    #             elif bartlett_flag:
    #                 action = self.env.bartlett_delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
    #             else:
    #                 # choose action from epsilon-greedy; epsilon has been set to -1
    #                 action, _, _ = self.egreedy_action(x)

    #             # store action to take a look
    #             action_store.append(action)

    #             # a step
    #             observation, reward, done, info = self.env.step(action)
    #             reward_store.append(reward)

    #         # get final wealth at the end of episode, and store it.
    #         w_T = sum(reward_store)
    #         w_T_store.append(w_T)

    #         if i % 1000 == 0:
    #             w_T_mean = np.mean(w_T_store)
    #             w_T_var = np.var(w_T_store)
    #             path_row = info["path_row"]
    #             print(info)
    #             with np.printoptions(precision=2, suppress=True):
    #                 print("episode: {} | final wealth: {:.2f}; so far mean and variance of final wealth was {} and {}".format(i, w_T, w_T_mean, w_T_var))
    #                 print("episode: {} | so far Y(0): {:.2f}".format(i, -w_T_mean + self.ra_c * np.sqrt(w_T_var)))
    #                 print("episode: {} | rewards: {}".format(i, np.array(reward_store)))
    #                 print("episode: {} | action taken: {}".format(i, np.array(action_store)))
    #                 print("episode: {} | deltas {}".format(i, self.env.delta_path[path_row] * 100))
    #                 print("episode: {} | stock price {}".format(i, self.env.path[path_row]))
    #                 print("episode: {} | option price {}\n".format(i, self.env.option_price_path[path_row] * 100))
