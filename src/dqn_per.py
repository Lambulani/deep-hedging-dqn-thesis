import os
import numpy as np
import random
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.losses import Huber
from keras.optimizers import Adam
from src.replay_buffer import PrioritizedReplayBuffer
from src.schedules import LinearSchedule
from src.drl import DRL
from src.envs import TradingEnv
import matplotlib.pyplot as plt
import pandas as pd

class DQN(DRL):
    def __init__(self, env, gamma=0.9, learning_rate=1e-4, buffer_size=1000000, batch_size=128, tau=0.005):
        super(DQN, self).__init__()
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.TAU = tau
        self.hidden_layers = 64

        # Q-networks
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

        # Replay buffer and PER schedule
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
        prioritized_replay_beta_iters = 50000
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
        self.prioritized_replay_eps = 1e-6

        self.t = 0

    def _build_q_network(self):
        inputs = Input(shape=(self.env.num_state,))
        x = Dense(self.hidden_layers, activation="relu")(inputs)
        x = Dense(self.hidden_layers, activation="relu")(x)
        x = Dense(self.hidden_layers, activation="relu")(x)
        x = Dense(self.hidden_layers, activation="relu")(x)
        x = Dense(self.hidden_layers, activation="relu")(x)
        output = Dense(self.env.action_space.n, activation="linear")(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss=Huber(delta=1.0), optimizer=Adam(self.learning_rate))
        return model

    def egreedy_action(self, state):

        state= np.array(state).flatten()
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.q_network.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def get_q_values(self, state):
        """
        Compute Q-values for all possible actions given a state.
        """
        x = np.array(state).flatten()
        q_values = self.q_network.predict(x[np.newaxis], verbose=0)
        return q_values[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def process_batch(self, batch_size):
        experience = self.replay_buffer.sample(batch_size, beta=self.beta_schedule.value(self.t))
        (states, actions, rewards, next_states, dones, weights, batch_idxes) = experience
        actions = actions.astype(int)
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1)
        return states, actions, rewards, next_states, dones, weights, batch_idxes

    def update_model(self, states, actions, rewards, next_states, dones, weights, batch_idxes):
        # Predict Q-values for next states using target network
        next_qs = self.target_q_network.predict(next_states, verbose=0)
        # Predict Q-values for current states
        q_values = self.q_network.predict(states, verbose=0)
        # Compute targets
        targets = q_values.copy()
        for i in range(states.shape[0]):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_qs[i])
        # Train
        history = self.q_network.fit(states, targets, sample_weight=weights.flatten(), verbose=0)
        # Update priorities if using PER
        td_errors = targets[np.arange(states.shape[0]), actions] - q_values[np.arange(states.shape[0]), actions]
        new_priorities = (np.abs(td_errors) + self.prioritized_replay_eps).flatten()
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)
        return np.mean(history.history['loss'])

    def update_target_network(self):
        q_weights = self.q_network.get_weights()
        target_weights = self.target_q_network.get_weights()
        for i in range(len(q_weights)):
            target_weights[i] = self.TAU * q_weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_q_network.set_weights(target_weights)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        history = {"episode": [], "reward": [], "loss": []}
        step_rewards = []

        for i in range(episodes):
            state = self.env.reset()
            state= np.array(state)
            action_store =[]
            bs_delta = self.env.delta_path[i]
            done = False
            total_reward = 0
            losses = []
            self.t = i
            target_update_freq = 10
            update_counter = 0

           
            while not done:
                action = self.egreedy_action(state)
                action_store.append(action)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.array(next_state) 
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                step_rewards.append(reward)

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.process_batch(self.batch_size)
                    loss = self.update_model(*batch)
                    update_counter+=1
                    if update_counter % target_update_freq ==0:
                        self.update_target_network()
                    losses.append(loss)

            self.update_epsilon()
            avg_loss = np.mean(losses) if losses else 0
            history["episode"].append(i)
            history["reward"].append(total_reward)
            history["loss"].append(avg_loss)

            if i % 1000== 0:
                print(f"Episode {i} | Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.3f}")
                print(action_store)
                print(bs_delta*100)


                df = pd.DataFrame.from_dict(history)
                df.to_csv("history/dqn/Training43/dqn_training_history.csv", index=False, encoding='utf-8')
        


            if (i + 1) % 1000== 0:
                save_path = f"model/dqn/Training43/dqn_model_{i+1}.h5"
                dqn.save(save_path)
                print(f"Saved model at episode {i+1} to {save_path}")
                
                # plt.figure(figsize=(12, 5))
                # plt.plot(step_rewards)
                # plt.xlabel("Step")
                # plt.ylabel("Reward")
                # plt.title("Reward vs. Data Point (Step) During Training")
                # plt.show()

        return history

    def save(self, path="dqn_model.h5"):
        self.q_network.save_weights(path)

    def load(self, path="dqn_model.h5"):
        if os.path.exists(path):
            self.q_network.load_weights(path)
            self.target_q_network.load_weights(path)
    
    def test(self, total_episode, delta_flag=False, bartlett_flag=False):
        """Test the trained DQN model."""
        print('Testing DQN model...')

        self.epsilon = 0  # Ensure greedy policy for DQN

        reinf_actions = []
        episode_rewards = []
        q_values_all_episodes = []  # Store Q-values for all episodes

        for i in range(total_episode):
            observation = self.env.reset()
            done = False
            action_store = []
            reward_store = []
            q_values_store = []  # Store Q-values for this episode

            while not done:
                x = np.array(observation).flatten()
                # For DQN, always use greedy action (no exploration)
                if delta_flag:
                    action = self.env.delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                elif bartlett_flag:
                    action = self.env.bartlett_delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                else:
                    # DQN: always pick the action with highest Q-value
                    q_values = self.q_network.predict(x[np.newaxis], verbose=0)
                    action = np.argmax(q_values[0])
                    q_values_store.append(q_values[0])  # Store Q-values for this time step
                action_store.append(action)
                observation, reward, done, info = self.env.step(action)
                reward_store.append(reward)

            reinf_actions.append(action_store)
            episode_rewards.append(reward_store)
            q_values_all_episodes.append(q_values_store)  # Store Q-values for this episode

            if i % 100 == 0:
                print(f"Episode {i}: Total Reward = {np.sum(reward_store):.2f}")

        print(f"\nAverage Reward over {total_episode} episodes: {np.mean(episode_rewards):.2f}")

    # Return Q-values along with rewards and actions
        return episode_rewards, reinf_actions, q_values_all_episodes
    
    def get_action(self, state):
        x = np.array(state).flatten()
        q_values = self.q_network.predict(x[np.newaxis], verbose=0)
        action = np.argmax(q_values[0])
        return action

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    random.seed(None)

    episodes = 1000000
    vol = 0.01 * np.sqrt(250)
    env = TradingEnv(continuous_action_flag=False, sabr_flag=False,
                     dg_random_seed=None, spread=0.0, num_contract=1,
                     init_ttm=10, trade_freq=0.2, num_sim=episodes, kappa=1/10, cost_multiplier=1,
                     mu=0, vol=vol, S=100, K=100, r=0, q=0)
    dqn = DQN(env)
    history = dqn.train(episodes=episodes)
    dqn.save_history(history, "dqn/Training43/dqn_training_history.csv")
    # Save the model after training 
    dqn.save("dqn_model.h5")