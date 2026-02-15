# deep-hedging-dqn-thesis

## Acknowledgement

This repository builds upon and extends the work originally developed in:
[rl-hedge-2029](https://github.com/tdmdal/rl-hedge-2019.git).

## Requirements
The code requires the following Python packages:  
- `gym==0.21.0` (for custom environments)  
- `tensorflow==2.x`  
- `keras==2.3.1`  

Additional packages used for data manipulation and plotting include `numpy`, `pandas`, and `matplotlib`.

## Project Structure

src/ – Contains all source code used for training the agents. 

notebooks/ – Contains exploratory data analysis (EDA) and experimentation for both DQN agents and Black--Scholes delta hedging strategies

models/ – Stores saved models from different training experiments. Each folder corresponds to a specific training run with hyperparameters outlined below (key trainings: 12, 25, 29–37).

---

## Usage
- To start training a DQN hedging agent:  
```bash
python src/dqn_per.py
```

## Key Training Experiments

The following trainings highlight the best performing DQN models in terms of tracking the delta-hedging benchmark and optimizing the cost function under different market frictions.

| Training | Cost Multiplier | Risk Aversion κ | Hidden Layers | Batch Norm | Notes on Performance |
|----------|----------------|----------------|---------------|------------|--------------------|
| 12       | 1              | 0.1            | 5 (64)        | No         | Closely replicated delta-hedging policy; stable convergence; good overall performance. |
| 25       | 0              | 1/10           | 5 (64)        | No         | Tracked delta more accurately than earlier models; stable policy; used as a reference for zero-cost hedging. |
| 29       | 0              | 0              | 5 (64)        | No         | Baseline cost-free model; simple policy, easy to interpret; stable but limited adaptability. |
| 30       | 0              | 1/6            | 5 (64)        | No         | Slightly more risk-averse; last model performed best in terms of cost reduction. |
| 31       | 0              | 1/4            | 5 (64)        | No         | Further increased risk aversion; last model was the best for the training set. |
| 32       | 0              | 1/2            | 5 (64)        | No         | High risk aversion; model 170k performed best; trades more selectively to control risk. |
| 33       | 1              | 1/8            | 5 (64)        | No         | Moderate cost and risk; good replication of delta; stable performance. |
| 34       | 1              | 1/6            | 5 (64)        | No         | Model 258k achieved best tracking of delta; moderate cost and risk settings. |
| 35       | 1              | 1/4            | 5 (64)        | No         | Stable convergence; slightly more aggressive policy due to higher risk aversion. |
| 36       | 1              | 1/2            | 5 (64)        | No         | Model 218k was the best; reduced trading frequency to control cost under high risk aversion. |
| 37       | 5              | 1/10           | 5 (64)        | No         | High cost scenario; last model performed best; demonstrated selective trading with cost consideration. |

**Notes:**
- All experiments used a **trade frequency of 5/day**, **discount factor 0.9**, **Huber loss δ = 1**, and **target updates every 10 steps/episodes**.  
