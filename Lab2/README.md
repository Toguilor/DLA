# Deep Reinforcement Learning Laboratory

## Description
This laboratory focuses on policy gradient methods for **Deep Reinforcement Learning***, with a particular emphasis on the **REINFORCE** algorithm and its variants.
The objective is to understand, implement, and analyze the behavior of policy-based agents on classical control environments from ***Gymnasium***, such as **CartPole** and **LunarLander**.

The lab is structured progressively:
- Implementation of the vanilla REINFORCE algorithm.
- Introduction of a baseline (state-value function) to reduce variance.
- Design of evaluation protocols beyond running averages.
- Empirical analysis of learning stability and performance.

All experiments are implemented in **PyTorch**.

## <ins>Exercice 1</ins>: Improving my `REINFORCE` Implementation (warm up)
The goal of this exercise is to implement the REINFORCE algorithm, and train an agent on a discrete control environment (CartPole).

- Exploration of the environnement: analyse of the observation space (4 continuous variables) and the action space (2 discret variables).
- Implementation of the REINFORCE: Implementation of the training loop where the agent collects complete trajectories before updating its policy via gradient climbing.
- Baseline by standardization: Adding a step to normalize cumulative returns (subtracting the mean and dividing by the standard deviation) within each episode to reduce the variance of the gradient.
- Training: The agent was trained on 1000 episodes. The `reinforce` function calculates the discounted returns and updates the neural network.
- Observations: The average score increased from approximately 15 (initial score) to a maximum of 500.00.
Although the agent reached the maximum score, some instability (performance drops) was observed, typical of REINFORCE without more advanced regularization mechanisms.

<img width="700" height="400" alt="lab2_1" src="https://github.com/user-attachments/assets/6e80dd2b-e63c-43e8-8333-f5e695061e6c" />

## <ins>Exercice 2</ins>: `REINFORCE` with a Value Baseline (warm up)
This exercise introduces a baseline to reduce the variance of policy gradient estimates.

To solve this problem, we experiment with a dual architecture: a PolicyNetwork (the actor) is used to select actions, and a ValueNetwork (the critique) is used to estimate the expected future gains from each state.

Next, we calculate the advantage: instead of using the gross return ***G***, the algorithm uses the advantage ***A = G - V(s)***. This allows us to "center" the updates: we encourage actions that have performed better than the expected average.

Finally, we perform joint optimization: the code simultaneously minimizes the policy loss and the mean squared error (MSE) of the value network.

<img width="1189" height="490" alt="output" src="https://github.com/user-attachments/assets/3bee88b3-928a-4b64-89f2-3b38fb31e2ed" />

## <ins>Exercice 3.1</ins>: Solving Lunar Lander with `REINFORCE`
In this exercise, the objective was to apply the REINFORCE policy gradient algorithm to a more complex environment than CartPole: LunarLander-v2. The agent must learn to land a lunar module on a platform while controlling three engines (main, left, right). The observation space is richer (8 continuous variables including position, velocity, and angle), and the rewards are more numerous but also more punishing (crash).

We use the advantage also here as:
```python
baseline = value_net(s)
advantage = G - baseline.item()
policy_loss += -log_prob * advantage
```
This approach allows an action to be reinforced only if it produces a better result than what the value network predicted for that state. `The value_loss` uses the mean squared error (MSE) so that ***V(s)*** gets as close as possible to the actual return ***G***.

<img width="1189" height="590" alt="output2" src="https://github.com/user-attachments/assets/8f902484-02fc-4c7f-97f1-f6f3a305056c" />

The agent shows gradual progress. Initially, the scores are very negative (around -200/-300 due to frequent crashes). Towards the end of the 1000 episodes, the agent manages to stabilize its landings, achieving positive scores.

Training scale: With 3000 episodes, the agent has time to explore the different landing zones. The evaluation interval every 100 episodes allows for tracking actual progress without the "noise" of exploration. 
Stability: The use of the advantage (***G - V(s)***) is crucial here. Without it, the very negative rewards of a crash could make the gradient unstable and prevent the agent from learning to stabilize the module. 
