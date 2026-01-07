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

## <ins>Exercice 1</ins>
The goal of this exercise is to implement the REINFORCE algorithm, and train an agent on a discrete control environment (CartPole).

- Exploration of the environnement: analyse of the observation space (4 continuous variables) and the action space (2 discret variables).
- Implementation of the REINFORCE: Implementation of the training loop where the agent collects complete trajectories before updating its policy via gradient climbing.
- Baseline by standardization: Adding a step to normalize cumulative returns (subtracting the mean and dividing by the standard deviation) within each episode to reduce the variance of the gradient.
- Training: The agent was trained on 1000 episodes. The `reinforce` function calculates the discounted returns and updates the neural network.
- Observations: The average score increased from approximately 15 (initial score) to a maximum of 500.00.
Although the agent reached the maximum score, some instability (performance drops) was observed, typical of REINFORCE without more advanced regularization mechanisms.

<img width="700" height="400" alt="lab2_1" src="https://github.com/user-attachments/assets/6e80dd2b-e63c-43e8-8333-f5e695061e6c" />

## <ins>Exercice 2</ins>
This exercise introduces a baseline to reduce the variance of policy gradient estimates.

To solve this problem, we experiment with a dual architecture: a PolicyNetwork (the actor) is used to select actions, and a ValueNetwork (the critique) is used to estimate the expected future gains from each state.

Next, we calculate the advantage: instead of using the gross return ***G***, the algorithm uses the advantage ***A = G - V(s)***. This allows us to "center" the updates: we encourage actions that have performed better than the expected average.

Finally, we perform joint optimization: the code simultaneously minimizes the policy loss and the mean squared error (MSE) of the value network.

<img width="1189" height="490" alt="output" src="https://github.com/user-attachments/assets/3bee88b3-928a-4b64-89f2-3b38fb31e2ed" />


