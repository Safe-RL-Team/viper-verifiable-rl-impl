# Learnings

A collection of learnings and observations in no particular order that I made when implementing this project.

**Increasing the number of training environments for SB can degrade performance, e.g. DQN on Atari Pong**


**PPO does not converge to an optimal policy on ToyPong (r=250)**

After trying out many hyperparameters that would always stabilize at ~210 reward, I noticed that the way the
environment is set up makes it **impossible** for any agent to play perfectly.

Namely, if the speed of the paddle is 1 whereas the ball is moving at a speed between 1 and 2 (inclusive), the paddle
will never be able to catch the ball even if the paddle's length is increased to 5 which is what the authors do after
finding an edge case using their correctness algorithm.

The paddle's speed is not specified in the paper and setting it to 1 initially seemed like a good start.
We know that it has to be constant, i.e. no acceleration, given the agent action {left, right, stay} otherwise the piecewise linearity
assumption from the correctness test would not hold.
A simple calculation suggests that the paddle's speed should be at least 1.5, but this is not specified in the paper.

**The entropy coefficient doesn't have a noticeable impact when training ToyPong using PPO**


**VIPER defines the currently training policy differently from DAGGER**

In DAGGER with each iteration the playing policy is a mix between the expert and the learner. 
We start with a 100% expert policy and gradually decrease the expert's influence down to 0% in the final iteration. 
Initially I copied this strategy into VIPER assuming that without it the first iterations would not generate useful data.
However, I was only able to achieve perfect reward on ToyPong by doing exactly what the paper said which is only used the expert policy in the first iteration and from then on only use the learner's policy.
