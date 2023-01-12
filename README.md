# Viper

Read the accompanying blog post here (tbd).

**V**erifiability via **I**terative **P**olicy **E**xt**R**action (2019) [paper](https://arxiv.org/abs/1805.08328)] 

In this paper the authors distill a Deep Reinforcement Learning such as DeepQN into a decision tree policy which can then be automatically checked for correctness, robustness, and stability. 

This repository implements and tests the viper algorithm on the following environments:

- CartPole
- Atari Pong
- ToyPong (tbd)

## Usage

### Training the oracle

Atari Pong:

```
python main.py train-oracle --env-name PongNoFrameskip-v4 --n-env 8 --total-timesteps 10_000_000
```

Cart pole:

```
python main.py train-oracle --env-name CartPole-v1 --n-env 8 --total-timesteps 100_000
```


### Running viper

Once the oracle policies are trained you can run viper on the same environment:

```
python main.py train-viper --env-name CartPole-v1 --n-env 1
```
