# demotivational-policy-descent
### AI agents playing pong in a gym env
#### Reinforcement Learning course final project

AI agent learning to play pong using policy descent, actor critic and other approaches.

A PDF report of this project is available [here](https://github.com/fcole90/demotivational-policy-descent/releases/download/report/RL_report.pdf).

#### Preprocessing
![rl_ai_preprocessing](https://user-images.githubusercontent.com/1292230/69499071-19283d80-0ef7-11ea-88f3-90b69b7c6ae9.png)

#### Performance comparison
![rl_ai_performance](https://user-images.githubusercontent.com/1292230/69499072-19283d80-0ef7-11ea-9c5d-4abd079572f3.png)

#### Instructions

Methods `load_model` and `save_model` are implemented in
the agent interface.

The method `reset` needs to have the same signature as
the as the agent's `__init__` to avoid code duplication. 
Then call `reset()` from `__init__` and write inside `reset` what you 
would have written in your `__init__`.

The method `get_action` needs to be implemented in each
agent child.

## How to run?
Like this (choose the options to your taste):

`PYTHONPATH=. python demotivational_policy_descent/tests/run_policy_gradient.py --combine --preprocess --red --cuda`

#### Project Structure:

  - `agents/` contains the agents
  - `envs/` contains the Pong environment
  - `test` contains tests to run test the agents (that's what you run)
  - `utils/io` egocentric I/O library
  - `save_models/` contains saved models, if it doesn't, you're not working enough

#### Actions:

  - Up -> `self.env.MOVE_UP`, 1
  - Stay -> `self.env.STAY`, 0
  - Down -> `self.env.MOVE_DOWN`, 2
 
#### Observation Frames
The function `env.reset()` returns a frame of the game as a couple of
mirrored frames. This is to have the frame in the same way for each player.

The function `env.step(...)` returns the couple of frames, a couple of 
rewards, done and info.

### References

1. Karpathy Pong with Policy Gradient: [https://karpathy.github.io/2016/05/31/rl/](https://karpathy.github.io/2016/05/31/rl/)
