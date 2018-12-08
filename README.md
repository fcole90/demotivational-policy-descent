# demotivational-policy-descent
Reinforcement Learning final project

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
