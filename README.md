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

## References
1. Karpathy Pong with Policy Gradient: [https://karpathy.github.io/2016/05/31/rl/](https://karpathy.github.io/2016/05/31/rl/)
