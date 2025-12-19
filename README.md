# Multi-Agent Mirror Descent Safe Policy Optimization (MAMDSPO)

MAMDSPO is a simple first-order method designed for **safe multi-agent reinforcement learning** under the **centralized training with decentralized execution** paradigm. It leverages **mirror descent** to maximize agent-specific returns while satisfying cost constraints. The algorithm consists of three key phases:
1. **Gradient descent** (without cost constraints)
2. **Projection onto the nonparametric policy space** (with cost constraints)
3. **Projection onto the parametric policy space**