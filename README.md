# QPLEX Implementation for MATE Environment

This repository contains a complete implementation of the QPLEX (Q-value decomposition with dueling architecture) algorithm for multi-agent reinforcement learning, specifically designed to work with the MATE (Multi-Agent Tracking Environment).

## Overview

QPLEX is a state-of-the-art multi-agent reinforcement learning algorithm that combines:
- Individual Q-networks for each agent
- A mixing network for value decomposition
- Dueling architecture for better value estimation
- Support for various network architectures (MLP, RNN, Attention)

## Features

- **Complete QPLEX Implementation**: Full implementation of the QPLEX algorithm with configurable architectures
- **MATE Environment Integration**: Seamless integration with the MATE multi-agent tracking environment
- **Flexible Network Architectures**: Support for MLP, RNN, LSTM, GRU, and Attention mechanisms
- **Multiple Mixing Networks**: Various mixing network implementations including hypernetworks and attention-based mixing
- **Comprehensive Training Pipeline**: Complete training script with logging, evaluation, and model saving
- **Configurable Experiments**: YAML-based configuration system for easy experiment management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QPLEX
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the MATE environment (if not already installed):
```bash
pip install mate-env
```

## Quick Start

### 1. Run Example
```bash
python example.py
```

### 2. Train QPLEX
```bash
python train_qplex_mate.py --config configs/qplex_4v4_9.yaml
```

### 3. Train with Custom Configuration
```bash
python train_qplex_mate.py --config configs/qplex_4v8_0.yaml --seed 123
```

## Configuration

The training is configured through YAML files in the `configs/` directory. Key configuration options include:

### Environment Settings
- `env.config_file`: Path to MATE environment configuration
- `env.max_episode_steps`: Maximum steps per episode
- `env.render_mode`: Rendering mode ("human" or "rgb_array")

### Algorithm Settings
- `algorithm.learning_rate`: Learning rate for the optimizer
- `algorithm.gamma`: Discount factor
- `algorithm.epsilon_start/end/decay`: Exploration parameters
- `algorithm.dueling`: Whether to use dueling architecture
- `algorithm.double_q`: Whether to use double Q-learning

### Network Architecture
- `network.q_network`: Individual Q-network configuration
- `network.mixing_network`: Mixing network configuration
- Support for RNN, LSTM, GRU, and Attention mechanisms

### Training Settings
- `training.total_timesteps`: Total training timesteps
- `training.batch_size`: Batch size for training
- `training.buffer_size`: Replay buffer size
- `training.learning_starts`: Timesteps before learning starts

## Project Structure

```
QPLEX/
├── algorithms/
│   └── qplex/
│       ├── agent.py          # QPLEX agent implementation
│       ├── learner.py        # Training and experience replay
│       ├── mixer.py          # Mixing network implementations
│       └── model.py          # Complete QPLEX model
├── networks/
│   ├── base_networks.py      # Base network architectures
│   └── rnn_networks.py       # RNN-based architectures
├── configs/
│   ├── qplex_4v4_9.yaml      # Configuration for 4v4 environment
│   └── qplex_4v8_0.yaml      # Configuration for 4v8 environment
├── mate/                     # MATE environment (if included)
├── train_qplex_mate.py       # Main training script
├── example.py                # Example usage script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Network Architectures

### Individual Q-Networks
- **MLP**: Standard multi-layer perceptron
- **RNN**: Recurrent neural network with LSTM/GRU
- **Attention RNN**: RNN with self-attention mechanism
- **Bidirectional RNN**: Bidirectional LSTM/GRU
- **Hierarchical RNN**: Multi-scale temporal modeling

### Mixing Networks
- **QPLEX Mixing**: Standard QPLEX hypernetwork mixing
- **Attention Mixing**: Attention-based value mixing
- **Monotonic Mixing**: Monotonicity-preserving mixing
- **Hierarchical Mixing**: Multi-level mixing networks
- **Adaptive Mixing**: Complexity-adaptive mixing

## Training

The training script provides comprehensive logging and evaluation:

### Logging
- Training statistics (loss, Q-values, TD errors)
- Episode statistics (rewards, lengths, coverage rates)
- Model checkpoints and evaluation results

### Evaluation
- Regular evaluation during training
- Coverage rate and transport rate metrics
- Final model evaluation

### Model Saving
- Automatic model checkpointing
- Training statistics logging
- Configuration saving

## Usage Examples

### Basic Training
```python
from algorithms.qplex.learner import QPLEXLearner
import yaml

# Load configuration
with open('configs/qplex_4v4_9.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create learner
learner = QPLEXLearner(config, device)
learner.setup(obs_dim, action_dim, state_dim, n_agents)

# Train
for timestep in range(total_timesteps):
    # ... environment interaction ...
    learning_info = learner.learn(obs, actions, rewards, next_obs, done, state, next_state)
```

### Custom Network Architecture
```python
# In configuration file
network:
  q_network:
    type: "attention_rnn"
    hidden_dims: [256, 256]
    use_rnn: true
    rnn_hidden_dim: 128
    rnn_layers: 2
    use_attention: true
    num_attention_heads: 4
  
  mixing_network:
    type: "attention"
    hidden_dims: [256, 256]
    num_attention_heads: 8
```

## Evaluation Metrics

The implementation tracks several important metrics:

- **Episode Reward**: Total reward per episode
- **Coverage Rate**: Percentage of targets being tracked
- **Transport Rate**: Efficiency of cargo delivery
- **Episode Length**: Number of steps per episode
- **Q-Values**: Individual and total Q-value statistics
- **TD Errors**: Temporal difference errors

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Environment Not Found**: Ensure MATE environment is properly installed
3. **Configuration Errors**: Check YAML syntax and file paths

### Performance Tips

1. **Use GPU**: Enable CUDA for faster training
2. **Adjust Batch Size**: Larger batch sizes for more stable training
3. **Network Architecture**: Use RNN for environments with temporal dependencies
4. **Hyperparameters**: Tune learning rate and exploration parameters

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite the original QPLEX paper:

```bibtex
@article{wang2020qplex,
  title={QPLEX: Duplex Dueling Multi-Agent Q-Learning},
  author={Wang, Jianhao and Ren, Zhizhou and Liu, Tonghan and Yu, Yang and Zhang, Chongjie},
  journal={arXiv preprint arXiv:2008.01062},
  year={2020}
}
```

## Acknowledgments

- Original QPLEX paper authors
- MATE environment developers
- PyTorch and Gymnasium communities