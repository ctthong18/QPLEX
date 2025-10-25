"""Test script to verify installation and basic functionality."""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
        return False
    
    try:
        import mate
        print("✓ MATE environment")
    except ImportError as e:
        print(f"✗ MATE environment: {e}")
        return False
    
    return True

def test_qplex_imports():
    """Test if QPLEX components can be imported."""
    print("\nTesting QPLEX imports...")
    
    try:
        from networks.base_networks import QPLEXNetwork, MixingNetwork
        print("✓ Base networks")
    except ImportError as e:
        print(f"✗ Base networks: {e}")
        return False
    
    try:
        from networks.rnn_networks import RNNQNetwork, AttentionRNNQNetwork
        print("✓ RNN networks")
    except ImportError as e:
        print(f"✗ RNN networks: {e}")
        return False
    
    try:
        from algorithms.qplex.agent import QPLEXAgent
        print("✓ QPLEX agent")
    except ImportError as e:
        print(f"✗ QPLEX agent: {e}")
        return False
    
    try:
        from algorithms.qplex.learner import QPLEXLearner
        print("✓ QPLEX learner")
    except ImportError as e:
        print(f"✗ QPLEX learner: {e}")
        return False
    
    try:
        from algorithms.qplex.mixer import QPLEXMixingNetwork
        print("✓ QPLEX mixer")
    except ImportError as e:
        print(f"✗ QPLEX mixer: {e}")
        return False
    
    try:
        from algorithms.qplex.model import QPLEXModel
        print("✓ QPLEX model")
    except ImportError as e:
        print(f"✗ QPLEX model: {e}")
        return False
    
    return True

def test_environment():
    """Test if MATE environment can be created."""
    print("\nTesting MATE environment...")
    
    try:
        from mate.environment import MultiAgentTracking
        from mate.agents import GreedyTargetAgent
        
        # Create a simple environment
        env = MultiAgentTracking(config='mate/assets/MATE-4v4-9.yaml')
        print(f"✓ Environment created: {env}")
        print(f"  - Cameras: {env.num_cameras}")
        print(f"  - Targets: {env.num_targets}")
        print(f"  - Obstacles: {env.num_obstacles}")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successful")
        
        # Test step
        camera_actions = [0, 0, 0, 0]  # No-op actions
        target_actions = [[0, 0] for _ in range(env.num_targets)]
        actions = (camera_actions, target_actions)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        print("✓ Environment step successful")
        
        env.close()
        print("✓ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_network_creation():
    """Test if networks can be created."""
    print("\nTesting network creation...")
    
    try:
        import torch
        from networks.base_networks import QPLEXNetwork
        
        # Test QPLEX network creation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        network = QPLEXNetwork(
            obs_dim=100,
            action_dim=2,
            state_dim=200,
            n_agents=4,
            q_hidden_dims=[128, 128],
            mix_hidden_dims=[128, 128]
        ).to(device)
        
        print("✓ QPLEX network created")
        
        # Test forward pass
        batch_size = 2
        obs = torch.randn(batch_size, 4, 100).to(device)
        state = torch.randn(batch_size, 200).to(device)
        
        q_values, q_total, hidden = network(obs, state)
        
        print(f"✓ Forward pass successful")
        print(f"  - Q-values shape: {q_values.shape}")
        print(f"  - Q-total shape: {q_total.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network creation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("QPLEX MATE Installation Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test QPLEX imports
    if not test_qplex_imports():
        all_tests_passed = False
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test network creation
    if not test_network_creation():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✓ All tests passed! Installation is successful.")
        print("\nYou can now run:")
        print("  python example.py")
        print("  python train_qplex_mate.py --config configs/qplex_4v4_9.yaml")
    else:
        print("✗ Some tests failed. Please check the installation.")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
