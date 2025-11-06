#!/usr/bin/env python3
"""
Simple test script to verify the RL NAS project works correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    try:
        from agents.nas_controller import NASController, ArchitectureConfig
        from agents.rl_algorithms import PPOAgent, RLConfig
        from envs.environments import EnvironmentFactory
        from utils.config import TrainingConfig
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_nas_controller():
    """Test NAS controller functionality."""
    print("\nTesting NAS Controller...")
    try:
        from agents.nas_controller import NASController, ArchitectureConfig
        
        config = ArchitectureConfig(n_layers=2)
        controller = NASController(config, device="cpu")
        
        # Generate architecture
        architecture, log_prob, info = controller.forward()
        
        print(f"✅ Generated architecture: {architecture}")
        print(f"✅ Log probability: {log_prob.item():.4f}")
        print(f"✅ Controller has {sum(p.numel() for p in controller.parameters())} parameters")
        return True
    except Exception as e:
        print(f"❌ NAS Controller error: {e}")
        return False

def test_environments():
    """Test environment functionality."""
    print("\nTesting Environments...")
    try:
        from envs.environments import EnvironmentFactory
        
        # Test GridWorld
        env = EnvironmentFactory.create_environment("gridworld", size=3)
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"✅ GridWorld: State shape {state.shape}, Reward {reward:.2f}")
        
        # Test CartPole
        env = EnvironmentFactory.create_environment("cartpole")
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"✅ CartPole: State shape {state.shape}, Reward {reward:.2f}")
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_rl_algorithm():
    """Test RL algorithm functionality."""
    print("\nTesting RL Algorithm...")
    try:
        from agents.nas_controller import NASController, ArchitectureConfig
        from agents.rl_algorithms import PPOAgent, RLConfig
        
        config = ArchitectureConfig()
        controller = NASController(config, device="cpu")
        rl_config = RLConfig(device="cpu")
        agent = PPOAgent(controller, rl_config)
        
        # Test action selection
        state = torch.tensor([[0.0]])
        action, log_prob, value = agent.select_action(state)
        
        print(f"✅ PPO Agent: Action {action}, Log prob {log_prob.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ RL Algorithm error: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\nTesting Configuration...")
    try:
        from utils.config import TrainingConfig, ConfigManager
        import tempfile
        import os
        
        # Create config
        config = TrainingConfig(algorithm="ppo", total_timesteps=1000)
        
        # Test saving/loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigManager.save_config(config, temp_path)
            loaded_config = ConfigManager.load_config(temp_path)
            
            print(f"✅ Config save/load: Algorithm {loaded_config.algorithm}")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RL NAS Project Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_nas_controller,
        test_environments,
        test_rl_algorithm,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create configs: python src/train.py --create_configs")
        print("3. Start training: python src/train.py --algorithm ppo --total_timesteps 10000")
        print("4. Run tests: python tests/test_components.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
