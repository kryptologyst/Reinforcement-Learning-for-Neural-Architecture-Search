#!/usr/bin/env python3
"""
Setup script for RL NAS project.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Setup the project."""
    print("🚀 RL NAS Project Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install basic dependencies
    basic_deps = [
        "torch",
        "numpy", 
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm"
    ]
    
    print("\n📦 Installing basic dependencies...")
    for dep in basic_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    # Create directories
    print("\n📁 Creating project directories...")
    directories = ["configs", "logs", "checkpoints", "data", "visualizations"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Test the project
    print("\n🧪 Testing project...")
    if run_command("python test_project.py", "Running project tests"):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Create configs: python src/train.py --create_configs")
        print("2. Start training: python src/train.py --algorithm ppo --total_timesteps 10000")
        print("3. Run full tests: python tests/test_components.py")
        return True
    else:
        print("\n❌ Setup completed with errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
