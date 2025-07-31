#!/usr/bin/env python3
"""
Environment Setup Script for LLM Fine-tuning

This script helps set up the development environment for the LLM fine-tuning project.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.training_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup development environment")
    
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check environment without installing"
    )
    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Install Python dependencies"
    )
    parser.add_argument(
        "--install_cuda",
        action="store_true",
        help="Install CUDA toolkit (Windows only)"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample training data"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logging.error(f"Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    logging.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_git():
    """Check if Git is installed."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Git version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Git not found. Please install Git.")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            logging.info(f"CUDA available: {cuda_version}")
            logging.info(f"GPU devices: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logging.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
            return True
        else:
            logging.warning("CUDA not available. Training will use CPU (slow).")
            return False
    except ImportError:
        logging.warning("PyTorch not installed. Cannot check CUDA.")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "accelerate",
        "llama-cpp-python"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logging.info(f"✓ {package}")
        except ImportError:
            logging.warning(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing packages: {missing_packages}")
        return False
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    try:
        logging.info("Installing Python dependencies...")
        
        # Install from requirements.txt
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
        
        logging.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False


def install_cuda_toolkit():
    """Install CUDA toolkit (Windows only)."""
    if os.name != 'nt':
        logging.warning("CUDA toolkit installation is for Windows only")
        return False
    
    try:
        logging.info("Installing CUDA toolkit...")
        logging.info("Please download and install CUDA Toolkit from:")
        logging.info("https://developer.nvidia.com/cuda-downloads")
        logging.info("Choose Windows x86_64 version")
        
        # Check if CUDA is already installed
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            logging.info(f"CUDA already installed at: {cuda_path}")
            return True
        
        logging.warning("CUDA toolkit not found. Please install manually.")
        return False
        
    except Exception as e:
        logging.error(f"Failed to check CUDA installation: {e}")
        return False


def create_sample_data():
    """Create sample training data."""
    try:
        from utils.data_processing import create_sample_data
        
        output_path = "data/sample_training.jsonl"
        os.makedirs("data", exist_ok=True)
        
        logging.info(f"Creating sample training data at: {output_path}")
        create_sample_data(output_path, num_samples=100)
        
        logging.info("Sample data created successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to create sample data: {e}")
        return False


def check_visual_studio():
    """Check Visual Studio Build Tools."""
    try:
        # Check for Visual Studio Build Tools
        vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools"
        ]
        
        for path in vs_paths:
            if os.path.exists(path):
                logging.info(f"Visual Studio Build Tools found at: {path}")
                return True
        
        logging.warning("Visual Studio Build Tools not found.")
        logging.info("Please install Visual Studio Build Tools with C++ and CMake components.")
        return False
        
    except Exception as e:
        logging.error(f"Failed to check Visual Studio: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions."""
    print("\n" + "="*60)
    print("LLM Fine-tuning Environment Setup")
    print("="*60)
    
    print("\n1. Prerequisites:")
    print("   - Python 3.9+")
    print("   - CUDA Toolkit (for GPU acceleration)")
    
    print("\n2. Installation Steps:")
    print("   a. Install Python 3.9+ from python.org")
    print("   d. Install CUDA Toolkit (optional, for GPU)")
    print("   e. Run: python scripts/setup_env.py --install_deps")
    
    print("\n3. Verify Installation:")
    print("   Run: python scripts/setup_env.py --check_only")
    
    print("\n4. Create Sample Data:")
    print("   Run: python scripts/setup_env.py --create_sample_data")
    
    print("\n5. Start Training:")
    print("   Run: python scripts/finetune.py --data_path data/sample_training.jsonl")
    
    print("="*60)


def main():
    """Main setup function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting environment setup check")
    
    # Check prerequisites
    checks_passed = 0
    total_checks = 0
    
    # Python version
    total_checks += 1
    if check_python_version():
        checks_passed += 1
    
    # Git
    total_checks += 1
    if check_git():
        checks_passed += 1
    
    # Visual Studio Build Tools
    total_checks += 1
    if check_visual_studio():
        checks_passed += 1
    
    # CUDA
    total_checks += 1
    if check_cuda():
        checks_passed += 1
    
    # Dependencies
    total_checks += 1
    if check_dependencies():
        checks_passed += 1
    
    # Summary
    logger.info(f"Environment check: {checks_passed}/{total_checks} passed")
    
    if checks_passed < total_checks:
        logger.warning("Some checks failed. Please install missing components.")
        print_setup_instructions()
    
    # Install dependencies if requested
    if args.install_deps:
        if install_dependencies():
            logger.info("Dependencies installed successfully")
        else:
            logger.error("Failed to install dependencies")
            return False
    
    # Install CUDA if requested
    if args.install_cuda:
        if install_cuda_toolkit():
            logger.info("CUDA toolkit installation completed")
        else:
            logger.warning("CUDA toolkit installation failed")
    
    # Create sample data if requested
    if args.create_sample_data:
        if create_sample_data():
            logger.info("Sample data created successfully")
        else:
            logger.error("Failed to create sample data")
            return False
    
    if checks_passed == total_checks:
        logger.info("Environment setup completed successfully!")
        logger.info("You can now start fine-tuning with:")
        logger.info("python scripts/finetune.py --data_path data/sample_training.jsonl")
    else:
        logger.warning("Environment setup incomplete. Please address the issues above.")
    
    return checks_passed == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
