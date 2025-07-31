#!/usr/bin/env python3
"""
Complete Example Pipeline for LLM Fine-tuning

This script demonstrates the complete pipeline from data preparation to inference.
It's designed for testing and learning purposes.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.training_utils import setup_logging, log_gpu_memory
from utils.data_processing import create_sample_data, analyze_dataset
from utils.model_utils import get_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Complete LLM fine-tuning example")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/DialoGPT-medium",  # Smaller model for testing
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of sample training examples to create"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run inference"
    )
    parser.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip GGUF conversion"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def create_training_data(num_samples: int) -> str:
    """Create sample training data."""
    output_path = "data/example_training.jsonl"
    os.makedirs("data", exist_ok=True)
    
    logging.info(f"Creating {num_samples} sample training examples...")
    create_sample_data(output_path, num_samples)
    
    return output_path


def run_fine_tuning(data_path: str, args) -> str:
    """Run the fine-tuning process."""
    logging.info("Starting fine-tuning process...")
    
    # Build command
    cmd = [
        sys.executable, "scripts/finetune.py",
        "--model_name_or_path", args.model_name,
        "--data_path", data_path,
        "--output_dir", "models/example_finetuned",
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--max_seq_length", str(args.max_length),
        "--logging_steps", "5",
        "--save_steps", "50",
        "--eval_steps", "50",
        "--config_preset", "memory_efficient"
    ]
    
    logging.info(f"Running fine-tuning command: {' '.join(cmd)}")
    
    # Run fine-tuning
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("Fine-tuning completed successfully")
        return "models/example_finetuned/merged"
    except subprocess.CalledProcessError as e:
        logging.error(f"Fine-tuning failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        raise


def run_gguf_conversion(model_path: str, args) -> str:
    """Run GGUF conversion."""
    logging.info("Starting GGUF conversion...")
    
    output_path = "models/example_model.gguf"
    
    # Build command
    cmd = [
        sys.executable, "scripts/convert_to_gguf.py",
        "--model_path", model_path,
        "--output_path", output_path,
        "--quantization", "q4_k_m",
        "--context_length", str(args.max_length),
        "--force"
    ]
    
    logging.info(f"Running GGUF conversion command: {' '.join(cmd)}")
    
    # Run conversion
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("GGUF conversion completed successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"GGUF conversion failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        raise


def run_inference(model_path: str, model_type: str, args):
    """Run inference with the fine-tuned model."""
    logging.info("Running inference...")
    
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does backpropagation work?",
        "What is deep learning?"
    ]
    
    for prompt in test_prompts:
        logging.info(f"Testing prompt: {prompt}")
        
        # Build command
        cmd = [
            sys.executable, "scripts/inference.py",
            "--model_path", model_path,
            "--model_type", model_type,
            "--prompt", prompt,
            "--max_tokens", "100",
            "--temperature", "0.7"
        ]
        
        logging.info(f"Running inference command: {' '.join(cmd)}")
        
        # Run inference
        import subprocess
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("Inference completed successfully")
            logging.info(f"Response: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Inference failed: {e}")
            logging.error(f"Error output: {e.stderr}")


def main():
    """Main example function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting complete LLM fine-tuning example")
    logger.info(f"Arguments: {args}")
    
    # Log system information
    log_gpu_memory()
    
    start_time = time.time()
    
    try:
        # Step 1: Create training data
        logger.info("=" * 50)
        logger.info("STEP 1: Creating training data")
        logger.info("=" * 50)
        
        data_path = create_training_data(args.num_samples)
        logger.info(f"Training data created at: {data_path}")
        
        # Step 2: Run fine-tuning (if not skipped)
        if not args.skip_training:
            logger.info("=" * 50)
            logger.info("STEP 2: Running fine-tuning")
            logger.info("=" * 50)
            
            model_path = run_fine_tuning(data_path, args)
            logger.info(f"Fine-tuned model saved at: {model_path}")
        else:
            logger.info("Skipping fine-tuning as requested")
            model_path = "models/example_finetuned/merged"  # Assume it exists
        
        # Step 3: Convert to GGUF (if not skipped)
        gguf_path = None
        if not args.skip_conversion:
            logger.info("=" * 50)
            logger.info("STEP 3: Converting to GGUF")
            logger.info("=" * 50)
            
            gguf_path = run_gguf_conversion(model_path, args)
            logger.info(f"GGUF model saved at: {gguf_path}")
        else:
            logger.info("Skipping GGUF conversion as requested")
        
        # Step 4: Run inference
        logger.info("=" * 50)
        logger.info("STEP 4: Running inference")
        logger.info("=" * 50)
        
        # Test with HuggingFace model
        if os.path.exists(model_path):
            logger.info("Testing HuggingFace model...")
            run_inference(model_path, "huggingface", args)
        
        # Test with GGUF model
        if gguf_path and os.path.exists(gguf_path):
            logger.info("Testing GGUF model...")
            run_inference(gguf_path, "gguf", args)
        
        # Summary
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("EXAMPLE PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Training data: {data_path}")
        logger.info(f"Fine-tuned model: {model_path}")
        if gguf_path:
            logger.info(f"GGUF model: {gguf_path}")
        
        logger.info("\nNext steps:")
        logger.info("1. Try different models and datasets")
        logger.info("2. Experiment with different hyperparameters")
        logger.info("3. Use your own training data")
        logger.info("4. Deploy the model for production use")
        
    except Exception as e:
        logger.error(f"Example pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 
