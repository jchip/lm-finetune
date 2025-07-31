#!/usr/bin/env python3
"""
GGUF Conversion Script for Fine-tuned Models

This script converts fine-tuned models to GGUF format for efficient inference with llama.cpp.
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
    parser = argparse.ArgumentParser(description="Convert fine-tuned model to GGUF format")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (merged model directory)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for GGUF file"
    )
    
    # Conversion arguments
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0"],
        help="Quantization level for GGUF conversion"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=2048,
        help="Context length for the model"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use (default: auto)"
    )
    
    # Advanced arguments
    parser.add_argument(
        "--rope_freq_base",
        type=float,
        default=10000.0,
        help="RoPE frequency base"
    )
    parser.add_argument(
        "--rope_freq_scale",
        type=float,
        default=1.0,
        help="RoPE frequency scale"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes"
    )
    
    # Other arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite output file"
    )
    
    return parser.parse_args()


def check_llama_cpp_installation():
    """Check if llama.cpp is properly installed."""
    try:
        result = subprocess.run(
            ["llama-cpp-python", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"llama-cpp-python version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("llama-cpp-python not found. Please install it first.")
        return False


def check_model_path(model_path: str) -> bool:
    """Check if model path exists and contains required files."""
    if not os.path.exists(model_path):
        logging.error(f"Model path does not exist: {model_path}")
        return False
    
    # Check for required files
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"Missing required files in model path: {missing_files}")
        return False
    
    # Check for model files
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
    if not model_files:
        logging.error("No model weight files found (.bin or .safetensors)")
        return False
    
    logging.info(f"Found model files: {model_files}")
    return True


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    context_length: int = 2048,
    threads: int = None,
    rope_freq_base: float = 10000.0,
    rope_freq_scale: float = 1.0,
    parallel: int = 1,
    force: bool = False
) -> bool:
    """
    Convert model to GGUF format using llama.cpp.
    
    Args:
        model_path: Path to the model
        output_path: Output path for GGUF file
        quantization: Quantization level
        context_length: Context length
        threads: Number of threads
        rope_freq_base: RoPE frequency base
        rope_freq_scale: RoPE frequency scale
        parallel: Number of parallel processes
        force: Force overwrite output file
        
    Returns:
        True if conversion successful
    """
    # Check if output file exists
    if os.path.exists(output_path) and not force:
        logging.error(f"Output file already exists: {output_path}. Use --force to overwrite.")
        return False
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build conversion command
    cmd = [
        "python", "-m", "llama_cpp.convert",
        model_path,
        "--outfile", output_path,
        "--outtype", quantization,
        "--context-length", str(context_length),
        "--rope-freq-base", str(rope_freq_base),
        "--rope-freq-scale", str(rope_freq_scale),
        "--parallel", str(parallel)
    ]
    
    if threads:
        cmd.extend(["--threads", str(threads)])
    
    logging.info(f"Running conversion command: {' '.join(cmd)}")
    
    try:
        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logging.info("Conversion completed successfully")
        logging.debug(f"Conversion output: {result.stdout}")
        
        # Check if output file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logging.info(f"GGUF file created: {output_path} ({file_size:.2f} MB)")
            return True
        else:
            logging.error("Conversion completed but output file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Conversion failed with error code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Conversion failed with exception: {e}")
        return False


def validate_gguf_file(gguf_path: str) -> bool:
    """Validate the generated GGUF file."""
    try:
        # Try to load the GGUF file with llama-cpp-python
        from llama_cpp import Llama
        
        model = Llama(
            model_path=gguf_path,
            n_ctx=512,  # Small context for validation
            n_threads=1
        )
        
        # Test a simple inference
        response = model("Hello", max_tokens=10, temperature=0.0)
        
        logging.info("GGUF file validation successful")
        logging.info(f"Test response: {response}")
        return True
        
    except Exception as e:
        logging.error(f"GGUF file validation failed: {e}")
        return False


def main():
    """Main conversion function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GGUF conversion pipeline")
    logger.info(f"Arguments: {args}")
    
    # Check llama.cpp installation
    if not check_llama_cpp_installation():
        logger.error("llama-cpp-python not available. Please install it first.")
        sys.exit(1)
    
    # Check model path
    if not check_model_path(args.model_path):
        logger.error("Invalid model path")
        sys.exit(1)
    
    # Convert to GGUF
    logger.info("Starting GGUF conversion...")
    success = convert_to_gguf(
        model_path=args.model_path,
        output_path=args.output_path,
        quantization=args.quantization,
        context_length=args.context_length,
        threads=args.threads,
        rope_freq_base=args.rope_freq_base,
        rope_freq_scale=args.rope_freq_scale,
        parallel=args.parallel,
        force=args.force
    )
    
    if not success:
        logger.error("GGUF conversion failed")
        sys.exit(1)
    
    # Validate the converted file
    logger.info("Validating GGUF file...")
    if validate_gguf_file(args.output_path):
        logger.info("GGUF conversion completed successfully!")
    else:
        logger.error("GGUF file validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 
