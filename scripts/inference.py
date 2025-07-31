#!/usr/bin/env python3
"""
Inference Script for Fine-tuned Models

This script provides inference capabilities for both HuggingFace and GGUF models.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.training_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference with fine-tuned models")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (HuggingFace directory or GGUF file)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["huggingface", "gguf"],
        default=None,
        help="Model type (auto-detected if not specified)"
    )
    
    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for inference"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=2048,
        help="Context length for the model"
    )
    
    # Model loading arguments
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision (HuggingFace only)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision (HuggingFace only)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads (GGUF only)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
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
    
    return parser.parse_args()


def detect_model_type(model_path: str) -> str:
    """Detect model type based on file extension and structure."""
    if model_path.endswith('.gguf'):
        return "gguf"
    elif os.path.isdir(model_path):
        # Check for HuggingFace model files
        required_files = ["config.json", "tokenizer.json"]
        if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
            return "huggingface"
    
    raise ValueError(f"Could not detect model type for: {model_path}")


def load_huggingface_model(model_path: str, args) -> Any:
    """Load HuggingFace model for inference."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        logging.info(f"Loading HuggingFace model from: {model_path}")
        
        # Determine quantization config
        quantization_config = None
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif args.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=args.device,
            torch_dtype=torch.float16,
        )
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Failed to load HuggingFace model: {e}")
        raise


def load_gguf_model(model_path: str, args) -> Any:
    """Load GGUF model for inference."""
    try:
        from llama_cpp import Llama
        
        logging.info(f"Loading GGUF model from: {model_path}")
        
        model = Llama(
            model_path=model_path,
            n_ctx=args.context_length,
            n_threads=args.threads,
            verbose=args.verbose
        )
        
        return model
        
    except Exception as e:
        logging.error(f"Failed to load GGUF model: {e}")
        raise


def generate_huggingface_response(model, tokenizer, prompt: str, args) -> str:
    """Generate response using HuggingFace model."""
    try:
        import torch
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.context_length
        )
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logging.error(f"Failed to generate response with HuggingFace model: {e}")
        raise


def generate_gguf_response(model, prompt: str, args) -> str:
    """Generate response using GGUF model."""
    try:
        response = model(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            echo=False
        )
        
        return response['choices'][0]['text']
        
    except Exception as e:
        logging.error(f"Failed to generate response with GGUF model: {e}")
        raise


def save_results(prompt: str, response: str, output_file: str):
    """Save inference results to file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting inference pipeline")
    logger.info(f"Arguments: {args}")
    
    # Detect model type if not specified
    if args.model_type is None:
        try:
            args.model_type = detect_model_type(args.model_path)
            logger.info(f"Detected model type: {args.model_type}")
        except ValueError as e:
            logger.error(f"Model type detection failed: {e}")
            sys.exit(1)
    
    # Load model
    try:
        if args.model_type == "huggingface":
            model, tokenizer = load_huggingface_model(args.model_path, args)
            logger.info("HuggingFace model loaded successfully")
        elif args.model_type == "gguf":
            model = load_gguf_model(args.model_path, args)
            logger.info("GGUF model loaded successfully")
        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Generate response
    try:
        logger.info(f"Generating response for prompt: {args.prompt}")
        
        if args.model_type == "huggingface":
            response = generate_huggingface_response(model, tokenizer, args.prompt, args)
        else:  # gguf
            response = generate_gguf_response(model, args.prompt, args)
        
        logger.info("Response generated successfully")
        
        # Print results
        print("\n" + "="*50)
        print("PROMPT:")
        print(args.prompt)
        print("\nRESPONSE:")
        print(response)
        print("="*50)
        
        # Save results if requested
        if args.output_file:
            save_results(args.prompt, response, args.output_file)
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        sys.exit(1)
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main() 
