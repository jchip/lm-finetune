#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for 7B LLMs

This script implements the complete QLoRA fine-tuning pipeline for 7B language models.
It handles model loading, data preparation, training, and model saving.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.training_config import TrainingConfig, get_default_config, get_memory_efficient_config
from utils.data_processing import prepare_dataset, analyze_dataset
from utils.model_utils import (
    load_model_and_tokenizer,
    prepare_model_for_training,
    create_trainer,
    save_model_and_tokenizer,
    get_model_info
)
from utils.training_utils import (
    setup_logging,
    setup_wandb,
    log_training_info,
    log_gpu_memory,
    estimate_training_time,
    TrainingMonitor
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for 7B LLMs")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSONL format)"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/finetuned",
        help="Output directory for model and checkpoints"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per device for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="*",
        default=None,
        help="Target modules for LoRA (default: auto-detect)"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save frequency"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="*",
        default=None,
        help="Reporting backends (wandb, tensorboard)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-finetune",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name"
    )
    
    # Configuration presets
    parser.add_argument(
        "--config_preset",
        type=str,
        choices=["default", "memory_efficient", "high_quality"],
        default="default",
        help="Configuration preset"
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
        "--save_adapter",
        action="store_true",
        default=True,
        help="Save LoRA adapter"
    )
    parser.add_argument(
        "--save_merged",
        action="store_true",
        default=True,
        help="Save merged model"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QLoRA fine-tuning pipeline")
    logger.info(f"Arguments: {args}")
    
    # Get configuration based on preset
    if args.config_preset == "default":
        config = get_default_config()
    elif args.config_preset == "memory_efficient":
        config = get_memory_efficient_config()
    elif args.config_preset == "high_quality":
        config = get_high_quality_config()
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    config.model_name_or_path = args.model_name_or_path
    config.trust_remote_code = args.trust_remote_code
    config.num_train_epochs = args.num_train_epochs
    config.per_device_train_batch_size = args.per_device_train_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    if args.target_modules:
        config.target_modules = args.target_modules
    config.max_seq_length = args.max_seq_length
    config.output_dir = args.output_dir
    config.logging_steps = args.logging_steps
    config.save_steps = args.save_steps
    config.eval_steps = args.eval_steps
    config.report_to = args.report_to
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup W&B if requested
    if args.report_to and "wandb" in args.report_to:
        setup_wandb(args.wandb_project, args.wandb_run_name, config.to_dict())
    
    # Log GPU memory
    log_gpu_memory()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        load_in_4bit=config.load_in_4bit,
        device_map="auto"
    )
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset_dict = prepare_dataset(
        file_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        validation_split=args.validation_split,
        padding=config.padding,
        truncation=config.truncation
    )
    
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation")
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    train_stats = analyze_dataset(train_dataset, tokenizer)
    logger.info(f"Training dataset stats: {train_stats}")
    
    if eval_dataset:
        eval_stats = analyze_dataset(eval_dataset, tokenizer)
        logger.info(f"Evaluation dataset stats: {eval_stats}")
    
    # Log training information
    log_training_info(model, tokenizer, train_dataset, eval_dataset, config.to_dict())
    
    # Estimate training time
    time_estimate = estimate_training_time(
        num_samples=len(train_dataset),
        batch_size=config.per_device_train_batch_size,
        num_epochs=config.num_train_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    logger.info(f"Estimated training time: {time_estimate['estimated_time_str']}")
    
    # Prepare model for training
    logger.info("Preparing model for QLoRA training...")
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_training(model, lora_config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=None,  # Will use config defaults
        **config.to_dict()
    )
    
    # Start training
    logger.info("Starting training...")
    monitor = TrainingMonitor(log_interval=args.logging_steps)
    monitor.start_training()
    
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save model
    logger.info("Saving model...")
    save_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=config.output_dir,
        save_adapter=args.save_adapter,
        save_merged=args.save_merged
    )
    
    # Log final model info
    model_info = get_model_info(model)
    logger.info(f"Final model info: {model_info}")
    
    logger.info("Fine-tuning pipeline completed successfully!")


if __name__ == "__main__":
    main() 
