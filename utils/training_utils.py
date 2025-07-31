"""
Training utilities for QLoRA fine-tuning.
"""

import logging
import os
import time
from typing import Dict, Optional, Any
import torch
from transformers import TrainingArguments
import wandb
from datetime import datetime


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def setup_wandb(
    project_name: str = "llm-finetune",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        run_name: W&B run name
        config: Configuration to log
    """
    if run_name is None:
        run_name = f"qlora-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config or {},
        tags=["qlora", "finetune", "llm"]
    )
    
    logger.info(f"W&B logging initialized: {project_name}/{run_name}")


def log_training_info(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log training information.
    
    Args:
        model: Model being trained
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
    """
    logger.info("=== Training Information ===")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    
    # Dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Tokenizer info
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Model max length: {tokenizer.model_max_length}")
    
    # Configuration
    if config:
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("==========================")


def create_training_arguments(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,
    evaluation_strategy: str = "steps",
    save_strategy: str = "steps",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    report_to: Optional[list] = None,
    logging_dir: str = "logs",
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments.
    
    Args:
        output_dir: Output directory
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        logging_steps: Logging frequency
        save_steps: Save frequency
        eval_steps: Evaluation frequency
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm
        evaluation_strategy: Evaluation strategy
        save_strategy: Save strategy
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric for best model selection
        greater_is_better: Whether metric is better when greater
        report_to: Reporting backends
        logging_dir: Logging directory
        **kwargs: Additional arguments
        
    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=max_grad_norm,
        report_to=report_to,
        logging_dir=logging_dir,
        **kwargs
    )


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check GPU memory usage.
    
    Returns:
        Dictionary with GPU memory information
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        memory_total = props.total_memory / 1024**3  # GB
        
        gpu_info[f"gpu_{i}"] = {
            "name": props.name,
            "memory_total_gb": memory_total,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "memory_free_gb": memory_total - memory_reserved,
            "memory_utilization_percent": (memory_reserved / memory_total) * 100
        }
    
    return gpu_info


def log_gpu_memory() -> None:
    """Log current GPU memory usage."""
    gpu_info = check_gpu_memory()
    
    if not gpu_info.get("gpu_available", True):
        logger.info("No GPU available")
        return
    
    logger.info("=== GPU Memory Usage ===")
    for gpu_id, info in gpu_info.items():
        if gpu_id == "gpu_available":
            continue
        logger.info(f"{gpu_id}: {info['name']}")
        logger.info(f"  Total: {info['memory_total_gb']:.2f} GB")
        logger.info(f"  Allocated: {info['memory_allocated_gb']:.2f} GB")
        logger.info(f"  Reserved: {info['memory_reserved_gb']:.2f} GB")
        logger.info(f"  Free: {info['memory_free_gb']:.2f} GB")
        logger.info(f"  Utilization: {info['memory_utilization_percent']:.1f}%")
    logger.info("========================")


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    estimated_steps_per_second: float = 2.0
) -> Dict[str, Any]:
    """
    Estimate training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        estimated_steps_per_second: Estimated steps per second
        
    Returns:
        Dictionary with time estimates
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    estimated_seconds = total_steps / estimated_steps_per_second
    estimated_hours = estimated_seconds / 3600
    estimated_minutes = (estimated_seconds % 3600) / 60
    
    return {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_hours,
        "estimated_minutes": estimated_minutes,
        "estimated_time_str": f"{int(estimated_hours)}h {int(estimated_minutes)}m"
    }


class TrainingMonitor:
    """Monitor training progress and performance."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.start_time = None
        self.step_count = 0
        self.loss_history = []
        
    def start_training(self):
        """Start monitoring training."""
        self.start_time = time.time()
        self.step_count = 0
        self.loss_history = []
        logger.info("Training started")
        
    def log_step(self, step: int, loss: float, learning_rate: float):
        """Log training step information."""
        self.step_count = step
        self.loss_history.append(loss)
        
        if step % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            avg_loss = sum(self.loss_history[-self.log_interval:]) / len(self.loss_history[-self.log_interval:])
            
            logger.info(f"Step {step}: loss={loss:.4f}, avg_loss={avg_loss:.4f}, lr={learning_rate:.6f}, time={elapsed_time:.1f}s")
            
            # Log to W&B if available
            if wandb.run is not None:
                wandb.log({
                    "train/loss": loss,
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": learning_rate,
                    "train/step": step,
                    "train/elapsed_time": elapsed_time
                })
    
    def end_training(self):
        """End training monitoring."""
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        if wandb.run is not None:
            wandb.finish()


def create_sample_training_data(output_path: str, num_samples: int = 100) -> None:
    """
    Create sample training data for testing.
    
    Args:
        output_path: Path to save sample data
        num_samples: Number of samples to create
    """
    from utils.data_processing import create_sample_data
    create_sample_data(output_path, num_samples) 
