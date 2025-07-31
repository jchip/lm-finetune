"""
Training configuration for QLoRA fine-tuning of 7B LLMs.
"""

from dataclasses import dataclass
from typing import Optional, List
import os


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    
    # Model settings
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = True
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Training settings
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    
    # Data settings
    max_seq_length: int = 2048
    padding: str = "max_length"
    truncation: bool = True
    
    # Output settings
    output_dir: str = "models/finetuned"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Optimization settings
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # Monitoring
    logging_dir: str = "logs"
    report_to: Optional[List[str]] = None  # ["wandb", "tensorboard"]
    
    # GGUF conversion settings
    gguf_quantization: str = "q4_k_m"
    
    def __post_init__(self):
        """Set default target modules for common model architectures."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary for training arguments."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "weight_decay": self.weight_decay,
            "logging_dir": self.logging_dir,
            "report_to": self.report_to,
        }


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_memory_efficient_config() -> TrainingConfig:
    """Get memory-efficient configuration for limited GPU memory."""
    config = TrainingConfig()
    config.per_device_train_batch_size = 1
    config.per_device_eval_batch_size = 1
    config.gradient_accumulation_steps = 8
    config.max_seq_length = 1024
    return config


def get_high_quality_config() -> TrainingConfig:
    """Get high-quality configuration for better results."""
    config = TrainingConfig()
    config.lora_r = 32
    config.lora_alpha = 64
    config.learning_rate = 1e-4
    config.num_train_epochs = 5
    config.max_seq_length = 4096
    return config 
