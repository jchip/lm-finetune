"""
Model utilities for QLoRA fine-tuning.
"""

import logging
import os
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer


logger = logging.getLogger(__name__)


def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for quantization.
    
    Args:
        load_in_4bit: Whether to load model in 4-bit precision
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        bnb_4bit_quant_type: Quantization type for 4-bit
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        BitsAndBytesConfig
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM
) -> LoraConfig:
    """
    Create LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: Target modules for LoRA
        lora_dropout: LoRA dropout rate
        bias: Bias type
        task_type: Task type for LoRA
        
    Returns:
        LoraConfig
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def load_model_and_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    device_map: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with quantization.
    
    Args:
        model_name_or_path: Model name or path
        trust_remote_code: Whether to trust remote code
        load_in_4bit: Whether to load in 4-bit precision
        bnb_config: BitsAndBytes configuration
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name_or_path}")
    
    # Create quantization config if not provided
    if bnb_config is None and load_in_4bit:
        bnb_config = create_bnb_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
        use_fast=False
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig
) -> AutoModelForCausalLM:
    """
    Prepare model for QLoRA training.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Prepared model with LoRA adapters
    """
    logger.info("Preparing model for QLoRA training")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("Model prepared for training")
    return model


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    eval_dataset=None,
    training_args: Optional[TrainingArguments] = None,
    **kwargs
) -> SFTTrainer:
    """
    Create SFTTrainer for fine-tuning.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        **kwargs: Additional arguments for SFTTrainer
        
    Returns:
        SFTTrainer instance
    """
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="models/finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_strategy="steps",
            warmup_ratio=0.03,
            weight_decay=0.01,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            report_to=None,
            logging_dir="logs",
        )
    
    # Filter out kwargs that aren't valid for SFTTrainer
    valid_sft_kwargs = {}
    invalid_keys = [
        'model_name_or_path', 'trust_remote_code', 'data_path', 'validation_split',
        'max_seq_length', 'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps', 'learning_rate',
        'lora_r', 'lora_alpha', 'lora_dropout', 'target_modules', 'load_in_4bit',
        'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant',
        'logging_steps', 'save_steps', 'eval_steps', 'report_to', 'wandb_project',
        'wandb_run_name', 'config_preset', 'log_level', 'log_file', 'save_adapter',
        'save_merged', 'max_grad_norm', 'warmup_ratio', 'padding', 'truncation',
        'save_total_limit', 'gradient_checkpointing', 'optim', 'lr_scheduler_type',
        'weight_decay', 'logging_dir'
    ]
    
    for key, value in kwargs.items():
        if key not in invalid_keys:
            valid_sft_kwargs[key] = value
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        **valid_sft_kwargs
    )
    
    return trainer


def save_model_and_tokenizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    save_adapter: bool = True,
    save_merged: bool = True
) -> None:
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_adapter: Whether to save LoRA adapter
        save_merged: Whether to save merged model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")
    
    if save_adapter:
        # Save LoRA adapter
        adapter_dir = os.path.join(output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        logger.info(f"LoRA adapter saved to {adapter_dir}")
    
    if save_merged:
        # Save merged model
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        
        # Merge LoRA weights with base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to {merged_dir}")


def load_model_for_inference(
    model_path: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model for inference.
    
    Args:
        model_path: Path to model
        device: Device to load model on
        load_in_8bit: Whether to load in 8-bit
        load_in_4bit: Whether to load in 4-bit
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model for inference: {model_path}")
    
    # Determine quantization config
    quantization_config = None
    if load_in_4bit:
        quantization_config = create_bnb_config()
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    
    logger.info("Model loaded for inference")
    return model, tokenizer


def get_model_info(model: AutoModelForCausalLM) -> dict:
    """
    Get model information.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
        "model_type": type(model).__name__,
    } 
