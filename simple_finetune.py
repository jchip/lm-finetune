#!/usr/bin/env python3
"""
Simple fine-tuning script for local DialoGPT model.
"""

import os
import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json

def load_data(file_path):
    """Load JSONL data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    print("🚀 Starting simple fine-tuning...")
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    model_name = "microsoft/DialoGPT-small"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load training data
    print("📊 Loading training data...")
    try:
        data = load_data("data/sample_training.jsonl")
        print(f"✅ Loaded {len(data)} training samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print(f"✅ Model prepared for training")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/simple_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        warmup_ratio=0.03,
        weight_decay=0.01,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_dir="logs",
    )
    
    # Create trainer
    print("🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )
    
    # Start training
    print("🎯 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        print("✅ Model saved to models/simple_finetuned/")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return
    
    print("🎉 Fine-tuning completed!")

if __name__ == "__main__":
    main() 
