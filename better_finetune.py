#!/usr/bin/env python3
"""
Improved fine-tuning script with better configuration.
"""

import os
import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json
import torch

def load_data(file_path):
    """Load JSONL data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    print("🚀 Starting improved fine-tuning...")
    
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
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
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
    
    # Configure LoRA with better settings
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=32,  # Increased rank
        lora_alpha=64,  # Increased alpha
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
    
    # Better training arguments
    training_args = TrainingArguments(
        output_dir="models/better_finetuned",
        num_train_epochs=3,  # More epochs
        per_device_train_batch_size=2,  # Larger batch size
        gradient_accumulation_steps=2,
        learning_rate=5e-4,  # Higher learning rate
        logging_steps=5,
        save_steps=25,
        warmup_ratio=0.1,  # More warmup
        weight_decay=0.01,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_dir="logs",
        remove_unused_columns=False,  # Important for SFTTrainer
    )
    
    # Create trainer with better configuration
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
        print("✅ Model saved to models/better_finetuned/")
        
        # Test the model immediately
        print("🧪 Testing the model...")
        test_prompt = "What is machine learning?"
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(test_prompt):
            response = response[len(test_prompt):].strip()
        
        print(f"Test prompt: {test_prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return
    
    print("🎉 Improved fine-tuning completed!")

if __name__ == "__main__":
    main() 
