#!/usr/bin/env python3
"""
Working fine-tuning script with proper configuration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json

def main():
    print("🚀 Starting working fine-tuning...")
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-small",
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded successfully!")
    
    # Create simple training data
    print("📊 Creating training data...")
    training_data = [
        {"text": "What is machine learning? Machine learning is a subset of artificial intelligence."},
        {"text": "Explain neural networks. Neural networks are computational models inspired by biology."},
        {"text": "How does deep learning work? Deep learning uses multiple layers of neural networks."},
        {"text": "What is AI? Artificial intelligence is the simulation of human intelligence in machines."},
        {"text": "What is data science? Data science uses scientific methods to extract knowledge from data."},
    ] * 20  # Repeat 20 times for more data
    
    dataset = Dataset.from_list(training_data)
    print(f"✅ Created {len(dataset)} training samples")
    
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
    
    # Prepare model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print(f"✅ Model prepared for training")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/working_finetuned",
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,  # Higher learning rate
        logging_steps=1,
        save_steps=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_dir="logs",
        remove_unused_columns=False,
    )
    
    # Create trainer
    print("🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )
    
    # Train
    print("🎯 Starting training...")
    trainer.train()
    
    # Save
    print("💾 Saving model...")
    trainer.save_model()
    
    # Test immediately
    print("🧪 Testing model...")
    test_prompt = "What is machine learning?"
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(test_prompt):
        response = response[len(test_prompt):].strip()
    
    print(f"Test prompt: {test_prompt}")
    print(f"Response: '{response}'")
    
    print("🎉 Working fine-tuning completed!")

if __name__ == "__main__":
    main() 
