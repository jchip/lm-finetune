#!/usr/bin/env python3
"""
Test the fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("🧪 Testing fine-tuned model...")
    
    # Load the fine-tuned model
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-small",
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, "models/simple_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", trust_remote_code=True)
    
    print("✅ Fine-tuned model loaded successfully!")
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        "What is artificial intelligence?"
    ]
    
    for prompt in test_prompts:
        print(f"\n🎯 Testing: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main() 
