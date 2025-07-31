#!/usr/bin/env python3
"""
Debug script to understand the fine-tuned model behavior.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("🔍 Debugging fine-tuned model...")
    
    # Load base model
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-small",
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Base model loaded")
    
    # Test base model first
    print("\n🧪 Testing base model...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = base_model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(test_prompt):
        response = response[len(test_prompt):].strip()
    
    print(f"Base model response: '{response}'")
    
    # Load fine-tuned model
    print("\n📥 Loading fine-tuned model...")
    try:
        model = PeftModel.from_pretrained(base_model, "models/simple_finetuned")
        print("✅ Fine-tuned model loaded")
        
        # Test fine-tuned model
        print("\n🧪 Testing fine-tuned model...")
        test_prompts = [
            "What is machine learning?",
            "Hello, how are you?",
            "Explain neural networks"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"Response: '{response}'")
            print(f"Response length: {len(response)}")
            print(f"Response tokens: {len(tokenizer.encode(response))}")
            
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
