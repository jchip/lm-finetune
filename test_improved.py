#!/usr/bin/env python3
"""
Test the improved fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("🧪 Testing improved fine-tuned model...")
    
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
    
    # Load fine-tuned model
    print("\n📥 Loading fine-tuned model...")
    try:
        model = PeftModel.from_pretrained(base_model, "models/working_finetuned")
        print("✅ Fine-tuned model loaded")
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "What is artificial intelligence?",
            "What is data science?"
        ]
        
        for prompt in test_prompts:
            print(f"\n🎯 Testing: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Add repetition penalty
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"Response: '{response}'")
            print(f"Response length: {len(response)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
