#!/usr/bin/env python3
"""
Simple test to verify model loading and basic functionality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("🧪 Testing model loading...")
    
    try:
        # Load model and tokenizer
        print("📥 Loading DialoGPT-small...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",
            trust_remote_code=True
        )
        print("✅ Model and tokenizer loaded successfully!")
        
        # Test basic inference
        print("🎯 Testing basic inference...")
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {response}")
        print("✅ Basic inference works!")
        
        # Test with sample data
        print("📊 Testing with sample data...")
        with open("data/sample_training.jsonl", "r") as f:
            sample = f.readline().strip()
            data = eval(sample)
            text = data["text"]
            print(f"Sample text: {text[:100]}...")
        
        print("✅ Sample data loaded!")
        
        print("🎉 All tests passed! The model is ready for fine-tuning.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
