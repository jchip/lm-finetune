#!/usr/bin/env python3
"""
Basic test script to verify the pipeline works.
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from utils.data_processing import load_jsonl_data, create_sample_data
        from utils.model_utils import load_model_and_tokenizer
        from utils.training_utils import setup_logging
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    try:
        from utils.data_processing import load_jsonl_data
        data = load_jsonl_data('data/sample_training.jsonl')
        print(f"✅ Loaded {len(data)} samples")
        print(f"Sample: {data[0]['text'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality."""
    print("\nTesting model loading...")
    try:
        from utils.model_utils import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(
            "microsoft/DialoGPT-small",
            load_in_4bit=False  # Disable quantization for testing
        )
        print(f"✅ Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation."""
    print("\nTesting sample data creation...")
    try:
        from utils.data_processing import create_sample_data
        create_sample_data("data/test_sample.jsonl", num_samples=5)
        print("✅ Sample data created successfully")
        return True
    except Exception as e:
        print(f"❌ Sample data creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running basic functionality tests...")
    
    tests = [
        test_imports,
        test_data_loading,
        test_model_loading,
        test_sample_data_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Try fine-tuning with a small model:")
        print("   python scripts/finetune.py --model_name_or_path microsoft/DialoGPT-small --data_path data/sample_training.jsonl --output_dir models/my_test --num_train_epochs 1 --target_modules c_attn c_proj")
        print("\n2. For a larger model (requires more memory):")
        print("   python scripts/finetune.py --model_name_or_path meta-llama/Llama-2-7b-hf --data_path data/sample_training.jsonl --output_dir models/my_llama --num_train_epochs 1")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 
