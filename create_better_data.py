#!/usr/bin/env python3
"""
Create better training data for DialoGPT.
"""

import json

def create_dialogpt_data():
    """Create training data in DialoGPT format."""
    
    # DialoGPT format: just the conversation without special tokens
    conversations = [
        {
            "text": "What is machine learning? Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "text": "Explain neural networks. Neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes that process information."
        },
        {
            "text": "How does deep learning work? Deep learning uses multiple layers of neural networks to learn hierarchical representations of data, enabling complex pattern recognition."
        },
        {
            "text": "What is artificial intelligence? Artificial intelligence is the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition."
        },
        {
            "text": "What is data science? Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge from structured and unstructured data."
        },
        {
            "text": "Explain supervised learning. Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on new, unseen data."
        },
        {
            "text": "What is unsupervised learning? Unsupervised learning is a type of machine learning where the algorithm finds hidden patterns in data without any labeled examples."
        },
        {
            "text": "How do you train a model? To train a model, you need to collect data, preprocess it, choose an algorithm, set hyperparameters, and iteratively optimize the model's performance."
        },
        {
            "text": "What is overfitting? Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor generalization on new data."
        },
        {
            "text": "What is underfitting? Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test sets."
        }
    ]
    
    # Create more variations
    additional_data = []
    for i in range(10):  # Create 10 more variations of each
        for conv in conversations:
            additional_data.append(conv)
    
    # Save to file
    with open("data/dialogpt_training.jsonl", "w", encoding="utf-8") as f:
        for item in additional_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Created {len(additional_data)} training samples")
    print("✅ Saved to data/dialogpt_training.jsonl")

if __name__ == "__main__":
    create_dialogpt_data() 
