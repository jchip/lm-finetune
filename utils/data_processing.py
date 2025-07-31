"""
Data processing utilities for LLM fine-tuning.
"""

import json
import logging
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import pandas as pd


logger = logging.getLogger(__name__)


def load_jsonl_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load JSONL data from file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries with text data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if 'text' not in item:
                        logger.warning(f"Line {line_num}: Missing 'text' field")
                        continue
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def validate_training_data(data: List[Dict[str, str]]) -> bool:
    """
    Validate training data format and content.
    
    Args:
        data: List of training samples
        
    Returns:
        True if data is valid
    """
    if not data:
        logger.error("No training data found")
        return False
    
    valid_samples = 0
    for i, sample in enumerate(data):
        if 'text' not in sample:
            logger.warning(f"Sample {i}: Missing 'text' field")
            continue
        
        text = sample['text']
        if not isinstance(text, str):
            logger.warning(f"Sample {i}: 'text' field is not a string")
            continue
        
        if len(text.strip()) == 0:
            logger.warning(f"Sample {i}: Empty text")
            continue
        
        valid_samples += 1
    
    logger.info(f"Validated {valid_samples}/{len(data)} samples")
    return valid_samples > 0


def create_dataset_from_jsonl(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    validation_split: float = 0.1
) -> DatasetDict:
    """
    Create HuggingFace dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        validation_split: Fraction of data to use for validation
        
    Returns:
        DatasetDict with train and validation splits
    """
    # Load and validate data
    data = load_jsonl_data(file_path)
    if not validate_training_data(data):
        raise ValueError("Invalid training data")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train/validation
    if validation_split > 0:
        dataset = dataset.train_test_split(test_size=validation_split, seed=42)
        return DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })
    else:
        return DatasetDict({'train': dataset})


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    padding: str = "max_length",
    truncation: bool = True
) -> Dict[str, List]:
    """
    Tokenize function for dataset processing.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate sequences
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt"
    )


def prepare_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    validation_split: float = 0.1,
    padding: str = "max_length",
    truncation: bool = True
) -> DatasetDict:
    """
    Prepare dataset for training.
    
    Args:
        file_path: Path to JSONL file
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        validation_split: Fraction of data to use for validation
        padding: Padding strategy
        truncation: Whether to truncate sequences
        
    Returns:
        Prepared DatasetDict
    """
    # Create dataset
    dataset_dict = create_dataset_from_jsonl(
        file_path, tokenizer, max_length, validation_split
    )
    
    # Tokenize datasets
    tokenized_datasets = {}
    for split_name, dataset in dataset_dict.items():
        tokenized_datasets[split_name] = dataset.map(
            lambda examples: tokenize_function(
                examples, tokenizer, max_length, padding, truncation
            ),
            batched=True,
            remove_columns=dataset.column_names
        )
    
    return DatasetDict(tokenized_datasets)


def create_sample_data(output_path: str, num_samples: int = 100) -> None:
    """
    Create sample training data for testing.
    
    Args:
        output_path: Path to save sample data
        num_samples: Number of sample entries to create
    """
    sample_data = []
    
    # Sample prompts and responses
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does backpropagation work?",
        "What is deep learning?",
        "Explain gradient descent",
        "What are transformers?",
        "How do attention mechanisms work?",
        "What is supervised learning?",
        "Explain unsupervised learning",
        "What is reinforcement learning?"
    ]
    
    responses = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes that process information.",
        "Backpropagation is an algorithm for training neural networks that calculates gradients of the loss function with respect to the network weights.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
        "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by following the negative gradient.",
        "Transformers are neural network architectures that use attention mechanisms to process sequential data, particularly effective for natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input when making predictions, improving performance on sequence tasks.",
        "Supervised learning is a machine learning approach where the model learns from labeled training data to make predictions on new, unseen data.",
        "Unsupervised learning is a machine learning approach where the model finds patterns in data without explicit labels or guidance.",
        "Reinforcement learning is a machine learning approach where an agent learns to make decisions by interacting with an environment and receiving rewards."
    ]
    
    for i in range(num_samples):
        prompt_idx = i % len(prompts)
        response_idx = i % len(responses)
        
        # Create conversation format
        text = f"<s>[INST] {prompts[prompt_idx]} [/INST] {responses[response_idx]}</s>"
        
        sample_data.append({"text": text})
    
    # Save to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created sample data with {num_samples} samples at {output_path}")


def analyze_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dict[str, Union[int, float]]:
    """
    Analyze dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer for length calculations
        
    Returns:
        Dictionary with dataset statistics
    """
    total_samples = len(dataset)
    
    # Check if dataset has been tokenized (has input_ids)
    if 'input_ids' in dataset.column_names:
        # Dataset has been tokenized
        token_lengths = []
        for sample in dataset:
            if 'input_ids' in sample:
                token_lengths.append(len(sample['input_ids']))
        
        return {
            'total_samples': total_samples,
            'avg_token_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'min_token_length': min(token_lengths) if token_lengths else 0,
        }
    else:
        # Dataset has raw text
        text_lengths = []
        token_lengths = []
        
        for sample in dataset:
            if 'text' in sample:
                text = sample['text']
                text_lengths.append(len(text))
                
                tokens = tokenizer.encode(text)
                token_lengths.append(len(tokens))
        
        return {
            'total_samples': total_samples,
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'avg_token_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'min_token_length': min(token_lengths) if token_lengths else 0,
        } 
