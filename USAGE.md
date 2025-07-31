# LLM Fine-tuning Usage Guide

This guide provides detailed instructions for using the LLM fine-tuning pipeline.

## Quick Start

### 1. Environment Setup

First, set up your environment:

```bash
# Check environment
python scripts/setup_env.py --check_only

# Install dependencies
python scripts/setup_env.py --install_deps

# Create sample data
python scripts/setup_env.py --create_sample_data
```

### 2. Run Complete Example

Test the entire pipeline with a small model:

```bash
python scripts/run_example.py --model_name microsoft/DialoGPT-medium --epochs 1
```

### 3. Fine-tune Your Own Model

```bash
python scripts/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/your_training_data.jsonl \
    --output_dir models/my_finetuned_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

## Detailed Usage

### Data Format

Your training data should be in JSONL format with a "text" field:

```json
{"text": "<s>[INST] What is machine learning? [/INST] Machine learning is..."}
{"text": "<s>[INST] Explain neural networks [/INST] Neural networks are..."}
```

### Fine-tuning Script Options

#### Model Configuration

- `--model_name_or_path`: Base model to fine-tune
- `--trust_remote_code`: Trust remote code when loading model
- `--config_preset`: Use preset configurations (default, memory_efficient, high_quality)

#### Training Parameters

- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--max_seq_length`: Maximum sequence length

#### LoRA Parameters

- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)

#### Monitoring

- `--report_to`: Reporting backends (wandb, tensorboard)
- `--logging_steps`: Logging frequency
- `--save_steps`: Save frequency
- `--eval_steps`: Evaluation frequency

### GGUF Conversion

Convert your fine-tuned model to GGUF format:

```bash
python scripts/convert_to_gguf.py \
    --model_path models/finetuned/merged \
    --output_path models/final_model.gguf \
    --quantization q4_k_m \
    --context_length 2048
```

### Inference

Test your model:

```bash
# HuggingFace model
python scripts/inference.py \
    --model_path models/finetuned/merged \
    --model_type huggingface \
    --prompt "What is machine learning?" \
    --max_tokens 100

# GGUF model
python scripts/inference.py \
    --model_path models/final_model.gguf \
    --model_type gguf \
    --prompt "What is machine learning?" \
    --max_tokens 100
```

## Configuration Presets

### Default Configuration

- LoRA rank: 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Batch size: 4
- Sequence length: 2048

### Memory Efficient Configuration

- LoRA rank: 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Batch size: 1
- Sequence length: 1024
- Gradient accumulation: 8

### High Quality Configuration

- LoRA rank: 32
- LoRA alpha: 64
- Learning rate: 1e-4
- Batch size: 4
- Sequence length: 4096
- Epochs: 5

## Hardware Requirements

### Minimum Requirements

- **RAM**: 16GB
- **Storage**: 50GB free space
- **GPU**: CPU-only training (slow)

### Recommended Requirements

- **RAM**: 24GB+
- **Storage**: 100GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8+

### Memory Usage Estimates

- **7B model**: ~14GB VRAM (4-bit quantized)
- **Training**: +2-4GB VRAM for gradients
- **Total**: 16-18GB VRAM recommended

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

- Reduce batch size: `--per_device_train_batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 8`
- Use memory efficient config: `--config_preset memory_efficient`
- Reduce sequence length: `--max_seq_length 1024`

#### 2. Model Loading Errors

**Symptoms**: `OSError: We couldn't connect to 'https://huggingface.co'`

**Solutions**:

- Check internet connection
- Use `--trust_remote_code` flag
- Download model locally first
- Check HuggingFace access permissions

#### 3. Training Data Issues

**Symptoms**: `ValueError: Invalid training data`

**Solutions**:

- Check JSONL format
- Ensure "text" field exists
- Validate JSON syntax
- Check file encoding (UTF-8)

#### 4. GGUF Conversion Failures

**Symptoms**: `Conversion failed with error code 1`

**Solutions**:

- Install llama-cpp-python: `pip install llama-cpp-python`
- Check model path exists
- Verify model format (HuggingFace)
- Try different quantization level

### Performance Optimization

#### For Limited GPU Memory

```bash
python scripts/finetune.py \
    --config_preset memory_efficient \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 1024
```

#### For Better Quality

```bash
python scripts/finetune.py \
    --config_preset high_quality \
    --num_train_epochs 5 \
    --learning_rate 1e-4
```

#### For Faster Training

```bash
python scripts/finetune.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 50 \
    --save_steps 1000
```

## Advanced Usage

### Custom Training Configuration

Create a custom configuration:

```python
from configs.training_config import TrainingConfig

config = TrainingConfig(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    lora_r=32,
    lora_alpha=64,
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=4096
)
```

### Multi-GPU Training

Use multiple GPUs with accelerate:

```bash
accelerate launch --multi_gpu scripts/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/training.jsonl \
    --output_dir models/finetuned
```

### Custom Data Processing

Process your own data:

```python
from utils.data_processing import create_sample_data

# Create custom training data
create_sample_data("data/custom_training.jsonl", num_samples=1000)
```

### Model Analysis

Analyze your model:

```python
from utils.model_utils import get_model_info

model_info = get_model_info(model)
print(f"Total parameters: {model_info['total_parameters']:,}")
print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
```

## Monitoring and Logging

### Weights & Biases Integration

Enable W&B logging:

```bash
python scripts/finetune.py \
    --report_to wandb \
    --wandb_project my-finetune \
    --wandb_run_name experiment-1
```

### TensorBoard Integration

Enable TensorBoard logging:

```bash
python scripts/finetune.py \
    --report_to tensorboard \
    --logging_dir logs
```

### Custom Logging

Monitor training progress:

```python
from utils.training_utils import TrainingMonitor

monitor = TrainingMonitor(log_interval=10)
monitor.start_training()

# During training
monitor.log_step(step=100, loss=0.5, learning_rate=2e-4)

# End training
monitor.end_training()
```

## Deployment

### HuggingFace Model Hub

Upload your model:

```bash
# Save model
python scripts/finetune.py --output_dir models/my_model

# Upload to Hub
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('models/my_model/merged')
tokenizer = AutoTokenizer.from_pretrained('models/my_model/merged')
model.push_to_hub('your-username/your-model-name')
tokenizer.push_to_hub('your-username/your-model-name')
"
```

### GGUF Model Deployment

Deploy GGUF model with llama.cpp:

```bash
# Convert to GGUF
python scripts/convert_to_gguf.py \
    --model_path models/finetuned/merged \
    --output_path models/deployment.gguf

# Use with llama.cpp
./llama.cpp/main -m models/deployment.gguf -n 100 -p "What is machine learning?"
```

## Best Practices

### Data Quality

- Use high-quality, diverse training data
- Ensure proper formatting with model-specific tokens
- Balance dataset size and quality
- Validate data before training

### Training Strategy

- Start with smaller models for experimentation
- Use validation data to monitor overfitting
- Experiment with different LoRA configurations
- Monitor training metrics closely

### Model Selection

- Choose base model appropriate for your task
- Consider model size vs. performance trade-offs
- Test multiple models before full training

### Resource Management

- Monitor GPU memory usage
- Use appropriate batch sizes
- Enable gradient checkpointing
- Use mixed precision training

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify your environment setup
4. Test with the example script first
5. Check hardware requirements

## Examples

### Complete Pipeline Example

```bash
# 1. Setup environment
python scripts/setup_env.py --install_deps --create_sample_data

# 2. Run complete example
python scripts/run_example.py --model_name microsoft/DialoGPT-medium

# 3. Fine-tune with your data
python scripts/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/your_data.jsonl \
    --output_dir models/my_model

# 4. Convert to GGUF
python scripts/convert_to_gguf.py \
    --model_path models/my_model/merged \
    --output_path models/my_model.gguf

# 5. Test inference
python scripts/inference.py \
    --model_path models/my_model.gguf \
    --model_type gguf \
    --prompt "Your test prompt"
```
