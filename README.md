# Local 7B LLM Fine-tuning and GGUF Conversion

A complete pipeline for fine-tuning 7B language models using QLoRA and converting them to GGUF format.

## Features

- 🔧 **QLoRA Fine-tuning**: Efficient fine-tuning with 4-bit quantization
- 🚀 **GPU Acceleration**: Optimized for consumer-grade GPUs
- 🔄 **GGUF Conversion**: Convert models to GGUF format for llama.cpp inference
- 📊 **Training Monitoring**: Integration with Weights & Biases and TensorBoard
- 🎯 **Easy Setup**: Automated environment setup and dependency management

## Prerequisites

### System Requirements

- Python 3.9+
- CUDA Toolkit (for GPU acceleration)

### Hardware Requirements

- **Minimum**: 16GB RAM, CPU-only training
- **Recommended**: 24GB+ RAM, NVIDIA GPU with 8GB+ VRAM

## Quick Start

1. **Clone and Setup**

   ```bash
   git clone <repository-url>
   cd lm-finetune
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**
   Create a JSONL file with your training data:

   ```json
   {"text": "<s>[INST] What is machine learning? [/INST] Machine learning is..."}
   {"text": "<s>[INST] Explain neural networks [/INST] Neural networks are..."}
   ```

3. **Run Fine-tuning**

   ```bash
   python scripts/finetune.py --model_name_or_path meta-llama/Llama-2-7b-hf --data_path data/training.jsonl --output_dir models/finetuned
   ```

4. **Convert to GGUF**

   ```bash
   python scripts/convert_to_gguf.py --model_path models/finetuned --output_path models/final.gguf
   ```

5. **Test Inference**
   ```bash
   python scripts/inference.py --model_path models/final.gguf --prompt "What is machine learning?"
   ```

## Project Structure

```
lm-finetune/
├── scripts/
│   ├── finetune.py          # QLoRA fine-tuning script
│   ├── convert_to_gguf.py   # GGUF conversion script
│   ├── inference.py         # Inference testing script
│   └── setup_env.py         # Environment setup
├── utils/
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── model_utils.py       # Model loading utilities
│   └── training_utils.py    # Training helpers
├── configs/
│   └── training_config.py   # Training configuration
├── data/                    # Training data directory
├── models/                  # Output models directory
├── requirements.txt
└── README.md
```

## Configuration

### Training Parameters

Key parameters can be configured in `configs/training_config.py`:

- **Model**: Base model to fine-tune
- **LoRA**: Rank, alpha, dropout settings
- **Training**: Learning rate, batch size, epochs
- **Quantization**: Bits, group size settings

### Data Format

Training data should be in JSONL format with a "text" field containing complete prompts including model-specific tokens:

```json
{ "text": "<s>[INST] User question here [/INST] Assistant response here</s>" }
```

## Advanced Usage

### Custom Training Configuration

```python
from configs.training_config import TrainingConfig

config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4
)
```

### Multi-GPU Training

```bash
accelerate launch --multi_gpu scripts/finetune.py --config_path configs/multi_gpu.yaml
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading Errors**: Ensure you have access to the base model
3. **GGUF Conversion Failures**: Check model format and llama.cpp installation

### Performance Tips

- Use gradient checkpointing for memory efficiency
- Enable mixed precision training
- Monitor GPU memory usage with `nvidia-smi`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
