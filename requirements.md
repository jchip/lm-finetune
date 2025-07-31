## Local 7B LLM Fine-tuning and GGUF Conversion

### 1. Project Goal

Establish a local development workflow to fine-tune a 7B LLM using QLoRA and convert the resulting model to GGUF format for efficient inference.

### 2. Requirements

#### Environment

- Python 3.9+
- CUDA Toolkit (for GPU acceleration)

#### Core Libraries and Tools

- `transformers` for model loading and management
- `peft` for QLoRA implementation
- `bitsandbytes` for 4-bit quantization
- `trl` for SFTTrainer
- `datasets` for training data loading
- `accelerate` for training optimization
- `llama.cpp` for GGUF conversion and inference

#### Model Format

- Input: Hugging Face PyTorch/Safetensors (7B parameters)
- Output: GGUF (quantized)
- Fine-tuning: QLoRA

#### Data Format

- JSONL with a "text" field containing full prompts with model-specific tokens (e.g., `<s>`, `</s>`, `[INST]`, `[/INST]`)

### 3. Functional Capabilities

- Environment setup automation or documentation
- Download and load base model
- Accept user-provided JSONL training data
- Execute QLoRA fine-tuning using GPU when available
- Save LoRA adapter
- Merge adapter with base model
- Convert merged model to GGUF
- Support configurable quantization level (e.g., Q4_K_M)
- Enable local inference with final model

### 4. Constraints and Considerations

- Optimize for consumer-grade GPUs using QLoRA
- Minimal manual steps after setup
- Clear fallback behavior for CPU-only environments
- Efficient memory and compute usage

### 5. Deliverables

- Fine-tuning and conversion pipeline with all scripts
- LoRA adapter and merged model outputs
- Final GGUF model with quantization
- Sample inference demonstrating successful fine-tuning
