

# LLM Training

This repository provides a full implementation of GPT2 and Llama2/Llama3 models from scratch in Python. It allows for flexible training configurations, including single and multi-GPU distributed training, activation checkpointing, FSDP (Fully Sharded Data Parallel), mixed precision training, and more. The repo also supports pretraining on raw text and finetuning on instruction datasets. Additionally, LoRA (Low-Rank Adaptation) integration is included for memory-efficient model finetuning.

## Features

- **GPT2 & Llama2/Llama3 Models**: Implemented from scratch.
- **Pretrained Weights**: Load pretrained weights from Hugging Face.
- **Training Configurations**:
  - Single GPU & Multi-GPU (Distributed Training).
  - Activation Checkpointing.
  - FSDP (Fully Sharded Data Parallel).
  - Mixed Precision Training (using `torch.cuda.amp`).
  - LoRA (Low-Rank Adaptation) support for efficient finetuning.
- **Training Modes**:
  - Pretraining on raw text.
  - Finetuning on instruction datasets.

  
## Requirements

- Python 3.8+
- PyTorch 1.10+ (with GPU support)
- Hugging Face Transformers
- CUDA (for GPU support)
- Other dependencies listed in `requirements.txt`

To install dependencies, run:
```bash
pip install -r requirements.txt
```

To download weights from hugging face, you need to first login into hugging face and get the access token which will go towards HF_ACCESS_TOKEN under `config_hf.json` file.

The repo code has been tested on aws instances - g4dn.xlarge for single-GPU run and g4dn.12xlarge for multi-GPU run.

## Quick Start

To run the training, the entry point is the `main.py` file. Here's how to get started:

### Download Gutenberg dataset for pretraining on raw text

Run setup.sh under Datasets/Gutenberg folder

```bash
 bash setup.sh
```

### Download Alpaca dataset for finetuning on instruction dataset

Run setup.sh under Datasets/Alpaca folder

```bash
 bash setup.sh
```

### Example 1: Pretrain GPT2 on Raw Text

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --load_weights
```

### Example 2: Distributed Data Parallel training with Low rank adapatation (LoRA)

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --load_weights --model GPT2 --run_type multi_gpu --use_lora
```

### Example 3: Enable Activation Checkpointing

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --load_weights --model GPT2 --num_params 774M --run_type multi_gpu --use_lora --use_actv_ckpt 
```

### Example 4: Fully-Sharded Data Parallelism

```bash
python main.py --data_dir ./Datasets/Gutenberg/data_dir --load_weights --model GPT2 --num_params 774M --run_type multi_gpu --use_actv_ckpt --use_lora --use_fsdp
```

### Example 5: Finetune Llama3.2 on Instruction Dataset

```bash
python main.py --dataset alpaca --data_dir ./Datasets/Alpaca/data --load_weights --model llama3_2 --num_params 1B --finetune --run_type multi_gpu --use_actv_ckpt --use_lora --lr 1e-5 --data_type bf16  
```

---
## Model configurations supported

- GPT2 -  124M, 355M, 774M, 1.5B
- Llama2 - 7B
- Llama3 - 8B
- Llama3.1 - 8B
- Llama3.2 - 1B

---
## Arguments

Here are some common arguments you can use to configure your training:

- `--model` : Type of model to train (`GPT2`, `llama2`, `llama3`).
- `--num_params` : Choose the model size that you want to train.
- `--data_type` : Datatype for the model.
- `--load_weights` : Use this argument if you want to load weights from hugging face.
- `--n_epochs` : Number of epochs for training.
- `--batch_size` : Batch size for training.
- `--run_type` : 'single_gpu' is default. 'multi_gpu' to enable distributed data parallel training. 
- `--lr` : Learning rate to use after finishing warmup steps. By default we are using cosine annealing as lr schedule.
- `--warmup_steps` : Number of warmup steps for training.
- `--dataset` : Dataset to be used for training.
- `--finetune` : To activate instruction finetuning. Default is to pretrain on raw text.
- `--use_zero_opt` : Activate Zero optimzer.
- `--use_lora` : Use LoRA for model finetuning.
- `--lora_rank` : Rank value for LoRA.
- `--lora_alpha` : Alpha value for LoRA.
- `--use_fsdp` : Enable FSDP (Fully Sharded Data Parallel) for distributed training.
- `--mixed_precision` : Use mixed precision training for better performance.(Only supported with mixed precision at this time).
- `--use_actv_ckpt` : Enable activation checkpointing to reduce memory usage.

---

## Advanced Usage

### Mixed Precision Training

This repository supports mixed precision training, which speeds up training and reduces memory consumption. You can enable it using the `--mixed_precision` flag. Only supported with FSDP at this time.

### FSDP (Fully Sharded Data Parallel)

For multi-GPU training, FSDP is supported for better memory efficiency and performance. Simply use the `--use_fsdp` argument to enable it.

### LoRA (Low-Rank Adaptation)

You can enable LoRA for efficient finetuning by using the `--use_lora` argument. LoRA reduces the computational cost while maintaining good performance on downstream tasks.

### Activation Checkpointing

Activation checkpointing is implemented to allow training on large models by saving memory. You can enable it with the `--use_actv_ckpt` flag.

## Notes

- **GPU Setup**: Ensure that you have CUDA installed and properly configured.
- **Pretrained Weights**: You can easily load pretrained weights from Hugging Face for your models. Simply pass the `--load_weights` flag to `main.py` script.
- **Scalability**: This code is designed to scale across multiple GPUs and large datasets.

---

## Contributing

Feel free to fork this repository and open issues for any questions.

---

## License

This code is licensed under the Apache License. See the `LICENSE` file for more information.

