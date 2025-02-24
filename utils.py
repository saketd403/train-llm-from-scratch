# Util functions useful for various modules.
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from enum import Enum
from huggingface_hub import login

from logger import setup_logger

# Create a logger specific to this module
logger = setup_logger('utils')

datasize_mapping = {
                    "fp32":4,
                    "fp16":2,
                    "bf16":2
                    }

datatype_mapping = {
                    "fp32":torch.float32,
                    "fp16":torch.float16,
                    "bf16":torch.bfloat16
                    }

model_params_mapping = {
                "GPT2":["124M","355M","774M","1.5B"],
                "llama2":["7B"],
                "llama3":["8B"],
                "llama3_1":["8B"],
                "llama3_2":["1B"]
                }

def set_seed():

    """Set random seeds."""

    RANDOM_SEED=123

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(seed=0)
    torch.random.manual_seed(seed=RANDOM_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def text_to_token_ids(text, tokenizer, cfg):
    encoded = tokenizer.encode(text, allowed_special={cfg["eos_text"]})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist()) #convert to list from tensor and then decode

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data

def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_num_params(model):

    total_params = sum(p.numel() for p in model.parameters())
    
    return total_params

def get_total_size(num_params,data_type):

    assert data_type in datasize_mapping, "This datatype is currently not supprted."

    datatype_size = int(datasize_mapping[data_type])
    logger.info(f"Since the datatype is {data_type}, each parameter is going to consume {datatype_size} bytes")
    
    logger.info(
        "Assuming that we are using Adam optimizer, this model will require 1N (N is number of parameters)"
        "for parameters, 1N for gradients and 2N for first and second moment estimates of Adam."
        "So total 4N of GPU memory"
        )
    
    total_size_bytes = 4 * num_params * datatype_size

    # Convert to gigabytes
    total_size_mb = total_size_bytes / (1024 * 1024 * 1024)

    logger.info(
        f"Estimated size of the model: {total_size_mb:.2f} GB.\n\n" 
          "During the forward pass, activations are stored for backpropagation."
          "These can significantly increase memory usage. This memory is not included in above calculations."
          "Please use activation checkpointing to decrease activation memory."
        )
    

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    logger.info(f"Estimated size of the model: {total_memory_gb:.2f} GB.\n\n")

    

def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.info("This notebook is intended for CUDA GPUs but CUDA is not available.")

def print_memory_usage():
    max_gpu_memory = float(torch.cuda.max_memory_allocated()) / (1024 ** 3)  # Convert bytes to GB
    logger.info(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(output_dir / "losses.pdf")

def login_hf():

    with open("config_hf.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)