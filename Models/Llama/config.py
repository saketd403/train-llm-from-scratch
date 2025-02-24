import torch
from Models.Llama.common_components import rescale_theta

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
}

LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # Larger vocabulary size
    "context_length": 8192,  # Larger context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 14_336,    # Larger size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,        # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta" was increased to 500_000
    "rope_freq": None,       # Additional configuration for adjusting the RoPE frequencies
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "eos_id":128001,
    "eos_text":"<|end_of_text|>"
}

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Larger supported context length
    "emb_dim": 4096,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 32,             # Number of layers
    "hidden_dim": 14_336,       # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
    "rope_freq": {              # RoPE frequency scaling
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "eos_id":128001,
    "eos_text":"<|end_of_text|>"
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,            # Half the embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Half the number of layers
    "hidden_dim": 8192,         # Almost half the size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,         # Adjustment of the rescaling factor
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "eos_id":128001,
    "eos_text":"<|end_of_text|>"
}

available_configs_llama2 = {
    "7B":LLAMA2_CONFIG_7B
}

available_configs_llama3 = {
    "8B":LLAMA3_CONFIG_8B
}

available_configs_llama3_1 = {
    "8B":LLAMA31_CONFIG_8B
}

available_configs_llama3_2 = {
    "1B":LLAMA32_CONFIG_1B
}

def get_config_llama(num_params,model_name):

    num_params = str(num_params)

    available_configs = globals().get(f"available_configs_{model_name}", None)

    assert num_params in available_configs, f"A {model_name} model for given number of parameters does not exists."

    config = available_configs[num_params]

    old_context_length = config["context_length"]
    if(old_context_length!=1024):

        config["context_length"] = 1024 #8192

        config["rope_base"] = rescale_theta(
            config["rope_base"],
            old_context_length,
            config["context_length"]
        )

        print("New RoPE theta:", config["rope_base"])

    return config


def get_config_llama2(num_params):

    num_params = str(num_params)

    assert num_params in available_configs_llama2, "A llama2 model for given number of parameters does not exists."

    return available_configs_llama2[num_params]


def get_config_llama3():
    
    return LLAMA3_CONFIG_8B

def get_config_llam31():

    old_context_length = LLAMA31_CONFIG_8B["context_length"]
    LLAMA31_CONFIG_8B["context_length"] = 1024 #8192

    LLAMA31_CONFIG_8B["rope_base"] = rescale_theta(
        LLAMA31_CONFIG_8B["rope_base"],
        old_context_length,
        LLAMA31_CONFIG_8B["context_length"]
    )

    print("New RoPE theta:", LLAMA31_CONFIG_8B["rope_base"])
    
    return LLAMA31_CONFIG_8B 

def get_config_llam32():

    old_context_length = LLAMA32_CONFIG_1B["context_length"]
    LLAMA32_CONFIG_1B["context_length"] = 1024 #8192

    LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
        LLAMA32_CONFIG_1B["rope_base"],
        old_context_length,
        LLAMA32_CONFIG_1B["context_length"]
    )

    print("New RoPE theta:", LLAMA32_CONFIG_1B["rope_base"])
    
    return LLAMA32_CONFIG_1B