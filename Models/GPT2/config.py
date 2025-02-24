

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,        # Query-key-value bias
    "eos_id":50256,
    "eos_text":"<|endoftext|>"
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,          # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,        # Query-key-value bias
    "eos_id":50256,
    "eos_text":"<|endoftext|>"
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1280,          # Embedding dimension
    "n_heads": 20,           # Number of attention heads
    "n_layers": 36,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,        # Query-key-value bias
    "eos_id":50256,
    "eos_text":"<|endoftext|>"
}

GPT_CONFIG_1_5B = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1600,          # Embedding dimension
    "n_heads": 25,           # Number of attention heads
    "n_layers": 48,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,        # Query-key-value bias
    "eos_id":50256,
    "eos_text":"<|endoftext|>"
}

available_configs = {
    "124M":GPT_CONFIG_124M,
    "355M":GPT_CONFIG_355M,
    "774M":GPT_CONFIG_774M,
    "1.5B":GPT_CONFIG_1_5B
}



def get_config_gpt2(num_params):

    num_params = str(num_params)

    assert num_params in available_configs, "A GPT2 model for given number of parameters does not exists."

    return available_configs[num_params]

