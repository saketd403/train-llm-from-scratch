import tiktoken
import torch
import torch.nn as nn

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
    enable_wrap,
    wrap,
)


import utils
from Models.GPT2.config import get_config_gpt2
from Models.Llama.config import get_config_llama

from Models.GPT2.GPT2 import GPTModel
from Models.Llama.Llama2 import Llama2Model
from Models.Llama.Llama3 import Llama3Model

from Models.Llama.Llama2 import Llama2Tokenizer
from Models.Llama.Llama3 import Llama3Tokenizer

from Models.Llama.common_components import rescale_theta

from huggingface_hub import hf_hub_download
from huggingface_hub import login

from logger import setup_logger

# Create a logger specific to this module
logger = setup_logger('build_components')


def build_config(args):

    """Build and returns config dictionary."""

    
    if(args.model=="GPT2"):

        config = get_config_gpt2(args.num_params)

    elif(args.model.startswith("llama")):

        config = get_config_llama(args.num_params,args.model)

    config.update({"dtype":utils.datatype_mapping[args.data_type]})

    if(args.load_weights and args.model=="GPT2"):

        config.update({"qkv_bias":True})

    if args.debug:

        config.update({
            "context_length": 10,    # Context length
            "emb_dim": 32,           # Embedding dimension
            "hidden_dim": 10,        # Hidden dimension of feedforward layer  
            "n_heads": 16,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "qkv_bias": False        # Query-key-value bias
        })


    return config

def load_model_weights(args,config,model):

    utils.login_hf()
    if(args.model=="GPT2"):

        from Models.GPT2.load_weights import load_hf_weights

        load_hf_weights(model,args.num_params,config)

    if(args.model=="llama2"):

        from Models.Llama.load_weights_llama2 import load_hf_weights

        load_hf_weights(model, config)


    if(args.model.startswith("llama3")):

        from Models.Llama.load_weights_llama3 import load_hf_weights

        load_hf_weights(model, args.model, config)

def activate_lora(args,cfg,model):

    from lora import replace_linear_with_lora

    logger.info("Using LoRA...")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters before: {total_params:,}")

    logger.info("Turning the weights off ...")

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters after: {total_params:,}")

    replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dtype=cfg["dtype"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable LoRA parameters: {total_params:,}")


def multigpu_setup(args,rank,model):

    if(torch.cuda.is_available()): #FSDP is only possible with GPU.

        fsdp_kwargs = {
            "sharding_strategy" : ShardingStrategy.FULL_SHARD,
            "cpu_offload" : None,
            "backward_prefetch" : BackwardPrefetch.BACKWARD_PRE,
            "mixed_precision" : None,
            "sync_module_states" : False,
            "device_id":torch.cuda.current_device(),
            "use_orig_params":False
        }

    if(args.use_fsdp):

        from datautils.mixed_precision import fpSixteen,bfSixteen
        mixed_precision_policy=None
        if(args.mixed_precision):
            if(args.mixed_precision=="fp16"):                   
                mixed_precision_policy = fpSixteen
            elif(args.mixed_precision=="bf16"):                
                mixed_precision_policy = bfSixteen  
                                
            fsdp_kwargs.update({"mixed_precision":mixed_precision_policy})

        if(args.use_lora):
            fsdp_kwargs.update({"use_orig_params":True})
            #     #ignored_modules = [module for name, module in model.named_modules() if ".lora" not in name]
            #     ignored_modules=[]
            #     for name, module in model.named_modules():
            #         if(".lora" not in name):
            #             ignored_modules.append(module)
            #     fsdp_kwargs.update({"ignored_modules":ignored_modules})

        from Models.GPT2.GPT2 import TransformerBlock
        # my_auto_wrap_policy = functools.partial(
        #                             size_based_auto_wrap_policy, min_num_params=100
        #                         )
        # trf_auto_wrap_policy = functools.partial(
        #                             transformer_auto_wrap_policy,
        #                             transformer_layer_cls={
        #                                 TransformerBlock,
        #                             },
        #                         )
        #my_auto_wrap_policy=None
        my_auto_wrap_policy = ModuleWrapPolicy(module_classes=[nn.Embedding,TransformerBlock])
        
        model = FSDP(model,
                        auto_wrap_policy=my_auto_wrap_policy,**fsdp_kwargs)
    else:
        model = DDP(model, device_ids=[rank])

    if(rank==0):
        logger.info("Model wrapped with DDP/FSDP ....")
        utils.print_memory_usage()

    return model

def build_model(config,rank,device,args):

    """

    Args:
        config: config dictionary object.
        rank: rank of process.
        device: CUDA device being used.
        args: command line input arguments.

    Returns:
        model: Instance of model class.

    """

    utils.start_memory_tracking()

    if(args.model=="GPT2"):
        model = GPTModel(config,args.use_actv_ckpt)
    elif(args.model=="llama2"):
        model = Llama2Model(config,args.use_actv_ckpt)
    elif(args.model.startswith("llama3")):
        model = Llama3Model(config,args.use_actv_ckpt)
    else:
        raise Exception("Invalid model Exception : This code does not support this model configuration.")

    total_params = utils.get_num_params(model)

    if(rank==0):
        logger.info(f"Total number of parameters in the model : {total_params:,}")

        utils.model_memory_size(model,config["dtype"])

    if(args.load_weights):

        if(rank!=0 and args.run_type=="multi_gpu"):
            torch.distributed.barrier()

        load_model_weights(args,config,model)

        if(rank==0 and args.run_type=="multi_gpu"):
            torch.distributed.barrier()

        if(rank==0):
            logger.info("Weights loaded ....")
            utils.print_memory_usage()

    if(args.use_lora):

        activate_lora(args,config,model)

    # FSDP should keep model on cpu and then using FSDP wrapper, 
    # put the weights on multiple GPUs. This will help save memory
    # especially when model is too large to be loaded in single GPU memory.
    if(not args.use_fsdp): 
        model.to(device)

    if(rank==0):
        logger.info("Model loaded into cuda device ....")
        utils.print_memory_usage()

    if args.use_zero_opt:

        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    if(args.run_type=="multi_gpu"):
        model = multigpu_setup(args,rank,model)

    if(rank==0):
        logger.info(f"Following is the model for rank {rank}: ")
        logger.info(model)

    return model


def build_optimizer(args,model):

    if args.use_zero_opt:

        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=0.1
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    return optimizer


def build_tokenizer(rank,args):

    if(rank!=0 and args.run_type=="multi_gpu"):
        torch.distributed.barrier()

    if(args.model=="GPT2"):

        tokenizer = tiktoken.get_encoding("gpt2")

    elif(args.model=="llama2"):

        utils.login_hf()

        tokenizer_file = hf_hub_download(
            repo_id="meta-llama/Llama-2-7b",
            filename="tokenizer.model",
            local_dir="Llama-2-7b"
        )

        tokenizer = Llama2Tokenizer(tokenizer_file)   

    elif(args.model.startswith("llama3")):

        utils.login_hf()

        if(args.model=="llama3"):

            tokenizer_file = hf_hub_download(
                                repo_id="meta-llama/Meta-Llama-3-8B",
                                filename="original/tokenizer.model",
                                local_dir="Llama-3-8B"
                            )

        elif(args.model=="llama3_1"):

            tokenizer_file = hf_hub_download(
                                    repo_id="meta-llama/Llama-3.1-8B",
                                    filename="original/tokenizer.model",
                                    local_dir="Llama-3.1-8B"
                                )

        elif(args.model=="llama3_2"):

            tokenizer_file = hf_hub_download(
                                    repo_id="meta-llama/Llama-3.2-1B",
                                    filename="original/tokenizer.model",
                                    local_dir="Llama-3.2-1B"
                                )

        tokenizer = Llama3Tokenizer(tokenizer_file)   

    else:

        raise Exception("Tokenizer Not Found Exception: No tokenizer found for the given model.")  
    
    if(rank==0 and args.run_type=="multi_gpu"):
        torch.distributed.barrier()  

    return tokenizer

def build_components(rank: int, device: torch.device, args):

    """

    Build and returns training objects such as config, model, optimizer and tokenizer.

    Args:

        rank: rank of process.
        device: CUDA device.
        args: command line arguments.

    Returns:
        config: config file.
        model: LLM model object.
        optimizer: optimizer object.
        tokenizer: tokenizer object

    """

    try:

        config = build_config(args)

        model = build_model(config,rank,device,args)

        optimizer = build_optimizer(args,model)

        tokenizer = build_tokenizer(rank,args)
  
        return config, model, optimizer, tokenizer

    except Exception as e:

        logger.exception(f"\nException while building training components :\n{e}")


    