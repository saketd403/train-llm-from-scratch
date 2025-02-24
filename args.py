import argparse
import warnings
import os
import torch
from utils import model_params_mapping


def perform_checks(args):

    """Performs validation of input arguments."""

    if(not args.warnings):
        warnings.filterwarnings("ignore")

    if(not os.path.exists(args.data_dir)):
        raise Exception("The data dir path specified does not exists.")
    
    if(args.num_params not in model_params_mapping[args.model]):
        raise Exception(f"You are asking to load {args.num_params} configuration for {args.model} model. This configuration is currently not supported.")
    
    if(args.run_type=="single_gpu" and args.use_fsdp):
        raise Exception("FSDP not supported on single GPU non-distributed training.")
    
    if(args.use_zero_opt and args.use_fsdp):
        raise Exception("Zero Optimizer is not supported with FSDP.")
    
    if(args.use_fsdp and not torch.cuda.is_available()):
        raise Exception("FSDP can only be activated when CUDA device is available.")
    
    if(not args.use_fsdp and args.mixed_precision):
        raise Exception("Mixed precision is only supported with FSDP at this time.")
    

def get_args():

    """Get command line arguments."""

    parser = argparse.ArgumentParser(description='Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='/home/ec2-user/train-llm-from-scratch/Datasets/Gutenberg/data_dir_small',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=10,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Warmup steps for training.')
    parser.add_argument('--initial_lr', type=float, default=1e-05,
                        help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate.')
    parser.add_argument('--debug', action="store_true",
                        help='Uses a very small model for debugging purposes')
    parser.add_argument('--model', type=str, default="GPT2",
                        choices=["GPT2","llama2","llama3","llama3_1","llama3_2"],
                        help='The model to use.')
    parser.add_argument('--num_params', type=str, default="124M",
                        help='Model size.')
    parser.add_argument('--load_weights',  action="store_true",
                        help='Do we need to load pretrained weights?')
    parser.add_argument('--data_type',type=str,default="fp32",
                        choices=["fp32","fp16","bf16"],
                        help="Datatype to use.bf16 is better choice for training compared to fp16 due to stability reasons.")
    parser.add_argument('--run_type',type=str,default="single_gpu",
                        choices=['single_gpu', 'multi_gpu'],
                        help="How to optmize the run? Should be multi_gpu for FSDP.")
    parser.add_argument('--use_zero_opt',action="store_true",
                        help="Don't use with FSDP. Use Zero Redeundancy Optimizer")
    parser.add_argument('--use_actv_ckpt',action="store_true",
                        help="Activation checkpointing")
    parser.add_argument('--use_fsdp',action="store_true",
                        help="Fully Sharded Data Parallelism. Requires multi-gpu run")
    parser.add_argument('--mixed_precision',type=str,
                        choices=['fp16', 'bf16'],
                        help="Mixed precision to be used for FSDP.")
    parser.add_argument('--finetune',action="store_true",
                        help="Enable finetuning.") 
    parser.add_argument('--dataset',type=str,default="gutenberg",
                        choices=['gutenberg', 'alpaca'],
                        help="Dataset to be used.")
    parser.add_argument('--use_lora',action="store_true",
                        help="Enable LoRA training.")
    parser.add_argument('--lora_rank',type=int,default=64,
                        help="Rank value for LoRA.")
    parser.add_argument('--lora_alpha',type=int,default=32,
                        help="Alpha value for LoRA.")
    parser.add_argument('--warnings',action="store_true",
                        help="Turn the warnings on.")
    
    args = parser.parse_args()

    perform_checks(args)

    return args