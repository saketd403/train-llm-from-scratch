
import torch
from pathlib import Path
import json
import os
from functools import partial

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from train import Trainer
import utils
from datautils.dataloader import DataloaderPT
from datautils.dataloader_instruction_finetune import DataloaderIF
from build_components import build_components
from args import get_args

from logger import setup_logger

# Create a logger specific to this module
logger = setup_logger('main')


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    destroy_process_group()


def main(rank: int, args):

    """
    Main function.

    Args:
        rank: rank of the process.
        args: Command line input arguments.
    """

    if(args.run_type=="multi_gpu"):
        ddp_setup(rank, args.world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.set_seed()

    config,model,optimizer,tokenizer = build_components(rank,device,args)

    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt",".json"))]
    total_files = len(all_files)

    if total_files == 0:
        raise Exception("No training text files found. Make sure you "
              "selected the correct input directory")
        

    if(rank==0):
        logger.info(f"Total data files: {total_files}")

    dataloader_kwargs = {"tokenizer":tokenizer,
                         "batch_size":args.batch_size,
                         "max_length":config["context_length"],
                         "dataset_name":args.dataset,
                         "run_type":args.run_type,
                         "train_ratio":0.9}

    collate_func = None
    if(args.finetune):
        from datautils.dataloader_instruction_finetune import custom_collate_fn
        customized_collate_fn = partial(
                                custom_collate_fn,
                                allowed_max_length=config["context_length"]
                                )
        loaderObj = DataloaderIF(                   
                    collate_func=customized_collate_fn,
                    **dataloader_kwargs)
    else:
        loaderObj = DataloaderPT(
                    stride=config["context_length"],
                    eos_text=config["eos_text"],
                    collate_func=collate_func,
                    **dataloader_kwargs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        config=config,
        data_files=all_files,
        loaderObj=loaderObj,
        device=device,
        model=model,
        optimizer=optimizer,
        save_dir=output_dir,
        rank=rank,
        warmup_steps=args.warmup_steps,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        eval_freq=args.eval_freq,
        save_ckpt_freq=args.save_ckpt_freq,
        print_sample_iter=args.print_sample_iter,
        eval_iter=5
    )

    
    #Test a single sentence 
    trainer.generate_and_print_sample("Every effort moves you",temperature=1.0,top_k=5,memory_check=True)

    if(args.finetune):
        train_losses, val_losses, tokens_seen, track_lrs = trainer.finetune_model(
            n_epochs=args.n_epochs
        )    
    else:
        train_losses, val_losses, tokens_seen, track_lrs = trainer.train_model(
            n_epochs=args.n_epochs
        )

    if(rank==0):
        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        utils.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

    final_model_file = "model_pg_final.pth"

    trainer.save_checkpoint(final_model_file)

    logger.info(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if(args.run_type=="multi_gpu"):
        torch.distributed.barrier()
        cleanup()

if __name__ == "__main__":

    input_args = get_args()

    if(input_args.run_type=="multi_gpu"):
        world_size = torch.cuda.device_count()
        input_args.world_size=world_size
        mp.spawn(main, args=[input_args], nprocs=world_size,join=True)
    else:
        main(0,input_args)