import torch
from pathlib import Path
import tqdm
from tqdm import tqdm
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import utils
from generate import generate

from logger import setup_logger

# Create a logger specific to this module
logger = setup_logger('train')

class Trainer:

    def __init__(self,model,optimizer,config,data_files,loaderObj,save_dir,
                 warmup_steps=10, initial_lr=1e-05, min_lr=1e-6,device="cpu",
                 rank=0,eval_freq=1,save_ckpt_freq=1,print_sample_iter=1,eval_iter=1):

        self.config = config
        self.data_files = data_files
        self.loaderObj = loaderObj
        self.device = device
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.rank = rank
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr=min_lr
        self.model = model
        self.eval_freq=eval_freq
        self.save_ckpt_freq=save_ckpt_freq
        self.print_sample_iter=print_sample_iter
        self.eval_iter = eval_iter

        self.global_step=-1
        self.track_lrs=[]
        self.train_losses=[]
        self.val_losses=[]
        self.track_tokens_seen=[]

        self.tokens_seen=0


    def calc_loss_batch(self,input_batch, target_batch):

        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def train_batch(self,input_batch, target_batch):

        try:
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.global_step < self.warmup_steps:
                lr = self.initial_lr + self.global_step * self.lr_increment
            else:
                progress = ((self.global_step - self.warmup_steps) / (self.total_training_steps - self.warmup_steps))
                lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress))

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.track_lrs.append(lr)
            # if(self.rank==0):
            #    utils.print_memory_usage()
            loss = self.calc_loss_batch(input_batch, target_batch)
            # if(self.rank==0):
            #    utils.print_memory_usage()
            loss.backward()
            # if(self.rank==0):
            #    utils.print_memory_usage()

            if isinstance(self.model, FSDP):
                self.model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
            elif(isinstance(self.model,DDP)):
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0) 
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
            self.optimizer.step()
           
            self.tokens_seen += input_batch.numel()

        except Exception as e:
            
            logger.error(f"An error occurred in train_batch(): {e}")

    def train_epoch(self,epoch_no,train_loader,val_loader,start_context="Every effort moves you"):

        try:
            self.model.train()
            for input_batch, target_batch in train_loader:

                self.train_batch(input_batch, target_batch)
                
                # Optional evaluation step
                if self.global_step % self.eval_freq == 0:

                    train_loss, val_loss = self.evaluate_model(train_loader, val_loader, 
                                                                self.eval_iter)
                    
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.track_tokens_seen.append(self.tokens_seen)

                    if(self.rank==0):
                        logger.info(f"\n Ep {epoch_no+1} (Step {self.global_step}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} \n")

                # Generate text passage           
                if self.global_step % self.print_sample_iter == 0:

                    self.generate_and_print_sample(start_context,temperature=1.0,top_k=5,memory_check=True)

                
                if self.global_step % self.save_ckpt_freq == 0:

                    self.save_checkpoint(f"model_pg_{self.global_step}.pth")
                    #logger.info(f"Successfully saved checkpoint for step {self.global_step}")

        except Exception as e:
            
            logger.error(f"An error occurred in train_epoch(): {e}")


    def train_model(self, n_epochs):

        self.peak_lr = self.optimizer.param_groups[0]["lr"]
        self.total_training_steps = self.loaderObj.get_total_steps_epoch(self.data_files) * n_epochs 
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps 

        try:
            if(self.rank==0):
                pbar = tqdm(total=n_epochs*len(self.data_files))
            for epoch in range(n_epochs):

                # Iterate over the books in the training corpus
                for index, file_path in enumerate(self.data_files, 1):

                    trailing_string = " " + self.config["eos_text"] +" "
                    text_data = utils.read_text_file(file_path) +  trailing_string  #" <|endoftext|> "

                    # Initialize new data loaders for each book
                    train_loader, val_loader = self.loaderObj.create_dataloaders(
                        text_data,
                        num_workers=2
                    )

                    if hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)
                        
                    self.train_epoch(epoch,train_loader, val_loader)
                    
                    if(self.rank==0):
                        pbar.update(1)
                    

        except KeyboardInterrupt:
            self.save_checkpoint(f"model_pg_{self.global_step}_interrupted.pth")

        return self.train_losses, self.val_losses, self.track_tokens_seen, self.track_lrs
    
    def finetune_model(self,n_epochs):

        self.peak_lr = self.optimizer.param_groups[0]["lr"]
        self.total_training_steps = self.loaderObj.get_total_steps_epoch(self.data_files) * n_epochs 
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps 

        try:
            if(self.rank==0):
                pbar = tqdm(total=n_epochs*len(self.data_files))
            for epoch in range(n_epochs):

                # Iterate over the books in the training corpus
                for index, file_path in enumerate(self.data_files, 1):

                    text_data = utils.read_json_file(file_path)

                    # Initialize new data loaders for each book
                    train_loader, val_loader = self.loaderObj.create_dataloaders(
                        text_data,
                        num_workers=2
                    )

                    if hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)
                        
                    self.train_epoch(epoch,train_loader, val_loader,start_context="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is an antonym of 'complicated'?")

                    if(self.rank==0):
                        pbar.update(1)

        except KeyboardInterrupt:
           
            self.save_checkpoint(f"model_pg_{self.global_step}_interrupted.pth")

        return self.train_losses, self.val_losses, self.track_tokens_seen, self.track_lrs



    def generate_and_print_sample(self,start_context,temperature=0.0,top_k=None,memory_check=False,max_new_tokens=200):

        self.model.eval()
        context_size = self.config["context_length"]
    
        encoded = utils.text_to_token_ids(start_context, self.loaderObj.tokenizer,self.config).to(self.device)
       
        token_ids = generate(
            model=self.model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size,temperature=temperature,top_k=top_k,eos_id=self.config["eos_id"])
        
        decoded_text = utils.token_ids_to_text(token_ids, self.loaderObj.tokenizer)
        
        self.model.train()

        if(self.rank==0 and memory_check):
            logger.info(decoded_text.replace("\n", " "))  

    def save_checkpoint(self,file_name):

        if isinstance(self.model, DDP):
            torch.distributed.barrier()

        try:
            file_name = self.save_dir / file_name 

            if self.rank==0:

                if isinstance(self.model, DDP) :
                    torch.save(self.model.module.state_dict(), file_name)
                elif(not isinstance(self.model, FSDP)):
                    torch.save(self.model.state_dict(), file_name)

            if isinstance(self.model, FSDP):
																	 										  
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                    cpu_state = self.model.state_dict()

                if self.rank==0:
                    torch.save(cpu_state, file_name)
                    
            logger.info(f"Saved checkpoint {file_name}")
                    
        except Exception as e:
            logger.error(f"An error occurred while saving checkpoint : {e}")

        if isinstance(self.model, DDP) :
            torch.distributed.barrier()


    def calc_loss_loader(self,data_loader, num_batches=None):

        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches


    def evaluate_model(self,train_loader, val_loader, eval_iter=5):

        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, val_loss

