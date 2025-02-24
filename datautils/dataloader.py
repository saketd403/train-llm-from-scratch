from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
from torch.utils.data.distributed import DistributedSampler

from datautils.dataset import DatasetPT
import utils

class DataloaderPT:

    def __init__(self,tokenizer, batch_size, max_length,stride,eos_text="<|endoftext|>",dataset_name="gutenberg",run_type="single_gpu",train_ratio=0.90,collate_func=None):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.run_type=run_type
        self.collate_func = collate_func
        self.dataset_name = dataset_name
        self.eos_text = eos_text

    def create_dataloader(self,txt,shuffle=True, drop_last=True, num_workers=0):

        if(self.dataset_name=="gutenberg"):
            dataset = DatasetPT(txt, self.tokenizer, self.max_length, self.stride)

        if(self.run_type=="multi_gpu"):
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size,pin_memory=True, 
                shuffle=False, drop_last=drop_last, 
                sampler=DistributedSampler(dataset), #rank and num_replicas is inferred automatically.
                collate_fn = self.collate_func)
        else:
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size,pin_memory=True, 
                shuffle=shuffle, drop_last=drop_last, 
                num_workers=num_workers,
                collate_fn = self.collate_func)

        return dataloader

    def create_dataloaders(self, text_data, num_workers=0):
        
        split_idx = int(self.train_ratio * len(text_data))

        train_loader = self.create_dataloader(
            text_data[:split_idx],
            drop_last=True,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = self.create_dataloader(
            text_data[split_idx:],
            drop_last=False,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, val_loader
    
    def get_total_steps_epoch(self,data_files):
        
        num_steps=0
        for index, file_path in enumerate(data_files, 1):
            trailing_string = " " + self.eos_text +" "
            text_data = utils.read_text_file(file_path) + trailing_string
            train_loader, val_loader = self.create_dataloaders(
                text_data,
                num_workers=2
            )
            num_steps = num_steps + len(train_loader)

        return num_steps

