from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
from torch.utils.data.distributed import DistributedSampler

from datautils.dataset import DatasetPT
from datautils.dataset_instruction_finetune import InstructionDataset
import utils


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for instruction_length,item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # instruction_length-1 since we have targets=padded[1:] i.e. it already lacks the first token.
        targets[:instruction_length-1] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    return inputs_tensor, targets_tensor

class DataloaderIF:

    def __init__(self,tokenizer, batch_size, max_length, dataset_name="alpaca", run_type="single_gpu",
                 train_ratio=0.90,collate_func=None):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.run_type=run_type
        self.collate_func = collate_func
        self.dataset_name = dataset_name
    
    def create_dataloader(self,txt,shuffle=True, drop_last=True, num_workers=0):
        
        if(self.dataset_name=="alpaca"):
            dataset = InstructionDataset(txt, self.tokenizer)

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
            text_data = utils.read_json_file(file_path)
            train_loader, val_loader = self.create_dataloaders(
                text_data,
                num_workers=2
            )
            num_steps = num_steps + len(train_loader)

        return num_steps

