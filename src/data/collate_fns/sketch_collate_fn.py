import torch


def sketch_collate_fn(batch):
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
