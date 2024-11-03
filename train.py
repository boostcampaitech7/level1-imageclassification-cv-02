import os
import gc
import timm
import wandb
import argparse
import importlib
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(config_path, use_wandb=True, sweep_dict=None):
    # load train config
    config = OmegaConf.load(config_path)
    print(config)

    # import data module class
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # data module instance
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42) # 
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # import pl module class
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )

    # model instance
    model = ModelModuleClass(config)

    # Wandb logger
    logger = None
    if use_wandb:
        logger = WandbLogger(project="Sketch", name=config.wandb_name)

    # callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )

    # trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        precision='16-mixed',   
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache() # cleaning cuda memory

    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()

    main(args.config, args.use_wandb)
