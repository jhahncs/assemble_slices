#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
import importlib


# In[3]:


from omegaconf import OmegaConf,open_dict
import omegaconf


# In[5]:



config_home_dir = './config'

cfg_ae_global_config = omegaconf.OmegaConf.load(config_home_dir+'/ae/global_config.yaml')
cfg_ae_data = omegaconf.OmegaConf.load(config_home_dir+'/ae/data.yaml')
cfg_ae_vq_vae = omegaconf.OmegaConf.load(config_home_dir+'/ae/vq_vae.yaml')
cfg_ae_global_config = omegaconf.OmegaConf.load(config_home_dir+'/ae/global_config.yaml')
cfg_ae_model = omegaconf.OmegaConf.load(config_home_dir+'/ae/model.yaml')





cfg = OmegaConf.merge(
    cfg_ae_data,
    cfg_ae_vq_vae,
    cfg_ae_global_config,
    cfg_ae_model,

)

data_home_dir = '/disk2/data/breaking_bad/'
data_home_dir = '/disk2/data/shape_dataset/'

data_type_name = 'shape'

cfg.ckpt_path= None
cfg.project_root_path = data_home_dir
cfg.experiment_name = 'shape_epoch10'
cfg.experiment_output_path =  '${project_root_path}/output/autoencoder/${experiment_name}'



cfg.data.data_dir = data_home_dir+f'data/pc_data/{data_type_name}/train/'
cfg.data.data_val_dir = data_home_dir+f'data/pc_data/{data_type_name}/val/'
cfg.data.mesh_data_dir = data_home_dir+'data/'
cfg.data.data_fn = data_type_name+".{}.txt"
cfg.data.batch_size = 3
cfg.data.val_batch_size= 3


cfg.trainer.devices=1
cfg.trainer.max_epochs =  3
cfg.trainer.check_val_every_n_epoch =  1

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config/ae", config_name="global_config")
def main(cfg):
    # fix the seed
    #print(cfg)
    #if True:
    #  return
    
    pl.seed_everything(cfg.train_seed, workers=True)
    # create directories for training outputs

    print(os.path.join(cfg.experiment_output_path, "training"))
    #os.rmdir(os.path.join(cfg.experiment_output_path, "training"))
    
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)
    
    # initialize data
    data_module = DataModule(cfg)
    
    '''
    train_count = 0
    val_count = 0
    for nth_batch, (batch,_) in enumerate(data_module.train_dataloader()):
        print(nth_batch, batch)
        if True:
            break
    if True:
        return
    '''    
        
    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)
    #print(model)
    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)
    # initialize callbacks
    callbacks = init_callbacks(cfg)
    # initialize trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)
    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."
    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    
# In[ ]:


from puzzlefusion_plusplus.vqvae.data.data_module import DataModule
from puzzlefusion_plusplus.vqvae.dataset import dataset
from puzzlefusion_plusplus.vqvae.dataset import pc_dataset
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader

importlib.reload(dataset)
importlib.reload(pc_dataset)

from puzzlefusion_plusplus.vqvae.data.data_module import DataModule
from puzzlefusion_plusplus.vqvae.dataset import dataset
from puzzlefusion_plusplus.vqvae.dataset import pc_dataset
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader



#data_module = DataModule(cfg)

main(cfg)
