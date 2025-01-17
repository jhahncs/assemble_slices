import os
import torch
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from puzzlefusion_plusplus.denoiser.dataset.dataset import build_geometry_dataloader
import importlib
from puzzlefusion_plusplus.denoiser.dataset import dataset
from omegaconf import OmegaConf,open_dict
import omegaconf

config_home_dir = './config'


data_home_dir = '/disk2/data/breaking_bad/'
data_home_dir = '/disk2/data/shape_dataset/'
data_type_name = 'shape'



cfg_denoiser_data = omegaconf.OmegaConf.load(config_home_dir+'/denoiser/data.yaml')
cfg_denoiser_encode = omegaconf.OmegaConf.load(config_home_dir+'/denoiser/encoder.yaml')
cfg_denoiser_global_config = omegaconf.OmegaConf.load(config_home_dir+'/denoiser/global_config.yaml')
cfg_denoiser_model = omegaconf.OmegaConf.load(config_home_dir+'/denoiser/model.yaml')

cfg = OmegaConf.merge(

    cfg_denoiser_data,
    cfg_denoiser_encode,
    cfg_denoiser_global_config,
    cfg_denoiser_model
)



cfg.data.data_dir = data_home_dir+f'data/pc_data/{data_type_name}/train/'
cfg.data.data_val_dir = data_home_dir+f'data/pc_data/{data_type_name}/val/'
cfg.data.mesh_data_dir = data_home_dir+'data/'
cfg.data.data_fn = data_type_name+".{}.txt"
cfg.data.batch_size = 3
cfg.data.val_batch_size= 3

cfg.experiment_name = 'shape_epoch10'
cfg.model.encoder_weights_path =  f'{data_home_dir}/output/autoencoder/{cfg.experiment_name}'+'/training/last.ckpt'


cfg.ckpt_path= None
cfg.experiment_output_path = data_home_dir+'output/denoiser/${experiment_name}/'
cfg.trainer.max_epochs =  3
cfg.trainer.check_val_every_n_epoch =  1


cfg.trainer.strategy='ddp'
# In[4]:


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # print_callback = PrintCallback()
    return [checkpoint_monitor, lr_monitor]


#@hydra.main(version_base=None, config_path="config/denoiser", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    train_loader, val_loader = build_geometry_dataloader(cfg)
    
    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    if cfg.model.encoder_weights_path is not None:
        encoder_weights = torch.load(cfg.model.encoder_weights_path)['state_dict']
        model.encoder.load_state_dict({k.replace('ae.', ''): v for k, v in encoder_weights.items()})
        # freeze the encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)
    
    # initialize trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer
    )

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path
    )



#77222a952435b1b516e70facc0fd8554f280f918
main(cfg)

