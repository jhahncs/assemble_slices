import os
import torch
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from puzzlefusion_plusplus.denoiser.dataset.dataset import build_geometry_dataloader
import importlib
from puzzlefusion_plusplus.denoiser.dataset import dataset
from omegaconf import OmegaConf,open_dict
import omegaconf
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

config_home_dir = './config'


data_home_dir = '/disk2/data/breaking_bad/'
data_home_dir = '/data/jhahn/data/shape_dataset/'
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


cfg.project_root_path = data_home_dir
cfg.data.data_dir = data_home_dir+f'pc_data/{data_type_name}/train/'
cfg.data.data_val_dir = data_home_dir+f'pc_data/{data_type_name}/val/'
cfg.data.mesh_data_dir = data_home_dir+'data/'
cfg.data.data_fn = data_type_name+".{}.txt"
cfg.data.batch_size = 32
cfg.data.val_batch_size= 32
cfg.data.num_workers = 32

cfg.experiment_name = 'shape_epoch10'
cfg.model.encoder_weights_path =  f'{data_home_dir}/output/autoencoder/{cfg.experiment_name}'+'/training/last.ckpt'


cfg.ckpt_path= None
cfg.experiment_output_path = data_home_dir+'output/denoiser/${experiment_name}/'
cfg.trainer.max_epochs =  100
cfg.trainer.check_val_every_n_epoch =  5
cfg.trainer.log_every_n_steps =  1
cfg.trainer.precision =  32


cfg.logger._target_ = 'pytorch_lightning.loggers.WandbLogger'
cfg.checkpoint_monitor._target_ = 'pytorch_lightning.callbacks.ModelCheckpoint'



cfg.trainer.strategy='ddp'
# In[4]:

torch.set_float32_matmul_precision('medium')

# create directories for training outputs
os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

checkpoint_callback = ModelCheckpoint(monitor='val_loss/total_loss',
                                    save_top_k=1,
                                    save_last=True,
                                    save_weights_only=False,
                                    verbose=False,
                                    mode='min',
                                    every_n_epochs = cfg.trainer.check_val_every_n_epoch,
                                    filename = "{epoch}",
                                    dirpath = f'{cfg.experiment_output_path}/training'
                                    )


earlystopping = EarlyStopping(monitor='val_loss/total_loss', mode='min')

'''

        'dropout':{
            'distribution': 'uniform',  # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
            'min':0.1,                 # 최소값을 설정합니다.
            'max':0.5                  # 최대값을 설정합니다.
        },
'''
sweep_config = {
    'method': 'grid', 
    
    'parameters': {

        'lr':{"values": [1e-5]},
        'dropout':{"values": [0.2]},
        'layers':{"values": [3, 5]},
        
        'optimizer':{"values": ["adam"]}
        
    },
    'name' : 'denoiser',
    'metric':{'name':'val_loss/total_loss', 'goal':'minimize'},
    #'early_terminate' : {'type' : 'hyperband', 'max_iter' : 30, 's' : 2, 'eta': 3},
    "entity" : 'sts',
    'project' : 'puzzlefusion_plusplus'
}
#@hydra.main(version_base=None, config_path="config/denoiser", config_name="global_config")
def sweep_train(config=None):
    _run = wandb.init(config=config, dir=f'{cfg.experiment_output_path}')
    print(wandb.config)
    
    name_str = "_".join(
        [f"{key}_{wandb.config[key]}" for key in sweep_config['parameters']]
    )
    _run.name = name_str
    print(name_str)
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)



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



    wandb_logger = WandbLogger(
            log_model=False, #"all" False=no checkpoint
            name=f'{name_str}',
            #offline = True,
            project = 'puzzlefusion_plusplus',
            save_dir = f'{cfg.experiment_output_path}/training'
            #entity=cfg.wandb.wandb_entity
    )
    wandb_logger.watch(model)
    #model = torch.compile(model)
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project='puzzlefusion_plusplus')


    
    # initialize trainer
    trainer = pl.Trainer(
        callbacks= [checkpoint_callback],
        logger=wandb_logger,
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
    #wandb.join()
    _run.finish()




sweep_id = wandb.sweep(
            sweep=sweep_config,     # config 딕셔너리를 추가합니다.
            project='puzzlefusion_plusplus',# project의 이름을 추가합니다.
)
wandb.agent(
        sweep_id=sweep_id,      # sweep의 정보를 입력하고
        function=sweep_train,   # train이라는 모델을 학습하는 코드를
        #count=5             # 총 5회 실행해봅니다.
)

#77222a952435b1b516e70facc0fd8554f280f918

