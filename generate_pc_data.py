#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Code to generate point cloud data from the dataset.
"""

import hydra
import os
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader
import numpy as np
from tqdm import tqdm


# In[2]:


from omegaconf import OmegaConf,open_dict
import omegaconf


# In[11]:


@hydra.main(config_path='config/ae', config_name='global_config.yaml')
def main(cfg):
    cfg.data.batch_size = 1
    cfg.data.val_batch_size = 1
    train_loader, val_loader = build_geometry_dataloader(cfg)
    
    def save_data(loader, data_type):
        save_path = f"{cfg.data.save_pc_data_path}/{data_type}/"
        print(save_path)
        os.makedirs(save_path, exist_ok=True)

        for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {data_type} data"):
            data_id = data_dict['data_id'][0].item()
            part_valids = data_dict['part_valids'][0]
            num_parts = data_dict['num_parts'][0].item()
            mesh_file_path = data_dict['mesh_file_path'][0]
            #print(data_id,mesh_file_path)
            graph = data_dict['graph'][0]
            category = data_dict['category'][0]
            part_pcs_gt = data_dict['part_pcs_gt'][0]
            ref_part = data_dict['ref_part'][0]
            
            if False:
                print("part_valids",part_valids.shape,part_valids)
                print("num_parts",num_parts)
                print(mesh_file_path)
                print("graph",graph.shape,graph)
                print("category",category)
                print("part_pcs_gt",part_pcs_gt.shape,part_pcs_gt)
                print("ref_part",ref_part.shape,ref_part)
            np.savez(
                os.path.join(save_path, f'{data_id:05}.npz'),
                data_id=data_id,
                part_valids=part_valids.cpu().numpy(),
                num_parts=num_parts,
                mesh_file_path=mesh_file_path,
                graph=graph.cpu().numpy(),
                category=category,
                part_pcs_gt=part_pcs_gt.cpu().numpy(),
                ref_part=ref_part.cpu().numpy()
            )
            print(f"Saved {data_id:05}.npz in {data_type} data.")
            #if True:
            #    break

    # Save train data
    save_data(train_loader, 'train')
    # Save validation data
    save_data(val_loader, 'val')
    



# In[12]:


import importlib
from puzzlefusion_plusplus.vqvae.dataset import dataset
importlib.reload(dataset)
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader



# In[13]:


# decompress the compessed breaking bad dataset
#python decompress.py --data_root /disk2/data/breaking-bad-dataset/data --subset everyday --category all


# In[14]:

data_root_dir = '/disk2/data/breaking_bad/'
cfg = omegaconf.OmegaConf.load('config/ae/data.yaml')
#cfg.data.save_pc_data_path = '/work/users/j/a/jahn25/breaking-bad-dataset/data/pc_data_test/everyday/'
#cfg.data.mesh_data_dir = '/work/users/j/a/jahn25/breaking-bad-dataset/data/'
cfg.data.save_pc_data_path = f'{data_root_dir}/pc_data/everyday/'
cfg.data.mesh_data_dir = f'{data_root_dir}/'
cfg.data.data_dir=f'{data_root_dir}/pc_data/everyday/train/'
cfg.data.data_val_dir=f'{data_root_dir}/pc_data/everyday/val/'
cfg.data.data_fn = "everyday.{}.txt"
main(cfg)   

# In[ ]:





# In[ ]:




