#!/usr/bin/env python
# coding: utf-8

# In[2]:


from myrenderer import MyRenderer
import os
import hydra
import json
import bpy
#sudo apt-get install ffmpeg


# In[3]:


from omegaconf import OmegaConf,open_dict
import omegaconf


config_home_dir = '/users/j/a/jahn25/puzzlepp/config'
config_home_dir = '../config'

cfg_auto_aggl = omegaconf.OmegaConf.load(config_home_dir+'/auto_aggl.yaml')


cfg = OmegaConf.merge( cfg_auto_aggl )

data_home_dir = '/disk2/data/breaking_bad/'
data_home_dir = '/data/jhahn/data/shape_dataset/'

cfg.experiment_output_path = ''
cfg.inference_dir= data_home_dir+'results'
cfg.renderer.output_path = data_home_dir+'results_render/'
cfg.renderer.mesh_path = f'{data_home_dir}data/'
cfg.ffmpeg_path = '/usr/bin/ffmpeg'


renderer = MyRenderer(cfg)
    
sampled_files = renderer.sample_data_files()
sampled_files = ['0']
# sampled_files = ["1"]

for file in sampled_files:
    transformation, gt_transformation, acc, init_pose = renderer.load_transformation_data(file)
    
    parts = renderer.load_mesh_parts(file, gt_transformation, init_pose)
    
    #save_path = f"/disk2/data/breaking-bad-dataset/results_render/{file}"
    #save_path = data_root_dir+f"results_render/{file}"
    save_path = cfg.renderer.output_path+f'{file}'
    os.makedirs(save_path, exist_ok=True)

    renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

    #if True:
    #    continue
    frame = 0

    # bpy.ops.wm.save_mainfile(filepath=save_path + "test" + '.blend')

    for i in range(transformation.shape[0]):
        renderer.render_parts(
            parts, 
            gt_transformation, 
            transformation[i], 
            init_pose, 
            frame,
        )
        frame += 1


    imgs_path = os.path.join(save_path, "imgs")
    os.makedirs(imgs_path, exist_ok=True)
    renderer.save_video(ffmpeg_path = cfg.ffmpeg_path, imgs_path=imgs_path, video_path=os.path.join(save_path, "video.mp4"), frame=frame)
    renderer.clean()


# In[ ]:



if True:
    quit()



file = '0'

transformation, gt_transformation, acc, init_pose = renderer.load_transformation_data(file)
parts = renderer.load_mesh_parts(file, gt_transformation, init_pose)


if False:
    print(init_pose.shape)
    print(init_pose)
    print(transformation.shape)
    print(transformation)
    print(parts[0])

    #save_path = f"/work/users/j/a/jahn25/breaking-bad-dataset/results_render/{file}"
    #os.makedirs(save_path, exist_ok=True)
    #renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

    #if True:
    #    renderer.clean()
    #    quit()

save_path = f"/work/users/j/a/jahn25/bio-dataset/results_render/{file}"
os.makedirs(save_path, exist_ok=True)
renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

frame = 0

# bpy.ops.wm.save_mainfile(filepath=save_path + "test" + '.blend')

for i in range(transformation.shape[0]):
    renderer.render_parts(
        parts, 
        gt_transformation, 
        transformation[i], 
        init_pose, 
        frame,
    )
    frame += 1


imgs_path = os.path.join(save_path, "imgs")
os.makedirs(imgs_path, exist_ok=True)
renderer.save_video(imgs_path=imgs_path, video_path=os.path.join(save_path, "video.mp4"), frame=frame)
renderer.clean()



if True:
    quit()
# In[77]:


def debug_my():
    import numpy as np
    #everyday/BeerBottle/6da7fa9722b2a12d195232a03d04563a/fractured_1
    data_dict = np.load(os.path.join(project_home_dir, 'data/pc_data/everyday/val/00001.npz'))
    pc = data_dict['part_pcs_gt']
    data_id = data_dict['data_id'].item()
    part_valids = data_dict['part_valids']
    num_parts = data_dict["num_parts"].item()
    mesh_file_path = data_dict['mesh_file_path'].item()
    category = data_dict["category"]
    print('category',category)

    print('num_parts',num_parts)
    print('part_valids',part_valids)

    print('mesh_file_path',mesh_file_path)

    print('part_pcs_gt')
    print(pc.shape)
    print(pc)

debug_my(renderer.inference_path)


# In[5]:



# In[86]:


print(init_pose.shape)
print(init_pose)


# In[87]:


print(transformation.shape)
print(transformation)


# In[13]:


parts = renderer.load_mesh_parts(file, gt_transformation, init_pose)


# In[17]:


type(parts[0])


