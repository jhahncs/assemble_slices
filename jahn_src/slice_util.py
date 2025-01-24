import numpy as np
import os
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d
import random
from chamferdist import ChamferDistance
chamferDist = ChamferDistance()

def combine_obj_files(_dir_list, output_dir):
    _obj_list = []
    for _dir in _dir_list:
        files = [_dir+"/"+f for f in os.listdir(_dir) if os.path.isfile(_dir+"/"+f)]
        files.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".obj",'')), reverse=False)
        
        _pc_list = []
        f_2_last = []
        with open(output_dir+"/"+_dir.split("/")[-1]+".obj", 'w') as outfile:

            f_2_last.append(0)
            for _,fname in enumerate(files):
                
                _c = 0
                with open(fname) as infile:
                    if fname.endswith('piece.obj'):
                        continue
                    print(fname)
                    _pcs = []
                    for line in infile:
                        if line.lower().startswith('v'):
                            _c += 1
                            _arr = line[2:].split()
                            _arr = np.array([float(a) for a in _arr])                    
                            _pcs.append(_arr)
                    
                    _pcs = torch.from_numpy(np.array(_pcs)).type(torch.float32)
                    #_pcs = np.array(_pcs)
                    #_pcs = trans_pc(_pcs)
                    #_pcs = rotate_pc(_pcs)
                    print(_pcs.shape)                                
                    _pc_list.append(_pcs)
                    
                    for _arr in _pcs.numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                f_2_last.append(_c)

            _delta = 0
            for fi, fname in enumerate(files):
                _delta += f_2_last[fi]
                with open(fname) as infile:
                    if fname.endswith('piece.obj'):
                        continue                
                    for line in infile:
                        if line.lower().startswith('f'):
                            _arr = line[2:].split()
                            outfile.write(f'f {int(_arr[0])+_delta} {int(_arr[1])+_delta} {int(_arr[2])+_delta}\n')
        _obj_list.append(_pc_list)
    return _obj_list

'''
# very slow
def pcs_matched(pc_1, pc_2, point_dist_threshold = 0.03, matched_threshold = 0.5):
    _matched = 0
    #_bottom_flag = torch.zeros(len(pc_2), dtype=torch.bool)
    _b_idx_last = 0
    for _t in pc_1:
        for _b_idx in range(_b_idx_last, len(pc_2)):
            _dist = torch.cdist(_t.unsqueeze(dim=0), pc_2[_b_idx].unsqueeze(dim=0))[0][0]
            #print(_dist)
            if torch.lt(_dist, point_dist_threshold):
                #_bottom_flag[_b_idx] = True
                _matched += 1
                _b_idx_last = _b_idx
                break

    if _matched >= len(pc_1)*matched_threshold:
        return True, _matched
    else:
        return False, _matched
'''

def top_bottom_pcs(_obj_list, max_num_of_pcs = 1000):
    '''
    _obj_list: [parts, the max number of points, 3]
    '''
    _top_pc_list_batch = None
    _bottom_pc_list_batch = None

    for _pc_list in _obj_list:
            
        _top_pc_list  = None
        _bottom_pc_list  = None
        for _part_pcs in _pc_list:
            #_top_pcs = torch.index_select(_part_pcs, 0, _part_pcs[:,1] >= torch.max(_part_pcs, axis=0).values[1].item() - 0.001)
            #print(_part_pcs.shape)
            #print(torch.max(_part_pcs, axis=0).values)
            _top_pcs = _part_pcs[_part_pcs[:,1] >= torch.max(_part_pcs, axis=0).values[1].item() - 0.001]
            _top_pcs = _top_pcs[np.random.choice(len(_top_pcs), max_num_of_pcs)]
            _top_pcs.sort()
            _top_pcs = _top_pcs.unsqueeze(dim=0)
            #_top_pc_list.append(_top_pcs)
            if _top_pc_list is None:
                _top_pc_list = _top_pcs
            else:
                #print('_top_pc_list',_top_pc_list.shape)
                #print('_top_pcs',_top_pcs.shape)
                _top_pc_list = torch.cat((_top_pc_list,_top_pcs),0)
                #print('_top_pc_list',_top_pc_list.shape)

            _bottom_pcs = _part_pcs[_part_pcs[:,1] <= torch.min(_part_pcs, axis=0).values[1].item() + 0.001]
            _bottom_pcs = _bottom_pcs[np.random.choice(len(_bottom_pcs), max_num_of_pcs)]
            _bottom_pcs.sort()
            _bottom_pcs = _bottom_pcs.unsqueeze(dim=0)
            if _bottom_pc_list is None:
                _bottom_pc_list = _bottom_pcs
            else:
                _bottom_pc_list = torch.cat((_bottom_pc_list,_bottom_pcs),0)
        #_top_pc_list = np.array(_top_pc_list)
        #_bottom_pc_list = np.array(_bottom_pc_list)
        _top_pc_list = _top_pc_list.unsqueeze(dim=0)
        _bottom_pc_list = _bottom_pc_list.unsqueeze(dim=0)
        
        if _top_pc_list_batch is None:
            _top_pc_list_batch = _top_pc_list
        else:
            _top_pc_list_batch = torch.cat((_top_pc_list_batch,_top_pc_list),0)

        if _bottom_pc_list_batch is None:
            _bottom_pc_list_batch = _bottom_pc_list
        else:
            _bottom_pc_list_batch = torch.cat((_bottom_pc_list_batch,_bottom_pc_list),0)
   



    #_top_pc_list_batch = np.array(_top_pc_list_batch)
    #_bottom_pc_list_batch = np.array(_bottom_pc_list_batch)

    return _top_pc_list_batch, _bottom_pc_list_batch


def rotate_pc(_part_pcs, rot_quat = None):
    shape_len = len(_part_pcs.shape)
    if shape_len == 4 or shape_len == 3: # [Batch, Parts, num of points, (x,y,z)]
        _centroid_min = torch.min(_part_pcs, axis = shape_len - 2).values
        _centroid_max = torch.max(_part_pcs, axis = shape_len - 2).values
        _centroid = (_centroid_max- _centroid_min)/2 +  _centroid_min
        _centroid = torch.repeat_interleave(_centroid.unsqueeze(shape_len - 2), _part_pcs.shape[shape_len - 2], dim=shape_len - 2)        
    else:
        _centroid = (torch.max(_part_pcs, axis=0).values - torch.min(_part_pcs, axis=0).values)/2 +  torch.min(_part_pcs, axis=0).values
        
    _part_pcs = _part_pcs - _centroid

    if rot_quat is None:
        rot_quat = torch.from_numpy(np.array([torch.rand(1).item(), 0, 1.0, 0]))
    elif rot_quat is not None and (shape_len == 4 or shape_len == 3):
        rot_quat = torch.repeat_interleave(rot_quat.unsqueeze(shape_len - 2),  _part_pcs.shape[shape_len - 2], dim= shape_len - 2)   

    rot_quat = rot_quat / rot_quat.norm(dim=-1, keepdim=True)
    #else:
    #print(rot_quat.shape)
    #print(_part_pcs.shape)
    #real_parts = _part_pcs.new_zeros(_part_pcs.shape[:-1] + (1,))
    #print(real_parts.shape)
    #point_as_quaternion = torch.cat((real_parts, _part_pcs), -1)
    #print('_centroid',_centroid.shape)
    #print('rot_quat',rot_quat.shape)
    #print('_part_pcs',_part_pcs.shape)
    
    _part_pcs = pytorch3d.transforms.quaternion_apply(rot_quat,_part_pcs)
    #print('_part_pcs',_part_pcs.shape)
    _part_pcs = _part_pcs + _centroid

    return _part_pcs, _centroid, rot_quat

def trans_pc(_part_pcs, trans_vec = None):
    shape_len = len(_part_pcs.shape)
    if trans_vec is None:
        trans_vec = torch.rand(3)
        trans_vec[1] = 0
    elif trans_vec is not None and (shape_len == 4 or shape_len == 3): # [Batch, Parts, num of points, (x,y,z)]
        trans_vec = torch.repeat_interleave(trans_vec.unsqueeze(shape_len - 2),  _part_pcs.shape[shape_len - 2], dim= shape_len - 2)   


    _part_pcs = torch.sub(_part_pcs[...,:], trans_vec)

    return _part_pcs, trans_vec