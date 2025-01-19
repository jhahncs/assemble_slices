import numpy as np
import os
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d
import random
def combine_obj_files(_dir, output_file_name):
    
    files = [_dir+"/"+f for f in os.listdir(_dir) if os.path.isfile(_dir+"/"+f)]
    #files.sort(key=lambda name: name.split()[1])
    #rot_mat = scipy.spatial.transform.Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()
    _pc_list = []
    f_2_last = []
    with open(output_file_name, 'w') as outfile:

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
                
                _pcs = torch.from_numpy(np.array(_pcs))
                #_pcs = np.array(_pcs)
                #_pcs = trans_pc(_pcs)
                _pcs = rotate_pc(_pcs)
                print(_pcs.shape)
                
                
                #_pcs = _pcs[np.random.choice(range(len(_pcs)), num_of_poinsts_per_part).tolist()]
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

    return _pc_list


def rotate_pc(_part_pcs):
        
    _centroid = (torch.max(_part_pcs, axis=0).values - torch.min(_part_pcs, axis=0).values)/2 +  torch.min(_part_pcs, axis=0).values
    #print(_part_pcs)
    #print(torch.min(_part_pcs, axis=0))
    #print(torch.max(_part_pcs, axis=0))
    #print(_centroid)
    _part_pcs = _part_pcs - _centroid
    noise_quat = torch.from_numpy(np.array([torch.rand(1).item(), 0, 1.0, 0]))

    #print(noise_quat)
    noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
    #print(noise_quat)
    #print(noise_quat.unsqueeze(1))

    _part_pcs = pytorch3d.transforms.quaternion_apply(noise_quat,_part_pcs)
    _part_pcs = _part_pcs + _centroid
    return _part_pcs

def trans_pc(part_pcs):
    #print(torch.rand(1).item())
    part_pcs[:,0] = part_pcs[:,0] - torch.rand(1).item()
    part_pcs[:,2] = part_pcs[:,2] - torch.rand(1).item()
    return part_pcs

