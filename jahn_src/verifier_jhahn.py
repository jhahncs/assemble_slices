import jahn_src.slice_util as slice_util
import torch
import numpy as np
from chamferdist import ChamferDistance


class VerifierSliceMatching():
    def __init__(self):
        self.chamferDist = ChamferDistance()
        self.threshold = 0.05

    def score(self, pts, trans1, trans2, rot1, rot2, valids, edge_indices, chamfer_distance):
        #print('valids',valids)
        _top_pc_list, _bottom_pc_list  = slice_util.top_bottom_pcs(pts, max_num_of_pcs = 500)

        #_score_matrix = torch.from_numpy(np.empty([pts.shape[0], int(pts.shape[1]*(pts.shape[1]-1)/2)]))
        _score_matrix = torch.zeros((pts.shape[0], int(pts.shape[1]*(pts.shape[1]-1)/2)), dtype=torch.float32)
        for batch_idx, b in enumerate(pts):
            seq_idx = 0
            for (src_idx, tar_idx) in edge_indices[batch_idx]:                
                _score_matrix[batch_idx][seq_idx] = 9999
                seq_idx += 1
        _top_dict = {}
        _bottom_dict = {}
        for batch_idx, b in enumerate(pts):
            seq_idx = -1

            for (src_idx, tar_idx) in edge_indices[batch_idx]:
                seq_idx += 1
                if not (valids[batch_idx][src_idx] and valids[batch_idx][tar_idx]):
                    continue
                if src_idx not in _top_dict:
                    _src_rotated, _, _ = slice_util.rotate_pc(_top_pc_list[batch_idx][src_idx],rot1[batch_idx, src_idx,:])
                    _src_rotated, _ = slice_util.trans_pc(_src_rotated,trans1[batch_idx, src_idx,:])
                    _top_dict[src_idx] = _src_rotated
                else:
                    _src_rotated = _top_dict[src_idx]
                
                if tar_idx not in _bottom_dict:
                    _tar_rotated, _, _ = slice_util.rotate_pc(_bottom_pc_list[batch_idx][tar_idx],rot2[batch_idx, tar_idx, :])
                    _tar_rotated, _ = slice_util.trans_pc(_tar_rotated,trans2[batch_idx, tar_idx,:])
                    _bottom_dict[tar_idx] = _tar_rotated
                else:
                    _tar_rotated = _bottom_dict[tar_idx]
                #print(_src_rotated.cuda().shape)
                #print(_tar_rotated.cuda().shape)
                loss_per_data = self.chamferDist(_src_rotated.unsqueeze(dim=0), _tar_rotated.unsqueeze(dim=0))  # [B*P, N]
                #print(_src_rotated.unsqueeze(dim=0))
                #print(_tar_rotated.unsqueeze(dim=0))
                #print(seq_idx,src_idx,tar_idx,loss_per_data)
                _score_matrix[batch_idx][seq_idx] = loss_per_data

        
        #_score_matrix = _score_matrix/torch.max(_score_matrix)
        return _score_matrix
        