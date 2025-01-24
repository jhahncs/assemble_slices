import jahn_src.slice_util as slice_util
import torch
import numpy as np
from chamferdist import ChamferDistance


class VerifierSliceMatching():
    def __init__(self):
        self.chamferDist = ChamferDistance()
        self.threshold = torch.FloatTensor(0.005)

    def score(self, pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):

        _top_pc_list, _bottom_pc_list  = slice_util.top_bottom_pcs(pts, max_num_of_pcs = 100)

        _score_matrix = torch.from_numpy(np.empty([pts.shape[0], int(pts.shape[1]*(pts.shape[1]-1)/2)]))

        for batch_idx, b in enumerate(pts):
            seq_idx = 0
            #print(_top_pc_list_rotated[b].shape)
            for src_idx in range(pts[batch_idx].shape[0]):
                #print(_top_pc_list_rotated[batch_idx][src_idx].shape)
                #print(_noisy_trans_and_rots[...,3:].shape)
                _src_rotated, _, _ = slice_util.rotate_pc(_top_pc_list[batch_idx][src_idx],rot1[batch_idx, src_idx,:])
                _src_rotated, _ = slice_util.trans_pc(_src_rotated,trans1[batch_idx, src_idx,:])

                for tar_idx in range(src_idx + 1, pts[batch_idx].shape[0]):
                    _tar_rotated, _, _ = slice_util.rotate_pc(_bottom_pc_list[batch_idx][tar_idx],rot2[batch_idx, tar_idx, :])
                    _tar_rotated, _ = slice_util.trans_pc(_tar_rotated,trans2[batch_idx, tar_idx,:])
                    #print(_src_rotated.cuda().shape)
                    #print(_tar_rotated.cuda().shape)
                    loss_per_data = self.chamferDist(_src_rotated.unsqueeze(dim=0), _tar_rotated.unsqueeze(dim=0))  # [B*P, N]
                    _score_matrix[batch_idx][seq_idx] = loss_per_data
                    seq_idx += 1

        _score_matrix = _score_matrix/torch.max(_score_matrix)
        return _score_matrix
        