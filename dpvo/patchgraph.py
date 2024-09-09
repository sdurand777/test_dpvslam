import numpy as np
import torch
from einops import asnumpy, reduce, repeat
import einops

from . import projective_ops as pops
from .lietorch import SE3
from .loop_closure.optim_utils import reduce_edges
from .utils import *


class PatchGraph:
    """ Dataclass for storing variables """

    def __init__(self, cfg, P, DIM, pmem, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")



    # loop closure based on deep learning
    def edges_loop(self):
        """ Adding edges from old patches to new frames """
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n - self.cfg.REMOVAL_WINDOW # l is the upper bound for "old" patches

        if l <= 0:
            return torch.empty(2, 0, dtype=torch.long, device='cuda')

        # on a l >= 1 donc on a plus de frames self.n que la removal_window la loop closure est au de la de removal window et en dessous de 1000 (lc_range)
        # config
#         import pdb; pdb.set_trace()

        # create candidate edges
        jj, kk = flatmeshgrid(
            torch.arange(self.n - self.cfg.GLOBAL_OPT_FREQ, self.n - self.cfg.KEYFRAME_INDEX, device="cuda"),
            torch.arange(max(l - lc_range, 0) * self.M, l * self.M, device="cuda"), indexing='ij')
        ii = self.ix[kk]

        # compute candidate edges
#         import pdb; pdb.set_trace()

        # Remove edges which have too large flow magnitude
        flow_mg, val = pops.flow_mag(SE3(self.poses), self.patches[...,1,1].view(1,-1,3,1,1), self.intrinsics, ii, jj, kk, beta=0.5)

        # # compute raw flow mag
#         # import pdb; pdb.set_trace()

        # # Alternative to einops.reduce for summing across a dimension
        # def reduce_sum(tensor, dim):
        #     return tensor.sum(dim=dim)
        #
        # tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
        # tensor = torch.tensor(tensor)  # Convert numpy array to PyTorch tensor
        # # Réduction en utilisant einops.reduce (moyenne sur la première dimension)
        # #result = reduce(tensor, 'b c -> c', 'mean')
        # result = tensor.mean(dim=0)  # mean across dimension 0


        # # version original
#         # import pdb; pdb.set_trace()
        # flow_mg_sum = einops.reduce(flow_mg * val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).float()

        # version san einops
#         import pdb; pdb.set_trace()
        flow_mg_mult = flow_mg * val
        reshaped_tensor = flow_mg_mult.view(1, -1, self.M, 1, 1)  # Shape (1, 11, 96, 1, 1)
        flow_mg_sum = reshaped_tensor.sum(dim=2)  # Réduit la dimension M
        flow_mg_sum = flow_mg_sum.squeeze()  # Pour obtenir une forme (11,)
   

        # # compute cummulated flow ?
#         # import pdb; pdb.set_trace()
        # num_val = einops.reduce(val, '1 (fl M) 1 1 -> fl', 'sum', M=self.M).clamp(min=1)


        # version sans einops
#         import pdb; pdb.set_trace()
        reshaped_val = val.view(1, -1, self.M, 1, 1)  # Shape (1, 11, 96, 1, 1)
        num_val = reshaped_val.sum(dim=2)  # Réduit la dimension M, forme finale (1, 11, 1, 1)
        num_val = num_val.clamp(min=1)
        num_val = num_val.squeeze()

        flow_mag = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)


        # compute flow mag
#         import pdb; pdb.set_trace()
        
        mask = (flow_mag < self.cfg.BACKEND_THRESH)

        # compute flow mag
#         import pdb; pdb.set_trace()
 
        #es = reduce_edges(asnumpy(flow_mag[mask]), asnumpy(ii[::self.M][mask]), asnumpy(jj[::self.M][mask]), max_num_edges=1000, nms=1)
        es = reduce_edges(  flow_mag[mask].detach().cpu().numpy(), 
                            ii[::self.M][mask].detach().cpu().numpy(), 
                            jj[::self.M][mask].detach().cpu().numpy(), 
                            max_num_edges=1000, 
                            nms=1)
        
        # compute mask to keep relevant edges
#         import pdb; pdb.set_trace()

        edges = torch.as_tensor(es, device=ii.device)

        #ii, jj = repeat(edges, 'E ij -> ij E M', M=self.M, ij=2)

# Transpose edges to get shape (2, E)
        edges_transposed = edges.T  # (2, E)

# Expand the transposed tensor to have the third dimension M
        edges_expanded = edges_transposed.unsqueeze(-1).expand(-1, -1, self.M)  # (2, E, M)

# Split the result into ii and jj along the first dimension (which corresponds to ij)
        ii, jj = edges_expanded[0], edges_expanded[1] 


        kk = ii.mul(self.M) + torch.arange(self.M, device=ii.device)
        
        # check new indices
#         import pdb; pdb.set_trace()

        return kk.flatten(), jj.flatten()



    def normalize(self):
        """ normalize depth and poses """
        s = self.patches_[:self.n,:,2].mean()
        self.patches_[:self.n,:,2] /= s
        self.poses_[:self.n,:3] *= s
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[[0]]).inv()).data

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]



    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
