import torch
import torch.nn.functional as F

from copy import deepcopy

from .otdd.utils import generate_unit_convolution_projections


def proj_support(x, y, num_projections, type_proj="linear", batch_projs=1000):
    """
        Code for convolution projection taken from 
        https://github.com/hainn2803/s-OTDD/blob/main/otdd/pytorch/utils.py
    
        Parameters
        ----------
        - x: (n, d) if linear or (n, c, d, d) for conv
        - y: (m, d) if linear or (m, c, d, d) for conv
        - num_projections: int
        - type_proj: "linear" or "conv"
        - batch_projs: int, batchs for the "conv" projections

        Output
        ------
        - Xp: (n, num_projections)
        - Yp: (m, num_projections)
    """
    d = x.shape[-1]
    device, dtype = x.device, x.dtype

    if type_proj == "linear":
        theta = torch.randn(num_projections, d, dtype=dtype, device=device)
        theta = F.normalize(theta, p=2, dim=-1)
    
        # Projections on R
        projs_x = torch.einsum("...nd,ld->...nl", x, theta)  # shape (n_samples, n_projs)
        projs_y = torch.einsum("...nd,ld->...nl", y, theta)
        
    if type_proj == "linear_chunked":
        n_chunks = num_projections // batch_projs + 1

        L_x = []
        L_y = []
        
        for n in range(n_chunks):
            n_projs = batch_projs if n<n_chunks-1 else (num_projections)%batch_projs

            Xp = deepcopy(x)
            Yp = deepcopy(y)
            
            if n_projs>0:
                theta = torch.randn(n_projs, d, dtype=dtype, device=device)
                theta = F.normalize(theta, p=2, dim=-1)
    
                # Projections on R
                Xp = torch.einsum("...nd,ld->...nl", x, theta)  # shape (n_samples, n_projs)
                Yp = torch.einsum("...nd,ld->...nl", y, theta)
                
                L_x.append(Xp)
                L_y.append(Yp)
                           
        projs_x = torch.cat(L_x, dim=1)
        projs_y = torch.cat(L_y, dim=1)

    elif type_proj == "conv":
        assert d == 32, "only implemented for CIFAR10"
        
        n_chunks = num_projections // batch_projs + 1

        L_x = []
        L_y = []

        for n in range(n_chunks):
            n_projs = batch_projs if n<n_chunks-1 else (num_projections)%batch_projs

            Xp = deepcopy(x)
            Yp = deepcopy(y)

            if n_projs>0:
                convolutions = generate_unit_convolution_projections(
                    image_size=d,
                    num_channels=x.shape[1],
                    num_projection=n_projs,
                    device=device,
                    dtype=dtype
                )
        
                for conv in convolutions:
                    Xp = conv(Xp).detach()
                    Yp = conv(Yp).detach()
    
                L_x.append(Xp.squeeze(-1).squeeze(-1))
                L_y.append(Yp.squeeze(-1).squeeze(-1))

        projs_x = torch.cat(L_x, dim=1)
        projs_y = torch.cat(L_y, dim=1)

    return projs_x, projs_y
