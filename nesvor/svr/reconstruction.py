from typing import Callable, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import axisangle2mat
from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.flatten(), y.flatten())


def cg(
    A: Callable, b: torch.Tensor, x0: torch.Tensor, n_iter: int, tol: float = 0.0
) -> torch.Tensor:
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        if dot_r_r_new <= tol:
            return x
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new


def psf_reconstruction(
    transforms: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    params: Dict,
) -> torch.Tensor:
    return slice_acquisition_adjoint(
        transforms,
        params["psf"],
        slices,
        slices_mask,
        vol_mask,
        params["volume_shape"],
        params["res_s"] / params["res_r"],
        params["interp_psf"],
        equalize=True,
    )


class SRR_CG(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.output_relu = output_relu

    def forward(
        self,
        theta: torch.Tensor,
        slices: torch.Tensor,
        volume: torch.Tensor,
        params: Dict,
        p: Optional[torch.Tensor] = None,
        mu: float = 0,
        z: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
        slices_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta

        # A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices

        b = At(y * p if p is not None else y)
        if mu and z is not None:
            b = b + mu * z
        x = cg(AtA, b, volume, self.n_iter, self.tol)

        if self.output_relu:
            return F.relu(x, True)
        else:
            return x

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            params["interp_psf"],
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        slices_mask: Optional[torch.Tensor],
        vol_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            params["interp_psf"],
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
        params: Dict,
        mu: float,
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol


def srr_update(
    transforms: torch.Tensor,
    err: torch.Tensor,
    volume: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    params: Dict[str, Any],
    alpha: float,
    beta: float,
    delta: float,
) -> torch.Tensor:
    # beta = beta * delta * delta
    if p is not None:
        err = p * err
    g = slice_acquisition_adjoint(
        transforms,
        params["psf"],
        err,
        slices_mask,
        vol_mask,
        params["volume_shape"],
        params["res_s"] / params["res_r"],
        params["interp_psf"],
        False,
    )
    if p is not None:
        cmap = slice_acquisition_adjoint(
            transforms,
            params["psf"],
            p,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            params["interp_psf"],
            False,
        )
        cmap_mask = cmap > 0
        g[cmap_mask] /= cmap[cmap_mask]
    reconstructed = F.relu(volume + alpha * g, True)

    g = torch.zeros_like(volume)
    D, H, W = volume.shape[-3:]
    v0 = volume[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1]
    r0 = reconstructed[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                v1 = volume[
                    :, :, 1 + dz : D - 1 + dz, 1 + dy : H - 1 + dy, 1 + dx : W - 1 + dx
                ]
                r1 = reconstructed[
                    :, :, 1 + dz : D - 1 + dz, 1 + dy : H - 1 + dy, 1 + dx : W - 1 + dx
                ]
                d2 = dx * dx + dy * dy + dz * dz
                dv2 = (v1 - v0) ** 2
                b = 1 / (d2 * torch.sqrt(1 + 1 / (d2 * delta * delta) * dv2))
                g[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1] += b * (r1 - r0)
    if p is not None:
        g *= cmap_mask
    reconstructed.add_(g, alpha=alpha * beta)
    return F.relu(reconstructed, True)


def simulate_slices(
    transforms: torch.Tensor,
    volume: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    slices_sim, weight = slice_acquisition(
        transforms,
        volume,
        vol_mask,
        slices_mask,
        params["psf"],
        params["slice_shape"],
        params["res_s"] / params["res_r"],
        True,
        params["interp_psf"],
    )
    return slices_sim, weight
