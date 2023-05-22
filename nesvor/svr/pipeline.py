import logging
from typing import List, Optional, Tuple
import torch
import numpy as np
from ..transform import axisangle2mat, RigidTransform, mat_update_resolution
from .registration import SVR
from .outlier import EM
from .reconstruction import psf_reconstruction, srr_update, simulate_slices, SRR_CG
from ..utils import get_PSF, ncc_loss, ssim_loss, DeviceType, PathType
from ..image import Volume, Slice, load_volume, load_mask
from ..nesvor.data import PointDataset


def _initial_mask(
    slices: List[Slice],
    output_resolution: float,
    sample_mask: Optional[PathType],
    sample_orientation: Optional[PathType],
    device: DeviceType,
) -> Tuple[Volume, bool]:
    dataset = PointDataset(slices)
    mask = dataset.mask
    if sample_mask is not None:
        mask = load_mask(sample_mask, device)
    transformation = None
    if sample_orientation is not None:
        transformation = load_volume(
            sample_orientation,
            device=device,
        ).transformation
    mask = mask.resample(output_resolution, transformation)
    return mask, sample_mask is None


def _check_resolution_and_shape(slices: List[Slice]) -> Tuple[float, float]:
    res_inplane = []
    thicknesses = []
    for s in slices:
        res_inplane.append(float(s.resolution_x))
        res_inplane.append(float(s.resolution_y))
        thicknesses.append(float(s.resolution_z))
        if s.image.shape != slices[0].image.shape:
            raise ValueError("Input data should have the same in-plane matrix size")

    if (
        max(res_inplane) - min(res_inplane) > 0.001
        or max(thicknesses) - min(thicknesses) > 0.001
    ):
        logging.warning(
            "The input data should be isotropic in-plane and have the same resolution and thickness!"
        )
    res_s = np.mean(res_inplane).item()
    s_thick = np.mean(thicknesses).item()
    return res_s, s_thick


def _normalize(
    slices_tensor: torch.Tensor, slices_mask: torch.Tensor, output_intensity_mean: float
) -> Tuple[torch.Tensor, float, float]:
    mean_intensity = slices_tensor[slices_mask].mean().item()
    max_intensity = slices_tensor[slices_mask].max().item()
    min_intensity = slices_tensor[slices_mask].min().item()
    slices_tensor = slices_tensor * (output_intensity_mean / mean_intensity)
    max_intensity = max_intensity * (output_intensity_mean / mean_intensity)
    min_intensity = min_intensity * (output_intensity_mean / mean_intensity)
    return slices_tensor, max_intensity, min_intensity


def slice_to_volume_reconstruction(
    slices: List[Slice],
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    n_iter: int = 3,
    n_iter_rec: List[int] = [7, 7, 21],
    global_ncc_threshold: float = 0.5,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_global_exclusion: bool = False,
    no_local_exclusion: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
    **unused
):
    # check data
    res_s, s_thick = _check_resolution_and_shape(slices)
    res_r = output_resolution

    # get data
    mask, is_refine_mask = _initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )
    shape = mask.image.shape
    volume_transform = mask.transformation
    volume_mask = mask.image > 0

    slices_tensor = torch.stack([s.image for s in slices], dim=0)
    slices_mask_backup = torch.stack([s.mask for s in slices], dim=0)
    slices_mask = slices_mask_backup.clone()
    slices_transform = RigidTransform.cat([s.transformation for s in slices])
    slices_transform = volume_transform.inv().compose(slices_transform)
    slices_transform_ax = slices_transform.axisangle()

    params = {
        "psf": get_PSF(
            r_max=5,
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
            device=device,
            # psf_type="sinc",
        ),
        "slice_shape": slices_tensor.shape[-2:],
        "interp_psf": False,
        "res_s": res_s,
        "res_r": res_r,
        "volume_shape": shape,
        "s_thick": s_thick,
    }

    # data normalization
    slices_tensor, max_intensity, min_intensity = _normalize(
        slices_tensor, slices_mask, output_intensity_mean
    )

    # outer loop
    for i in range(n_iter):
        logging.info("outer %d", i)
        # slice-to-volume registration
        volume: torch.Tensor
        if i > 0:  # skip slice-to-volume registration for the first iteration
            svr = SVR(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=30,
                optimizer={"name": "gd", "momentum": 0.1},
                loss={"name": "ncc", "win": None},
                auto_grad=False,
            )
            svr.volume_mask = volume_mask
            svr.slices_mask = slices_mask_backup
            slices_transform_ax, _ = svr(
                slices_transform_ax,
                volume,
                slices_tensor,
                params,
            )
        slices_transform_mat = mat_update_resolution(
            axisangle2mat(slices_transform_ax), 1, params["res_r"]
        )
        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            slices_sim, _ = simulate_slices(
                slices_transform_mat, volume, volume_mask, slices_mask_backup, params
            )
            ncc = -ncc_loss(
                slices_sim,
                slices_tensor,
                slices_mask_backup,
                win=None,
                reduction="none",
            )
            excluded = ncc < global_ncc_threshold
            num_excluded = torch.count_nonzero(excluded).item()
            slices_mask = slices_mask_backup.clone()
            if num_excluded == excluded.shape[0]:
                logging.warning("All slices excluded according to global NCC. Reset.")
                excluded = torch.zeros_like(excluded)
                num_excluded = 0
            slices_mask[excluded] = False
            logging.info(
                "global structural exlusion: mean ncc = %f, num_excluded = %d, mean ncc after exclusion = %f",
                ncc.mean().item(),
                num_excluded,
                ncc[~excluded].mean().item(),
            )
        # PSF reconstruction & volume mask
        if is_refine_mask:
            volume_mask = psf_reconstruction(
                slices_transform_mat, slices_mask.float(), None, None, params
            )
            volume_mask = SRR_CG(n_iter=2)(
                slices_transform_mat, slices_mask.float(), volume_mask, params
            )
            volume_mask = volume_mask > 0.3
        volume = psf_reconstruction(
            slices_transform_mat,
            slices_tensor,
            None if with_background else slices_mask,
            None if with_background else volume_mask,
            params,
        )
        # init EM
        em = EM(max_intensity, min_intensity)
        p_voxel = torch.ones_like(slices_tensor)
        # super-resolution reconstruction (inner loop)
        for j in range(n_iter_rec[i]):
            logging.info("inner %d", j)
            # simulate slices
            if with_background:
                slices_sim, slices_weight = simulate_slices(
                    slices_transform_mat, volume, None, None, params
                )
            else:
                slices_sim, slices_weight = simulate_slices(
                    slices_transform_mat, volume, volume_mask, slices_mask, params
                )
            # scale
            scale = p_voxel * slices_tensor * slices_mask * (slices_weight > 0.99)
            scale = (scale * slices_sim).sum((1, 2, 3)) / (scale * slices_tensor).sum(
                (1, 2, 3)
            )
            scale[~torch.isfinite(scale)] = 1.0
            # err
            err = slices_tensor * scale.view(-1, 1, 1, 1) - slices_sim
            # EM robust statistics
            if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
                p_voxel, p_slice = em(err, slices_weight, scale, slices_mask, 1)
                if no_pixel_robust_statistics:  # reset p_voxel
                    p_voxel = torch.ones_like(slices_tensor)
            p = p_voxel
            if not no_slice_robust_statistics:
                p = p_voxel * p_slice.view(-1, 1, 1, 1)
            # local structural exclusion
            if not no_local_exclusion:
                ssim_map = -ssim_loss(slices_sim, slices_tensor, slices_mask)
                p = p * torch.where(ssim_map > local_ssim_threshold, 1.0, 0.1)
                logging.info(
                    "local structural exlusion: mean ssim = %f, ratio downweighted = %f",
                    ssim_map[slices_mask].mean().item(),
                    (ssim_map[slices_mask] <= local_ssim_threshold)
                    .float()
                    .mean()
                    .item(),
                )
            # super-resolution update
            beta = max(0.01, 0.08 / (2**i))
            alpha = min(1, 0.05 / beta)
            delta = 150 * output_intensity_mean
            volume = srr_update(
                slices_transform_mat,
                err,
                volume,
                None if with_background else slices_mask,
                None if with_background else volume_mask,
                p,
                params,
                alpha=alpha,
                beta=beta,
                delta=delta,
            )
    # reconstruction finished
    # prepare outputs
    output_volume = Volume(
        volume.squeeze(),
        transformation=volume_transform,
        resolution_x=res_r,
        resolution_y=res_r,
        resolution_z=res_r,
    )
    slices_transform = volume_transform.compose(RigidTransform(slices_transform_ax))

    slices_sim, _ = simulate_slices(
        slices_transform_mat, volume, volume_mask, slices_mask, params
    )
    output_slices = []
    simulated_slices = []
    for i, s in enumerate(slices):
        output_slice = s.clone()
        output_slice.transformation = slices_transform[i]
        output_slices.append(output_slice)
        simulated_slice = s.clone()
        simulated_slice.image = slices_sim[i]
        simulated_slice.transformation = slices_transform[i]
        simulated_slices.append(simulated_slice)

    return output_volume, output_slices, simulated_slices
