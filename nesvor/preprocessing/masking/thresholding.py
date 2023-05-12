from skimage.filters import threshold_multiotsu
from skimage.morphology import dilation, ball
import torch
from typing import List
from ...image import Stack


def otsu_thresholding(stacks: List[Stack], nbins=256) -> List[Stack]:
    for stack in stacks:
        thresholds = threshold_multiotsu(
            image=stack.slices.cpu().numpy(), classes=2, nbins=nbins
        )
        assert len(thresholds) == 1
        mask = stack.slices > thresholds[0]
        stack.mask = torch.tensor(
            dilation(mask.squeeze().cpu().numpy(), footprint=ball(3)),
            dtype=mask.dtype,
            device=mask.device,
        ).view(mask.shape)
    return stacks
