import os
from typing import List
import numpy as np
import torch
import logging
from .architectures import model_architecture, INPUT_SHAPE
from ....image import Stack
from .... import __checkpoint_dir, __iqa3d


def get_iqa3d_checkpoint() -> str:
    model_dir = __checkpoint_dir
    model_name = "iqa3d.hdf5"
    if not os.path.exists(os.path.join(model_dir, model_name)):
        logging.info(
            "3D IQA CNN checkpoint not found. trying to download the checkpoint."
        )
        url = __iqa3d
        torch.hub.download_url_to_file(url, os.path.join(model_dir, model_name))
    return os.path.join(model_dir, model_name)


def iqa3d(stacks: List[Stack], batch_size=8, augmentation=True) -> List[float]:
    # torch -> numpy
    data: List[np.ndarray] = []
    for stack in stacks:
        d = stack.slices * stack.mask.float()
        d = d.squeeze(1).permute((2, 1, 0))
        idx = torch.nonzero(d > 0, as_tuple=True)
        x1, x2 = int(idx[0].min().item()), int(idx[0].max().item())
        y1, y2 = int(idx[1].min().item()), int(idx[1].max().item())
        z1, z2 = int(idx[2].min().item()), int(idx[2].max().item())
        d = d[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
        d = d[: INPUT_SHAPE[0], : INPUT_SHAPE[1], : INPUT_SHAPE[2]]
        d[d < 0] = 0
        d[d >= 10000] = 10000
        d = d / d.max()
        d = d[..., None]
        data.append(d.cpu().numpy())

    # load model
    model = model_architecture()
    model.compile()
    model.load_weights(get_iqa3d_checkpoint())

    # inference
    data_all = []
    if augmentation:
        flip_dims = [None, (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    else:
        flip_dims = [None]
    for _data in data:
        data_aug = []
        for flip_dim in flip_dims:
            d_aug = np.flip(_data, flip_dim) if flip_dim else _data
            pad = np.zeros([*INPUT_SHAPE, 1], dtype=np.float32)
            pad[: d_aug.shape[0], : d_aug.shape[1], : d_aug.shape[2]] = d_aug
            data_aug.append(pad)
        data_all.append(data_aug)
    stacked_data = np.array(data_all, dtype=np.float32).reshape(
        (len(data) * len(flip_dims), *INPUT_SHAPE, 1)
    )
    predict_all = model.predict(stacked_data, batch_size=batch_size).reshape(
        len(data), len(flip_dims)
    )
    predict_all = np.flip(np.sort(predict_all, -1), -1)
    scores = predict_all[:, :2].mean(axis=-1)
    return [float(score) for score in scores]
