import os
from .version import *

__base_dir = os.path.dirname(__file__)
__checkpoint_dir = os.path.join(__base_dir, "checkpoints")
__pretrained_svort = {
    "v1": "https://zenodo.org/record/7486938/files/checkpoint.pt?download=1",
    "v2": "https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1",
}
__monaifbs = "https://zenodo.org/record/4282679/files/models.tar.gz?download=1"
__iqa2d = "https://zenodo.org/record/7368570/files/pytorch.ckpt?download=1"
__iqa3d = "https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/weights_resnet_sw2_k3.hdf5"
