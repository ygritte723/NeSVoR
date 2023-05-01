# NeSVoR: **Ne**ural **S**lice-to-**Vo**lume **R**econstruction

NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction (both rigid and deformable).

This package is the accumulation of the following works:

\[1\] SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1) | [Arxiv](https://arxiv.org/abs/2206.10802))

\[2\] NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI ([IEEE](https://ieeexplore.ieee.org/document/10015091)|[TechRxiv](https://www.techrxiv.org/articles/preprint/NeSVoR_Implicit_Neural_Representation_for_Slice-to-Volume_Reconstruction_in_MRI/21398868/1))

<p align="center">
   <img src="./images/recon.gif" align="center" width="600">
</p>
<p align="center"><p align="center">

<!-- toc -->
<!--
  - [Slice-to-Volume Registration Transformers (SVoRT)](#slice-to-volume-registration-transformers-svort)
  - [Neural Slice-to-Volume Reconstruction (NeSVoR)](#neural-slice-to-volume-reconstruction-nesvor) -->
- [Overview](#overview)
- [Installation](#installation)
  - [Docker Image](#docker-image)
  - [From Source](#from-source)
- [Quick Start](#quick-start)
  - [Fetal Brain Reconstruction](#fetal-brain-reconstruction)
  - [Fetal Body/Uterus Reconstruction](#fetal-bodyuterus-reconstruction)
- [Usage](#usage)
  - [Reconstruction ](#reconstruction)
  - [Registration (Motion Correction)](#registration-motion-correction)
  - [Sampling](#sampling)
    - [Sample Volume](#sample-volume)
    - [Sample Slices](#sample-slices)
  - [Preprocessing](#preprocessing)
    - [Brain Masking](#brain-masking)
    - [Bias Field Correction](#bias-field-correction) <!-- - [Resources](#resources) -->
    - [Stack Quality Assessment](#stack-quality-assessment)
  - [3D Brain Segmentation](#3d-brain-segmentation)
- [Cite Our Work](#cite-our-work)
- [Resources](#resources)

<!-- tocstop -->

## Overview

NeSVoR is a deep learning package for solving slice-to-volume reconstruction problems (i.e., reconstructing a 3D isotropic high-resolution volume from a set of motion-corrupted low-resolution slices) with application to fetal/neonatal MRI, which provides
- Motion correction by mapping 2D slices to a 3D canonical space using [Slice-to-Volume Registration Transformers (SVoRT)](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1).
- Volumetric reconstruction of multiple 2D slices using implicit neural representation ([NeSVoR](https://www.techrxiv.org/articles/preprint/NeSVoR_Implicit_Neural_Representation_for_Slice-to-Volume_Reconstruction_in_MRI/21398868/1)).

<!--### Slice-to-Volume Registration Transformers (SVoRT)-->

<p align="center">
   <img src="./images/SVoRT_network.png" align="center" width="600">
</p>
<p align="center">Figure 1. SVoRT: an iterative Transformer for slice-to-volume registration. (a) The k-th iteration of SVoRT. (b) The detailed network architecture of the SVT module.<p align="center">

<!--### Neural Slice-to-Volume Reconstruction (NeSVoR)-->

<p align="center">
   <img src="./images/NeSVoR.png" align="center" width="900">
</p>
<p align="center">Figure 2. NeSVoR: A) The forward imaging model in NeSVoR. B) The architecture of the implicit neural network in NeSVoR.<p align="center">

## Installation

### Docker Image

We recommend to use our docker image to run `nesvor`.

#### Install Docker and NVIDIA Container Toolkit

You may follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install Docker and NVIDIA Container Toolkit

#### Download and Run NeSVoR Image

```
docker pull junshenxu/nesvor:v0.3.0
docker run  --gpus '"device=0"' -it junshenxu/nesvor:v0.3.0
nesvor -h
```
Note: our latest image was built with CUDA 11.7.

### From Source

<details>
  <summary>Show details</summary>
  
#### Prerequisites

If you are installing from source, you will need:
- An NVIDIA GPU
- Python 3.8 or later;
- GCC/G++ 7.5 or higher;
- CUDA v10.2 or higher;
- CMake v3.21 or higher.

see [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for more on prerequisites

#### Get the NeSVoR Source

Since the master branch might be messy sometimes, it is recommanded to clone a specific release.

```
git clone https://github.com/daviddmc/NeSVoR --branch v0.3.0
cd NeSVoR
```

#### Install Dependencies

```
pip install -r requirements.txt
```
Install PyTorch extension of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). Make sure the installed CUDA version mismatches the version that was used to compile PyTorch. Then, run the following command (see [this](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) for more details)
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

#### Install NeSVoR.

```
pip install -e .
```
</details>

## Quick Start

### Fetal Brain Reconstruction

This example reconstruct a 3D fetal brain from mutiple stacks of 2D images in the following steps:

1. Segment fetal brain from each image using a CNN.
2. Correct bias field in each stack using the N4 algorithm.
3. Register slices using SVoRT.
4. Reconstruct a 3D volume using NeSVoR.

```
nesvor reconstruct \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--thicknesses <thick-1> ... <thick-N> \
--output-volume volume.nii.gz \
--output-resolution 0.8 \
--registration svort \
--svort-version v2\
--segmentation \
--bias-field-correction
```

### Fetal Body/Uterus Reconstruction

This is an example for deformable NeSVoR which consists of the following steps:

1. Create an ROI based on the intersection of all input stacks.
3. Perform stack-to-stack registration.
3. Reconstruct a 3D volume using Deformable NeSVoR (`--deformable`).

```
nesvor reconstruct \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--thicknesses <thick-1> ... <thick-N> \
--output-volume volume.nii.gz \
--output-resolution 1.0 \
--stacks-intersection \
--registration stack \
--deformable \
--weight-transformation 1 \
--weight-deform 0.1 \
--weight-image 0.1 \
--single-precision \
--log2-hashmap-size 22 \
--batch-size 8192
```

## Usage

NeSVoR currently supports the following commands.

[`nesvor reconstruct`](#reconstruction): reconstruct a 3D volume (i.e., train a NeSVoR model) from either multiple stacks of slices (NIfTI) or a set of motion-corrected slices (the output of `register`). It can also perform multiple preprocessing steps, including brain segmentation, bias field correction, and registration, by setting the corrsponding flags.

[`nesvor register`](#registration-motion-correction): register stacks of slices using a pretrained SVoRT model or stack-to-stack registration.

[`nesvor sample-volume`](#sample-volume): sample a volume from a trained NeSVoR model.

[`nesvor sample-slices`](#sample-slices): simulate slices from a trained NeSVoR model.

[`nesvor segment-stack`](#brain-masking): 2D fetal brain segmentation/masking in input stacks.

[`nesvor correct-bias-field`](#bias-field-correction): bias field correction using the N4 algorihtm.

[`nesvor assess`](#stack-quality-assessment): quality and motion assessment of input stacks.

[`nesvor segment-volume`](#3d-brain-segmentation): 3D fetal brain segmentation in reconstructed volumes.

run `nesvor <command> -h` for a full list of parameters of a command.

### Reconstruction

#### Reconstruct from Mutiple Stacks of Slices

The `reconstruct` command can be used to reconstruct a 3D volume from `N` stacks of 2D slices (in NIfTI format, i.e. `.nii` or `.nii.gz`). 

A basic usage of `reconstruct` is as follows, where `mask-i.nii.gz` is the ROI mask of the i-th input stack.
```
nesvor reconstruct \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--stack-masks mask-1.nii.gz ... mask-N.nii.gz \
--output-volume volume.nii.gz
```

A more elaborate example could be 
```
nesvor reconstruct \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--stack-masks mask-1.nii.gz ... mask-N.nii.gz \
--thicknesses <thick-1> ... <thick-N> \
--output-volume volume.nii.gz \
--output-resolution 0.8 \
--output-model model.pt \
--weight-image 1.0 \
--image-regularization edge \
--delta 0.2 \
--n-iter 5000 \
--level-scale 1.38 \
--coarsest-resolution 16.0 \
--finest-resolution 0.5 \
--n-levels-bias 0 \
--n-samples 128
```
Run `nesvor reconstruct --h` to see the meaning of each parameter.

Given multiple stacks at inputs, `reconstruct` first corrects the motion in the input stacks using SVoRT (the same as what `register` did), and then trains a NeSVoR model that implicitly represent the underlying 3D volume, from which a discretized volume (i.e., a 3D tensor) can be sampled.

#### Reconstruct from Motion-Corrected Slices

`reconstruct` can also take a folder of motion-corrected slices as inputs. 
```
nesvor reconstruct \
--input-slices <path-to-slices-folder> \
--output-volume volume.nii.gz
```
This enables the separation of registration and reconstruction. That is, you may first run `register` to perform motion correction, and then use `reconstruct` to reconstruct a volume from a set of motion-corrected slices.

#### Deformable NeSVoR

NeSVoR can now reconstruct data with deformable (non-rigid) motion! To enable deformable motion, use the flag `--deformable`. 
```
nesvor reconstruct \
--deformable \
......
```
This feature is still experimental.

### Registration (Motion Correction)
`register` takes mutiple stacks of slices as inputs, performs motion correction, and then saves the motion-corrected slices to a folder.
```
nesvor register \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--stack-masks mask-1.nii.gz ... mask-N.nii.gz \
--registration <method> \
--output-slices <path-to-save-output-slices>
```
`register` currently supports the following methods:
- `svort`: the full SVoRT model;
- `svort-stack`: only apply stack transformations of SVoRT;
- `stack`: stack-to-stack rigid registration;
- `none`: no registration.

Note: SVoRT can only be used for registering fetal brain images withou backgound.

### Sampling

#### Sample Volume
Upon training a NeSVoR model with the `reconstruct` command, you can sample a volume at arbitrary resolutions with the `sample-volume` command.
```
nesvor sample-volume \
--output-volume volume.nii.gz \
--input-model model.pt \
--output-resolution 0.5
```

#### Sample Slices
You may sample slices from the model using the `sample-slices` command. For each slice in `<path-to-slices-folder>`, the command simulates a slice from the NeSVoR model at the corresponding slice location.
```
nesvor sample-slices \
--input-slices <path-to-slices-folder> \
--input-model model.pt \
--simulated-slices <path-to-save-simulated-slices>
```
For example, after running `reconstruct`, you can use `sample-slices` to simulate slices at the motion-corrected locations and evaluate the reconstruction results by comparing the input slices and the simulated slices. 

### Preprocessing

#### Brain Masking
We integrate a deep learning based fetal brain segmentation model ([MONAIfbs](https://github.com/gift-surg/MONAIfbs)) into our pipeline to extract the fetal brain ROI from each input image. Check out their [repo](https://github.com/gift-surg/MONAIfbs) and [paper](https://arxiv.org/abs/2103.13314) for details. The `segment-stack` command generates brain mask for each input stack as follows.
```
nesvor segment-stack \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--output-stack-masks mask-1.nii.gz ... mask-N.nii.gz \
```
You may also perform brain segmentation in the `reconstruct` command by setting `--segmentation`.

#### Bias Field Correction
We also provide a wrapper of [the N4 algorithm in SimpleITK](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html) for bias field correction. The `correct-bias-field` command correct the bias field in each input stack and output the corrected stacks.
```
nesvor correct-bias-field \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--stack-masks mask-1.nii.gz ... mask-N.nii.gz \
--output-corrected-stacks corrected-stack-1.nii.gz ... corrected-stack-N.nii.gz
```
You may perform bias field correction in the `reconstruct` command by setting `--bias-field-correction`

#### Stack Quality Assessment
The `assess` command evalutes the image quality / motion of input stacks. This information can be used to find a template stack with the best quality or filter out low-quality data. 
An example is as follows.
```
nesvor assess \
--input-stacks stack-1.nii.gz ... stack-N.nii.gz \
--stack-masks mask-1.nii.gz ... mask-N.nii.gz \
--metric <metric> \
--output-json result.json 
```

The provided metrics are:

- `ncc`: cross correlaiton between adjacent slices
- `matrix-rank`: A motion metric based on the rank of the data matrix as described in [(Kainz, B., et al. 2015)](https://ieeexplore.ieee.org/document/7064742)
- `volume`: volume of the masked ROI
- `iqa2d`: image quality score generated by a [2D CNN](https://github.com/daviddmc/fetal-IQA) (only for fetal brain)
- `iqa3d`: image quality score generated by a [3D CNN](https://github.com/FNNDSC/pl-fetal-brain-assessment) (only for fetal brain)

### 3D Brain Segmentation
The coherent 3D volume generated by our pipeline can be used for downstream analysis, for example, segmentation or parcellation of 3D brain volume. The `segment-volume` command provides a wrapper of the TWAI segmentation algorithm for T2w fetal brain MRI. You may find more detials of this method in the authors' [repo](https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation). To use this tool, you need to clone their repo and update the path in `config.py` (see the comment in `config.py` for details). An exmaple of `segment-volume` is as follows:
```
nesvor segment-volume
--input-volume reconstructed-volume.nii.gz \
--output-folder <path-to-save-segmentation>
```

<!-- ## Resources -->

## Cite Our Work

SVoRT
```
@inproceedings{xu2022svort,
  title={SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI},
  author={Xu, Junshen and Moyer, Daniel and Grant, P Ellen and Golland, Polina and Iglesias, Juan Eugenio and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={3--13},
  year={2022},
  organization={Springer}
}
```

NeSVoR
```
@article{10015091,
  author={Xu, Junshen and Moyer, Daniel and Gagoski, Borjan and Iglesias, Juan Eugenio and Ellen Grant, P. and Golland, Polina and Adalsteinsson, Elfar},
  journal={IEEE Transactions on Medical Imaging}, 
  title={NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3236216}
}
```

Fetal IQA
```
@inproceedings{xu2020semi,
  title={Semi-supervised learning for fetal brain MRI quality assessment with ROI consistency},
  author={Xu, Junshen and Lala, Sayeri and Gagoski, Borjan and Abaci Turk, Esra and Grant, P Ellen and Golland, Polina and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={386--395},
  year={2020},
  organization={Springer}
}
```

## Resources

This project has been greatly inspired by the following list of fantastic works.

- [gift-surg/NiftyMIC](https://github.com/gift-surg/NiftyMIC)
- [bkainz/fetalReconstruction](https://github.com/bkainz/fetalReconstruction)
- [SVRTK](https://github.com/SVRTK/SVRTK)
- [daviddmc/fetal-IQA](https://github.com/daviddmc/fetal-IQA)
- [FNNDSC/pl-fetal-brain-assessment](https://github.com/FNNDSC/pl-fetal-brain-assessment)
- [LucasFidon/trustworthy-ai-fetal-brain-segmentation](https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation)
- [gift-surg/MONAIfbs](https://github.com/gift-surg/MONAIfbs)