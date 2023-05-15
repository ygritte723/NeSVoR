Quick start
===========

Fetal Brain Reconstruction
--------------------------

This example reconstruct a 3D fetal brain from mutiple stacks of 2D images in the following steps:

1. Segment fetal brain from each image using a CNN.
2. Correct bias field in each stack using the N4 algorithm.
3. Register slices using SVoRT.
4. Reconstruct a 3D volume using NeSVoR.

.. code-block:: console

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume volume.nii.gz \
        --output-resolution 0.8 \
        --registration svort \
        --svort-version v2\
        --segmentation \
        --bias-field-correction


Neonatal Brain Reconstruction
-----------------------------

This example reconstruct a 3D neonatal brain from mutiple stacks of 2D images in the following steps:

1. Removal background (air) with Otsu thresholding.
2. Correct bias field in each stack using the N4 algorithm.
3. Register slices using SVoRT.
4. Reconstruct a 3D volume using NeSVoR.

.. code-block:: console

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume volume.nii.gz \
        --output-resolution 0.8 \
        --registration svort \
        --svort-version v2\
        --otsu-thresholding \
        --bias-field-correction

Fetal Body/Uterus Reconstruction
--------------------------------

This is an example for deformable NeSVoR which consists of the following steps:

1. Create an ROI based on the intersection of all input stacks.
3. Perform stack-to-stack registration.
3. Reconstruct a 3D volume using Deformable NeSVoR (`--deformable <reconstruct.html#deformable>`_).

.. code-block:: console

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
