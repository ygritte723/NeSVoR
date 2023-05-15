Quick start
===========

This toolkit is optimized for slice-to-volume reconstruction in fetal MRI, including reconstruction of 
fetal/neonatal brain volume (i.e., a relatively small ROI that can be consider rigid) 
or fetal body/uterus (i.e., a larger ROI that may undergo deformable motion).

Fetal Brain Reconstruction
--------------------------

NeSVoR is able to reconstruct a 3D fetal brain from mutiple stacks of 2D images in the following steps 
(This is also the most common use case of the toolkit):

#. Segment fetal brain from each image using a CNN (``--segmentation``  `ref <reconstruct.html#segmentation>`__ ).
#. Correct bias field in each stack using the N4 algorithm (``--bias-field-correction``  `ref <reconstruct.html#bias-field-correction>`__ ).
#. Register slices using SVoRT (it is the default of ``--registration``  `ref <reconstruct.html#registration>`__ ).
#. Reconstruct a 3D volume using NeSVoR.

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume volume.nii.gz \
        --output-resolution 0.8 \
        --segmentation \
        --bias-field-correction

Neonatal Brain Reconstruction
-----------------------------

The reconstuction of neonatal brain is similar. 
The only difference is that, in postnatal MRI, 
the brain is surround by air instead of maternal tissues. 
Therefore, the CNN segmenter is replaced by the Otsu method.

The following example reconstructs a 3D neonatal brain in the following steps:

#. Removal background (air) with Otsu thresholding (``--otsu-thresholding``  `ref <reconstruct.html#otsu-thresholding>`__).
#. Correct bias field in each stack using the N4 algorithm (``--bias-field-correction``  `ref <reconstruct.html#bias-field-correction>`__).
#. Register slices using SVoRT.
#. Reconstruct a 3D volume using NeSVoR.

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume volume.nii.gz \
        --output-resolution 0.8 \
        --otsu-thresholding \
        --bias-field-correction

Fetal Body/Uterus Reconstruction
--------------------------------

The NeSVoR toolkit is also capable of reconstructing a larger volumetric ROI that may undergo deformable motion, 
e.g., fetal body and uterus. 

.. warning::
    The deformable NeSVoR method is experimental and might be unstable!

This is an example for deformable NeSVoR which consists of the following steps:

#. Create an ROI based on the intersection of all input stacks (``--stacks-intersection``  `ref <reconstruct.html#stacks-intersection>`__).
#. Perform stack-to-stack registration (``--registration stack``  `ref <reconstruct.html#registration>`__).
#. Reconstruct a 3D volume using Deformable NeSVoR (``--deformable``  `ref <reconstruct.html#deformable>`__).

.. code-block:: nesvorcommand

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

See more
--------

:doc:`Docs of the commands <cmd>`