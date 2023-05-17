Examples
========

Reconstruction
--------------

Reconstruct from mutiple stacks of slices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``reconstruct`` command can be used to reconstruct a 3D volume from ``N`` stacks of 2D slices (in NIfTI format, i.e. ``.nii`` or ``.nii.gz``). 
A basic usage of ``reconstruct`` is as follows, where ``mask-i.nii.gz`` is the ROI mask of the i-th input stack.

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --output-volume volume.nii.gz

A more elaborate example could be 

.. code-block:: nesvorcommand

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

Run ```nesvor reconstruct --h``` to see the meaning of each parameter.

Given multiple stacks at inputs, 
``reconstruct`` first corrects the motion in the input stacks using SVoRT (the same as what ``register`` did), 
and then trains a NeSVoR model that implicitly represent the underlying 3D volume, 
from which a discretized volume (i.e., a 3D tensor) can be sampled.

Reconstruct from motion-corrected slices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``reconstruct`` can also take a folder of motion-corrected slices as inputs. 

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-slices <path-to-slices-folder> \
        --output-volume volume.nii.gz

This enables the separation of registration and reconstruction. 
That is, you may first run ``register`` to perform motion correction, 
and then use ``reconstruct`` to reconstruct a volume from a set of motion-corrected slices.

Deformable NeSVoR
^^^^^^^^^^^^^^^^^

NeSVoR can now reconstruct data with deformable (non-rigid) motion! To enable deformable motion, use the flag ``--deformable``. 

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --deformable \
        .....

This feature is still experimental.

Registration (motion correction)
--------------------------------

``register`` takes mutiple stacks of slices as inputs, performs motion correction, 
and then saves the motion-corrected slices to a folder.

.. code-block:: nesvorcommand

    nesvor register \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --registration <method> \
    --output-slices <path-to-save-output-slices>

The list of supported methods in ``register`` is `here <register.html#registration>`__.

Sampling
--------

Sample volume
^^^^^^^^^^^^^

Upon training a NeSVoR model with the ``reconstruct`` command, 
you can sample a volume at arbitrary resolutions with the ``sample-volume`` command.

.. code-block:: nesvorcommand

    nesvor sample-volume \
    --output-volume volume.nii.gz \
    --input-model model.pt \
    --output-resolution 0.5

Sample slices
^^^^^^^^^^^^^

You may sample slices from the model using the ``sample-slices`` command. 
For each slice in ``<path-to-slices-folder>``, 
the command simulates a slice from the NeSVoR model at the corresponding slice location.

.. code-block:: nesvorcommand

    nesvor sample-slices \
        --input-slices <path-to-slices-folder> \
        --input-model model.pt \
        --simulated-slices <path-to-save-simulated-slices>

For example, after running ``reconstruct``, you can use ``sample-slices`` to simulate slices at the motion-corrected locations and evaluate the reconstruction results by comparing the input slices and the simulated slices. 

Preprocessing
-------------

Brain masking
^^^^^^^^^^^^^

We integrate a deep learning based fetal brain segmentation model (`MONAIfbs <https://github.com/gift-surg/MONAIfbs>`_) 
into our pipeline to extract the fetal brain ROI from each input image.
The ``segment-stack`` command generates brain mask for each input stack as follows.

.. code-block:: nesvorcommand

    nesvor segment-stack \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --output-stack-masks mask-1.nii.gz ... mask-N.nii.gz \

You may also perform brain segmentation in the ``reconstruct`` command by setting ``--segmentation``.

Bias field correction
^^^^^^^^^^^^^^^^^^^^^

We also provide a wrapper of `the N4 algorithm in SimpleITK <https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html>`_ for bias field correction. 
The ``correct-bias-field`` command correct the bias field in each input stack and output the corrected stacks.

.. code-block:: nesvorcommand

    nesvor correct-bias-field \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --output-corrected-stacks corrected-stack-1.nii.gz ... corrected-stack-N.nii.gz

You may perform bias field correction in the ``reconstruct`` command by setting ``--bias-field-correction``.

Stack quality assessment
^^^^^^^^^^^^^^^^^^^^^^^^

The ``assess`` command evalutes the image quality / motion of input stacks. 
This information can be used to find a template stack with the best quality or filter out low-quality data. 
An example is as follows.


.. code-block:: nesvorcommand

    nesvor assess \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --metric <metric> \
        --output-json result.json 

The provided metrics are listed `here <assess.html#metric>`__.


3D brain segmentation
---------------------

The coherent 3D volume generated by our pipeline can be used for downstream analysis, 
for example, segmentation or parcellation of 3D brain volume. 
The ``segment-volume`` command provides a wrapper of the `TWAI segmentation algorithm <https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation>`_ for T2w fetal brain MRI. 
An exmaple of ``segment-volume`` is as follows:

.. code-block:: nesvorcommand

    nesvor segment-volume
        --input-volume reconstructed-volume.nii.gz \
        --output-folder <path-to-save-segmentation>
