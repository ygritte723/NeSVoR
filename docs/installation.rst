Installation
============

Docker image
------------

We recommend to use our docker image to run ``nesvor``.

Install docker and NVIDIA container toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may follow this `guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ to install docker and NVIDIA container toolkit.

Download NeSVoR image
^^^^^^^^^^^^^^^^^^^^^

Pull the latest NeSVoR image from docker hub.

.. code-block:: console

    docker pull junshenxu/nesvor

**Note**: our latest image was built with CUDA 11.7.

Run NeSVoR with docker
^^^^^^^^^^^^^^^^^^^^^^

You may run a container in an iterative way.

.. code-block:: console

    docker run -it --gpus all junshenxu/nesvor
    nesvor -h

You may also run the ``nesvor`` command directly as follows.

.. code-block:: console

    docker run --rm --gpus all \
        -v <path-to-inputs>:/incoming:ro -v <path-to-outputs>:/outgoing:rw \
        junshenxu/nesvor \
        nesvor reconstruct \
        --input-stacks /incoming/stack-1.nii.gz ... /incoming/stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume /outgoing/volume.nii.gz

From Source
------------

Prerequisites
^^^^^^^^^^^^^

If you are installing from source, you will need:

.. raw:: html

    <ul>
        <li>An NVIDIA GPU;</li>
        <li>Python 3.8 or later;</li>
        <li>GCC/G++ 7.5 or higher;</li>
        <li>CUDA v10.2 or higher;</li>
        <li>CMake v3.21 or higher.</li>
    </ul>

See `tiny-cuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_ for more about prerequisites.

Get the NeSVoR source
^^^^^^^^^^^^^^^^^^^^^

Since the master branch might be messy sometimes, it is recommanded to clone a specific release.

.. code-block:: console

    git clone https://github.com/daviddmc/NeSVoR --branch <tag>
    cd NeSVoR


Install dependencies
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    pip install -r requirements.txt

Install PyTorch extension of `tiny-cuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_. 
Make sure the installed CUDA version mismatches the version that was used to compile PyTorch. 
Then, run the following command (see `this <https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension>`_ for more details)

.. code-block:: console

    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

Other dependencies
^^^^^^^^^^^^^^^^^^

Some functionalities of NeSVoR require extra dependencies.

3D IQA
++++++

The 3D MR quality assessment tool uses a pretrained Tensorflow model.
Tensorflow 2 needs to be installed to enable this tool.
Check out the `orginal repo <https://github.com/FNNDSC/pl-fetal-brain-assessment>`_ for more details.

TWAI Segmentation
+++++++++++++++++

The toolkit provides a wrapper of the TWAI segmentation algorithm for T2w fetal brain MRI. 
You may find more detials of this method in the authors' `repo <https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation>`_. 
To use this tool, you need to clone their repo and update the path in ``config.py`` (see the comment in ``config.py`` for details). 

Install NeSVoR
^^^^^^^^^^^^^^

The last step is to install NeSVoR itself.

.. code-block:: console

    pip install -e .
