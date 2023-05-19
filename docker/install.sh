#
# torch
# python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# cu117 is the default
python3 -m pip install torch==1.13.1 torchvision==0.14.1 
# tinycudann 
python3 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
# tensorflow
# python3 -m pip install tensorflow==2.8.* 
# other deps
python3 -m pip install -r requirements.txt 
python3 -m pip install "SimpleITK>=2.2.1" 
python3 -m pip install "monai==0.3.0" 
# install nesvor
python3 -m pip install -e . 
# TWAI
# cd /home; git clone https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation.git 
# install nnunet
# python3 -m pip install gdown matplotlib dicom2nifti medpy==0.4.0 batchgenerators==0.21.0 pandas scikit-learn 
# mkdir -p /home/trustworthy-ai-fetal-brain-segmentation/docker/third-party/nnUNet_preprocessed 
# mkdir -p /home/trustworthy-ai-fetal-brain-segmentation/docker/third-party/nnUNet_raw_data_base 
# mkdir -p /home/trustworthy-ai-fetal-brain-segmentation/docker/third-party/nnUNet_trained_models/nnUNet/3d_fullres/ 
# cd /home/trustworthy-ai-fetal-brain-segmentation/docker/third-party/nnUNet_trained_models/nnUNet/3d_fullres/ 
# gdown https://drive.google.com/uc\?export\=download\&id\=1DC5qVOrarjkt8NhSIUFazKgqgLVcuWco 
# unzip Task225_FetalBrain3dTrust.zip && rm Task225_FetalBrain3dTrust.zip 
# cd /home/trustworthy-ai-fetal-brain-segmentation/docker/third-party/nnUNet 
# python3 -m pip install -e . --no-deps 
# fix libpng
# cd /home 
# wget https://sourceforge.net/projects/libpng/files/libpng16/1.6.37/libpng-1.6.37.tar.gz 
# tar xvf libpng-1.6.37.tar.gz && cd libpng-1.6.37/  
# ./configure; make; make install 
# cd /home; rm ./libpng-1.6.37.tar.gz; rm -r ./libpng-1.6.37 
# cd /home/NeSVoR; rm ./install.sh 
