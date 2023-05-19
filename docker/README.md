Here is how to build a docker image for NeSVoR (only with SVoRT model)

``
# pull source
rm -rf ./NeSVoR
git clone https://github.com/daviddmc/NeSVoR.git
# download SVoRT model
mkdir ./NeSVoR/nesvor/checkpoints
wget -O ./NeSVoR/nesvor/checkpoints/SVoRT_v2.pt https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1
# build docker
docker build --force-rm -t junshenxu/nesvor .
# run docker and install
docker run --name nesvor_container --gpus all -it junshenxu/nesvor /bin/bash
cd NeSVoR
sh install.sh
exit
# commit
docker commit nesvor_container junshenxu/nesvor
``
