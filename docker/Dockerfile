FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
ENV PIP_ROOT_USER_ACTION ignore
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3-venv python3-dev python3-distutils python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 2
RUN apt-get -y install build-essential git vim unzip wget
RUN python3 -m pip install cmake
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
COPY NeSVoR /home/NeSVoR
COPY install.sh /home/NeSVoR/
WORKDIR /home