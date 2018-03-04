FROM tensorflow/tensorflow:1.4.0-rc0-gpu-py3

MAINTAINER Zhihao Zhao <zhihao.zhao@ou.edu>

# install necessary libraries
RUN pip install keras
RUN apt-get -y update
RUN apt-get -y install git wget unzip eog vlc vim cmake 
RUN apt-get -y install firefox tmux
RUN pip install cython scikit-image
RUN sed -i 's/geteuid/getppid/' /usr/bin/vlc

# change default python from python2 to python3
RUN cd /usr/bin && rm ./python && ln -s python3 python

# install pycocotools
RUN git clone https://github.com/pdollar/coco.git && cd coco/PythonAPI && make -j12 && make install -j12 && python setup.py install

# clone MASK RCNN
RUN git clone https://github.com/matterport/Mask_RCNN.git


