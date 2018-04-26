FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Zhihao Zhao <zhihao.zhao@ou.edu>

# install necessary libraries
RUN apt-get -y update
RUN apt-get -y install git wget unzip eog python-pip python-dev python-tk libopenblas-dev cmake liblapacke-dev
RUN pip install Cython opencv-python==3.2.0.6 easydict==1.6 pyyaml Pillow matplotlib
RUN apt-get -y install vim

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get install -y libatlas-base-dev 
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
RUN apt-get install -y libopencv-dev
RUN pip install scikit-image protobuf scipy

# build caffe with custom python layer
RUN git clone https://github.com/BVLC/caffe.git
RUN git clone https://github.com/asc-kit/vespa.git
ENV PYTHONPATH $PYTHONPATH:/vespa/layers
RUN cd caffe && mkdir build && cd build && cmake .. 
RUN cd caffe/build && make all -j12
RUN cd caffe/build && make test -j12
  # it's weird that runtest would fail outside docker, but succeed inside docker.
#RUN cd caffe/build && make runtest -j12
RUN cd caffe/build && WITH_PYTHON_LAYER=1 make all pycaffe -j12
ENV PYTHONPATH $PYTHONPATH:/caffe/python

# download pretrained model
RUN cd /vespa && wget https://cvhci.anthropomatik.kit.edu/%7Easchuman/vespa/vespa-peta_iter_12000.caffemodel

#COPY ./PETA.zip /vespa
COPY ./evaluation_zz.py /vespa/utils
#RUN cd /vespa && mkdir PETA_dataset && unzip PETA.zip -d PETA_dataset && cd PETA_dataset && mv PETA\ dataset/ PETA_dataset
COPY ./inference_zz.py /vespa/utils
COPY ./images /vespa/utils/images
