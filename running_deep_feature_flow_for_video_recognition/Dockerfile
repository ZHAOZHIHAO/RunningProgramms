FROM mxnet/python:gpu

MAINTAINER Zhihao Zhao <zhihao.zhao@ou.edu>

# install necessary libraries
RUN apt-get -y update
RUN apt-get -y install git wget unzip eog vlc python-pip python-dev python-tk libopenblas-dev cmake liblapacke-dev
RUN pip install Cython opencv-python==3.2.0.6 easydict==1.6 pyyaml Pillow matplotlib
RUN sed -i 's/geteuid/getppid/' /usr/bin/vlc

# clone Deep-Feature-Flow
RUN git clone https://github.com/msracver/Deep-Feature-Flow.git
RUN cd Deep-Feature-Flow && chmod +x init.sh && ./init.sh

# clone mxnet
#RUN git clone --recursive https://github.com/dmlc/mxnet.git && cd mxnet && git checkout 62ecb60 && git submodule update
# copy files for segementation, can't do both at once
#RUN cp -r /Deep-Feature-Flow.git/rfcn/operator_cxx/* /mxnet/src/operator/contrib/
# copy files for detection 
RUN cp -r /Deep-Feature-Flow/dff_rfcn/operator_cxx/* /mxnet/src/operator/contrib/

# copy another config.mk I modified it just to support CUDA and CuDNN
COPY ./config.mk /mxnet/make
# compile mxnet
RUN cd mxnet && make clean && make -j8
RUN cd mxnet/python && python setup.py install

# place the trained model
COPY ./demo_model.zip /Deep-Feature-Flow/model 
RUN cd /Deep-Feature-Flow/model && unzip demo_model.zip && cp demo_model/* ./ && rm ./demo_model.zip  && rm ./demo_model -rf
