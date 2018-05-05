FROM tensorflow/tensorflow:1.5.0-gpu-py3

MAINTAINER Zhihao Zhao <zhihao.zhao@ou.edu>

# install necessary libraries
RUN apt-get -y update
RUN apt-get -y install vim unzip
RUN apt-get -y install git

RUN apt-get -y install libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

WORKDIR /
RUN git clone https://github.com/nwojke/deep_sort.git
# MOT16.zip https://motchallenge.net/data/MOT16.zip
COPY MOT16.zip /deep_sort
# deep_sort_data.zip https://owncloud.uni-koblenz.de/owncloud/s/f9JB0Jr7f3zzqs8
COPY deep_sort_data.zip /deep_sort 
RUN cd /deep_sort && unzip -d ./MOT16 MOT16.zip
RUN cd /deep_sort && unzip deep_sort_data.zip
RUN mv /deep_sort/deep_sort_data/resources /deep_sort

