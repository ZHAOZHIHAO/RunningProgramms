FROM floydhub/dl-docker:gpu

MAINTAINER Zhihao Zhao <zhihao.zhao@ou.edu>

RUN apt-get -y update

# install torch-hdf5
RUN apt-get -y install libhdf5-serial-dev hdf5-tools
RUN git clone https://github.com/deepmind/torch-hdf5 && cd torch-hdf5 && luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"

# install lua-cjson
RUN luarocks install lua-cjson

# install helpful tools
RUN apt-get -y install eog vim tmux wget

# clone the repository and download pretrained modeli, Vocabulary File on COCO
RUN git clone https://github.com/jiasenlu/AdaptiveAttention.git && \
    cd AdaptiveAttention && \
    mkdir save && \
    cd save && \
    wget https://filebox.ece.vt.edu/%7Ejiasenlu/codeRelease/AdaptiveAttention/model/COCO/coco_train/model_id1_36.t7 && \
    wget https://filebox.ece.vt.edu/%7Ejiasenlu/codeRelease/AdaptiveAttention/model/COCO/coco_train/model_id1.json && \
    cd ../data && \
    wget https://filebox.ece.vt.edu/%7Ejiasenlu/codeRelease/AdaptiveAttention/data/COCO/cocotalk_challenge_vocab.json && \
    wget https://filebox.ece.vt.edu/%7Ejiasenlu/codeRelease/AdaptiveAttention/data/COCO/cocotalk_vocab.json

# install the specific cudnn v4 version according to the code
COPY cudnn-7.0-linux-x64-v4.0-prod.tgz ./ 
RUN tar -xvf cudnn-7.0-linux-x64-v4.0-prod.tgz
RUN cd cuda && cp -P include/cudnn.h /usr/include && cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/ && chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
RUN echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
