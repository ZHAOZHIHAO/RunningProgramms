# Running deep feature flow for video recognition

Run "deep feature flow for video recognition CVPR 2017"   [code](https://github.com/msracver/Deep-Feature-Flow) in Docker.   
The default setting is for detection, if you want to see result on segmentation  
1. Uncomment and comment some corrspoding lines in Dockerfile through the comments in that file.  
2. Use python rfcn/demo.py instead of python dff_rfcn/demo.py

## Steps

1. Install docker:    
curl -fsSL https://get.docker.com/ | sh  
2. Install nvidia-docker:  
Follow Quickstart part on https://github.com/NVIDIA/nvidia-docker   
 a. If you meet problem related to docker-ce version, just install the required the version. For example:  
             		sudo apt-get install docker-ce=17.12.0\~ce-0~ubuntu  
             b. If the test command "docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi" failed, see the error information.  
               If it's caused by CUDA version(at the very end of the error information), pull another correct nvidia/cuda version.  
               You can see all the tags of nvidia/cuda here https://hub.docker.com/r/nvidia/cuda/tags/. Then test it again.  
3. Build base mxnet docker  
  a. git clone https://github.com/apache/incubator-mxnet.git  
  b. cd $incubator-mxnet/docker  
  c. ./tool.sh build python gpu  
4. Build deep feature flow docker  
   Manually downdload the demo model to this repository, because I don't know how to download files on onedrive via wget. Model [here](https://onedrive.live.com/?authkey=%21AC4xhgrwHnIkH5Y&cid=F371D9563727B96F&id=F371D9563727B96F%21102799&parId=F371D9563727B96F%21102795&action=locate).   
I also modified config.mk to compile mxnet with CUDA and CuDNN, which is here and will be automatically read by Dockerfile.  
sudo docker build -t zz:dff .  
It takes a very long time because of compiling mxnet.  
5. To see images in docker  
   sudo apt-get install x11-xserver-utils  
    xhost + (this is a command)  
   In docker, to see images, use 'eog image_name'; to see videos, use 'vlc video_name'.  
6. Run docker  
sudo nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY zz:dff /bin/bash  
cd Deep-Feature-Flow  
python dff_rfcn/demo.py  
The input and result are both in demo directory.  

