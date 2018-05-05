# Running Mask RCNN

Run "Mask RCNN ICCV 2017"   [code](https://github.com/matterport/Mask_RCNN.git) in Docker.   

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
3. Pull base tensorflow docker 
    When pull the docker image, be careful on the CUDA version. Most tensorflow images are built with CUDA>=9.0. Since I have CUDA8.0, I pull the following image:  
sudo docker pull tensorflow/tensorflow:1.4.0-rc0-gpu-py3  
For other tensorflow docker images, see "Which containers exist?" section on [tensorflow page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker).
4. Build Mask RCNN docker  
sudo docker build -t zz:mask_rcnn .  
5. To see images in docker  & To copy files inside docker to host  
   i) To see images in docker:
   sudo apt-get install x11-xserver-utils  
   xhost + (type this command in host)  
   In docker, to see images, use 'eog image_name'; to see videos, use 'vlc video_name'.  
  
   ii)To copy files inside docker to host  
    (sudo) docker ps, to see the name of the running docker  
    (sudo) docker cp $running_docker_name:/path/to/file/inside/docker /host/path
6. Run docker  
 nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /sharedfolder:/root/sharedfolder zz:mask_rcnn /bin/bash  
 cd Mask_RCNN  
 jupyter notebook --allow-root & copy jupyter's link to your local host firefox/chrome/etc & run ./samples/demo.ipynb  
To test your own image, add *image = skimage.io.imread(your_image_path)* in *Run Object Detection* section.  
[Deprecated]
~~sudo nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY zz:mask_rcnn /bin/bash
*cd Mask_RCNN* (You are at the default *notebooks* directory when you enter docker)  
*tmux* & Open another window ([A Quick and Easy Guide to tmux](http://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/))  
*jupyter notebook --allow-root* in one window & *friefox* in another window, then you are able to run jupyter in firefox inside docker.  
To see result on your default image, change the variable *image* in *demo.ipynb Run Object Detection part*.~~

