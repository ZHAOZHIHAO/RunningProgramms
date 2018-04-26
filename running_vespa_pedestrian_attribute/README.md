# Running Deep View-Sensitive Pedestrian Attribute Inference in an end-to-end Model (vespa)

Run "Deep View-Sensitive Pedestrian Attribute Inference in an end-to-end Model BMVC 2017"   [code](https://github.com/asc-kit/vespa) in Docker.   

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
3. Pull base  docker  
sudo docker pull nvidia/cuda:8.0-cudnn5-devel  
4. Build vespa docker  
sudo docker build -t zz:vespa .  
5. Run docker  
sudo nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY zz:vespa /bin/bash  
6. Test your own images  
  i. put all your images in "images" directory and rebuild vespa docker and run it  
  ii. cd /vespa/utils && python inference_zz.py  
  iii. results(attributes confidences) are stored in image along with the original image in /vespa/utils  directory, named as out*.jpg  
7. Evaluate on PETA dataset  
   i. [Download](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html]), name it as PETA.zip and put it at this directory. To make sure, you should see "PETA dataset" and "Readme.txt" in this zip file.  
   ii. uncomment lines containing PETA.zip in Dockerfile  
   iii. cd /vespa/utils && python inference_zz.py  
   iv. result is stored as text in /vespa/eval_peta/metrics.txt  
8. Help  
   i) To see images in docker:
   sudo apt-get install x11-xserver-utils  
   xhost + (this is a command)  
   In docker, to see images, use 'eog image_name'; to see videos, use 'vlc video_name'.  
  
   ii)To copy files inside docker to host  
    (sudo) docker ps, to see the name of the running docker  
    (sudo) docker cp $running_docker_name:/path/to/file/inside/docker /host/path  

## Output Example
![Example](https://github.com/ZHAOZHIHAO/RunningProgramms/raw/master/running_vespa_pedestrian_attribute/out2.jpg)

