# Running Simple Online and Realtime Tracking with a Deep Association Metric

Run "Simple Online and Realtime Tracking with a Deep Association Metric"   [code](https://github.com/nwojke/deep_sort) in Docker.   

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
sudo docker pull tensorflow/tensorflow:1.5.0-gpu-py3  
4. Download data & model
  Download MOT16.zip [here](https://motchallenge.net/data/MOT16.zip).  
  Download deep_sort_data.zip [here](https://owncloud.uni-koblenz.de/owncloud/s/f9JB0Jr7f3zzqs8), including model and other data.
5. Build deep sort docker  
sudo docker build -t zz:deep_sort .  
6. Run docker  
sudo nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY zz:deep_sort /bin/bash  
7. Evaluate on MOT16
    cd deep_sort/  

    preparation  
    ```
    python tools/generate_detections.py \
        --model=resources/networks/mars-small128.pb \
        --mot_dir=./MOT16/train \
        --output_dir=./resources/detections/MOT16_train  
    ```
    association,  
        &ensp; &ensp; sudo apt-get install x11-xserver-utils  
        &ensp; &ensp; xhost + (type this command in host)  
        &ensp; &ensp; You also need  to repeat this command(simply replace the MOT16-01 at three places) for MOT16-01, MOT16-03, MOT16-06, MOT16-07, MOT16-08, MOT16-12, MOT16-14. You can see the output_file=MOT16-*.txt to know about association, with the format described in https://motchallenge.net/instructions/
    ```
    python deep_sort_app.py \
        --sequence_dir=./MOT16/test/MOT16-01 \
        --detection_file=./resources/detections/MOT16_POI_test/MOT16-01.npy \
        --min_confidence=0.3 \
        --nn_budget=100 \
        --display=True \
        --output_file=MOT16-01.txt
    ```
    evaluation, copy these MOT16-*.txt from docker to host, register in https://motchallenge.net/ and login to submit.
8. Help  
   i) To see images in docker:
   sudo apt-get install x11-xserver-utils  
   xhost + (type this command in host)  
   In docker, to see images, use 'eog image_name'; to see videos, use 'vlc video_name'.  
  
   ii)To copy files inside docker to host  
    (sudo) docker ps, to see the name of the running docker  
    (sudo) docker cp $running_docker_name:/path/to/file/inside/docker /host/path


