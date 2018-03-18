# Running Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning

Run "Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning CVPR 2017"   [code](https://github.com/jiasenlu/AdaptiveAttention) in Docker.   

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
3. Build base docker 
    git clone https://github.com/floydhub/dl-docker.git  
    In Dockerfile.gpu, Modify -DBUILD_EXAMPLES=ON to OFF. [[The original problem ](https://github.com/floydhub/dl-docker/issues/77)]  
    Build the GPU base docker: docker build -t floydhub/dl-docker:gpu -f Dockerfile.gpu .
4. Build Mask RCNN docker  
sudo docker build -t zz:adaptive_attention .  
5. Run docker and test the method
(sudo) nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /sharedfolder:/root/sharedfolder zz:adaptive_attention /bin/bash  
jupyter notebook --allow-root & chosee demo.ipynb  
Change opt.checkpoint_path = 'save/coco_train.t7' to opt.checkpoint_path = 'save/model_id1_36.t7' becasue coco_train.t7 doesn't exist in their web server. I think the author updated their model, but not the demo.ipynb.  
Modify image_path = '/root/caffe/examples/images/cat.jpg' because this image doesn't exist. Here is an option /root/caffe/examples/images/cat.jpg

## Output Example
![Example](https://github.com/ZHAOZHIHAO/RunningProgramms/raw/master/running_knowing_when_to_look/imageCaption1.png)


