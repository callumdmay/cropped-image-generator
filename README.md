# Image Sequence Generator

Python project for converting a video into a sequence of images filtered by classification. Simply add a trained model, input your classes and the video will be converted into a sequence of images filtered to ones that contain the specified classes

## Installation
Download Pipenv and Python3.7
 
 `pipenv install`


- Configure the 'cropped_image_sequence_generator' module.
	- Download the tensorflow models repo from https://github.com/tensorflow/models and extract it into a folder named models in the `cropped_image_generator` folder
	- Download whatever model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and extract the frozen_inference_graph.pb into the `cropped_image_generator` folder
	- Go to models/research and run protoc object_detection/protos/*.proto --python_out=.

## Quickstart
1. Place all .mp4 videos in a folder named `input_videos`
2. `pipenv shell`
3. `python exgen.py`
