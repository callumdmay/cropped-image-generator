## Instructions

Download the tensorflow models repo from https://github.com/tensorflow/models
and extract it into a folder named `models`

Download whatever model from https://github.com/tensorflow/models/object_detection
and extract the `frozen_inference_graph.pb` into the `cropped_image_sequence_generator` folder

Run `pipenv --two shell` followed by `pipenv install`

Go to `models/research` and run `protoc object_detection/protos/*.proto --python_out=.`

Then to execute the image sequence generation run `python main.py` withing the pipenv shell
