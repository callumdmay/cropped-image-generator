# Hockey Event Annotator

## Installation

 - Add the project directory to your $PYTHONPATH environment variable.
 - Adjust file-paths as necessary in the "proj_settings.py" configuration file.
 - Install all required libraries:
   - Anaconda (Python 3.5)
   - TensorFlow
   - OpenCV

- Configure the 'cropped_image_sequence_generator' module.
	- Download the tensorflow models repo from https://github.com/tensorflow/models and extract it into a folder named models in the `cropped_image_sequence_generator` folder
	- Download whatever model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and extract the frozen_inference_graph.pb into the `cropped_image_sequence_generator` folder
	- Go to models/research and run protoc object_detection/protos/*.proto --python_out=.

## Quickstart
After following the installation instructions, example usage is provided in the "examples" directory.

## Project Structure
- Data
	- Used to store any external data necessary to run the program.
  - Contains all training examples
- Examples
	- Contains many sample scripts of different actions.
- Model Training
	- Stores any previously trained model logs, as well as the scripts used to train our models.
- Tools
	- Contains all modules belonging to the project.
