import os
import cv2
import glob
import proj_settings
import shutil
from tools.video_utils.video_reader import VideoReader
from tools.cropped_image_generator.cropped_image_generator import CroppedImageGenerator

# Stop tensorflow from logging GPU info every time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# -------------------------------------------------------
# Settings
# -------------------------------------------------------


# Input video.
VIDEO_PATHS = glob.glob(proj_settings.ROOT_DIR + '/input_videos/*.mp4')
# Output dimensions.
OUTPUT_IMAGE_SHAPE = (224, 224)

OUTPUT_DIRECTORY = "output_images"
shutil.rmtree(OUTPUT_DIRECTORY, ignore_errors=True)

# -------------------------------------------------------
# Generate Cropped Image Sequences
# -------------------------------------------------------

sg = CroppedImageGenerator()
for path in VIDEO_PATHS:
    with VideoReader(path) as vid_reader:
        video_frames = vid_reader.get_frames()
        sg.process_video(video_frames, "{}/{}".format(OUTPUT_DIRECTORY, os.path.basename(path).split(".")[0]))