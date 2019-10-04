""" File used to define project settings."""
from os import path

# -------------------------------------------------------
# Project Root
# -------------------------------------------------------
ROOT_DIR = path.dirname(path.abspath(__file__))
# -------------------------------------------------------
# Tools
# -------------------------------------------------------
TOOLS_DIR = path.join(ROOT_DIR, 'tools')
CROPPED_IMAGE_GENERATOR_DIR = path.join(TOOLS_DIR, 'cropped_image_generator')

