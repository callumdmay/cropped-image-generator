import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import proj_settings
from PIL import Image

sys.path.append(os.path.join(proj_settings.CROPPED_IMAGE_GENERATOR_DIR, "models/research"))
from tools.cropped_image_generator.image_processor import ImageProcessor
from tools.cropped_image_generator.models.research.object_detection.utils import label_map_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(proj_settings.CROPPED_IMAGE_GENERATOR_DIR,
                              'models/research/object_detection/data/mscoco_label_map.pbtxt')
PATH_TO_GRAPH = os.path.join(proj_settings.CROPPED_IMAGE_GENERATOR_DIR, 'frozen_inference_graph.pb')
NUM_CLASSES = 90


class CroppedImageGenerator:
    def __init__(self):
        with tf.device('/gpu:0'):
            print("Using GPU")
        print("Initializing tensorflow graph...")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def process_video(self, video_frames, path):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_processor = ImageProcessor()
                print("Running image sequence object detection...")
                for index, image_np in enumerate(video_frames):
                    if index < 48:
                        continue
                    sys.stdout.write("\rProcessed: {0}".format(index))
                    sys.stdout.flush()
                    #Reverse RGB encoding
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB).astype('uint8')
                    image_pil = Image.fromarray(image_np)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores,
                                                              detection_classes, num_detections],
                                                             feed_dict={image_tensor: image_np_expanded})

                    # Processing result
                    image_processor.process_image(
                        image_pil,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        path
                        )
