import os
import proj_settings

class ImageProcessor:
    min_score_thresh = 0.8
    index = 0

    def __init__(self):
        self.image_sequence_list = []

    def process_image(self, image, boxes, classes, scores, category_index, path):
        # Group any boxes that correspond to the same location.
        filtered_boxes = []

        for i in range(boxes.shape[0]):
            if scores[i] > ImageProcessor.min_score_thresh:
                box = {
                    "ymin": boxes[i][0],
                    "xmin": boxes[i][1],
                    "ymax": boxes[i][2],
                    "xmax": boxes[i][3],
                }
                if classes[i] in category_index.keys() and category_index[classes[i]]["name"] == "person":
                    filtered_boxes.append(box)

        for box in filtered_boxes:
            box_scaling_constant = 0.02
            (im_width, im_height) = image.size
            (left, right, top, bottom) = (max(0, box["xmin"] - box_scaling_constant) * im_width,
                                            min(1, box["xmax"] + box_scaling_constant) * im_width,
                                            min(0, box["ymin"] - box_scaling_constant) * im_height,
                                            max(1, box["ymax"] + box_scaling_constant) * im_height)
            temp_img = image.crop((left, top, right, bottom))

            save_path = os.path.join(proj_settings.ROOT_DIR, 'output')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            temp_img.save(save_path + "/{}.jpg".format(ImageProcessor.index))
            ImageProcessor.index += 1