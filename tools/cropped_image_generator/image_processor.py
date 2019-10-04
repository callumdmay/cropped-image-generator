import os
import proj_settings

class ImageProcessor:
    min_score_thresh = 0.8

    def __init__(self):
        self.image_sequence_list = []
        self.index = 0

    def process_image(self, image, boxes, classes, scores, category_index, path):
        # Group any boxes that correspond to the same location.
        filtered_boxes = []
        
        found_person = False
        for i in range(boxes.shape[0]):
            if scores[i] > ImageProcessor.min_score_thresh:
                if classes[i] in category_index.keys() and category_index[classes[i]]["name"] == "person":
                    box = {
                        "ymin": boxes[i][0],
                        "xmin": boxes[i][1],
                        "ymax": boxes[i][2],
                        "xmax": boxes[i][3],
                    }
                    filtered_boxes.append(box)
                    found_person = True

        for box in filtered_boxes:
            save_path = os.path.join(proj_settings.ROOT_DIR, 'output')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image.save(save_path + "/{}.jpg".format(self.index))
            self.index += 1

        return found_person