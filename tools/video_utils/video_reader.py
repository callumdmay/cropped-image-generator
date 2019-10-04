import os
import cv2

class VideoReader(object):

    def __init__(self, filename):
        self.vid_capture = cv2.VideoCapture(filename)

    def get_frames(self, show_frames=False):
        while self.vid_capture.isOpened():
            ret, frame = self.vid_capture.read()
            if frame is None:
                break
            if show_frames is True:
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            yield frame

    def save_frames(self, out_folder_path):
        out_folder_path = os.path.abspath(out_folder_path)
        for i, frame in enumerate(self.get_frames()):
            if frame is None:
                break

            file_name = "{}.jpg".format(str(i).zfill(8))
            out_path = os.path.join(out_folder_path, file_name)
            cv2.imwrite(out_path, frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
