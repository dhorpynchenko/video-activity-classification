from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

"""VIDEO"""
###################################################

IMAGE_DIR=("/home/adil_cp/Documents/projects/vision/TEST/")
def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(IMAGE_DIR+'outvid1.mp4', fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
file_names = next(os.walk(IMAGE_DIR))[2]
print((file_names))
file_names1=list()
for i in range(len(file_names)):
    file_names1.append(IMAGE_DIR+'frame%d.jpg' % i)
print(file_names1)
make_video(file_names1, outimg=None, fps=30, size=None,
               is_color=True, format="XVID")

#"""AUDIO"""
"""
###################################################
import subprocess
command = "/home/adil_cp/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1 -i /home/adil_cp/Desktop/3FUVKpQA6IY.mp4 output.mp3                  "
subprocess.call(command, shell=True)
###################################################"""
