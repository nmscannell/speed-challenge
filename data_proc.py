import cv2
import numpy as np
import skvideo.io


reader = skvideo.io.FFmpegReader('files/test.mp4')
fc = 0
for f in reader.nextFrame():
    cv2.imwrite('files/test/frame%d.jpg' % fc, f)
    fc += 1
