import cv2
import numpy as np


def adjust_brightness(image, factor, slice):
    # Convert to hue, saturation, value model
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, slice] = hsv[:, :, slice] * factor
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def crop_sky_and_dashboard(frame):
    frame = frame[190:360, :-90]
    image = cv2.resize(frame, (220, 66), interpolation=cv2.INTER_AREA)
    return image


def optical_flow(frame1, frame2):
    flow = np.zeros_like(frame1)
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow_data = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 1, 15, 2, 5, 1.3, 0)
    #convert data to hsv
    mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
    flow[...,1] = 255
    flow[...,0] = ang*180/np.pi/2
    flow[...,2] = (mag *15).astype(int)
    return flow


def preprocess_image_from_path(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = adjust_brightness(img, 0.8, 2)
    img = crop_sky_and_dashboard(img)
    return img


#for i in range(20400):
#    img = preprocess_image_from_path('files/train/frame%d.jpg' % i)
#    cv2.imwrite('files/train/crop/frame%d.jpg' % i, img)


#for i in range(10798):
#    img = preprocess_image_from_path('files/test/frame%d.jpg' % i)
#    cv2.imwrite('files/test/crop/frame%d.jpg' % i, img)
