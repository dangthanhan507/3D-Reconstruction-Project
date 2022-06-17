import cv2
import os
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)


idx = 0
while 1:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    im = np.asanyarray(frames.get_color_frame().as_frame().get_data())

    cv2.imshow('frame', im)
    key = cv2.waitKey(1)
    if key == ord('c'):
        cv2.imwrite(f'calibration_im/im{idx}.jpg', im)
        idx+=1
    elif key == ord('q'):
        break
cv2.destroyAllWindows()
