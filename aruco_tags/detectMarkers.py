import cv2
import numpy as np
import pyrealsense2 as rs



if __name__ == '__main__':
    #intel setup
    pipeline = rs.pipeline()
    config  = rs.config()
    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)

    #aruco setup
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    
    while 1:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        im = np.asanyarray(frames.get_color_frame().as_frame().get_data())

        #detect markers
        corners,idx,rejected = cv2.aruco.detectMarkers(im, arucoDict, parameters=arucoParams)
        im = cv2.aruco.drawDetectedMarkers(im, corners, idx)
        if idx is not None:
            print(idx.shape)

        cv2.imshow('im',im)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

