import numpy as np
import pyrealsense2 as rs
import cv2


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()


    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)

    try:
        while 1:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            im = frames.get_color_frame()
            depth_frame = np.asanyarray( depth.as_frame().get_data() )
            im_frame = np.asanyarray( im.as_frame().get_data() )

            color = cv2.applyColorMap(  cv2.convertScaleAbs(depth_frame,alpha=0.03) , cv2.COLORMAP_JET)

            cv2.imshow('depth_frame', color)
            cv2.imshow('color_frame', im_frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pipeline.stop()
