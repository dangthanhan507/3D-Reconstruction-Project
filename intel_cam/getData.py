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

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    with open('../calibration/intrinsics.npy', 'rb') as f:
        intr = np.load(f)
        dist = np.load(f)

    fx = intr[0,0]
    fy = intr[1,1]

    frames = pipeline.wait_for_frames()

    images = []
    depths = []
    while 1:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        im = frames.get_color_frame()
        depth_frame = np.asanyarray( depth.as_frame().get_data() )
        im_frame = np.asanyarray( im.as_frame().get_data() )

        
        color = cv2.applyColorMap( cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        stitch = np.hstack((color,im_frame))
        cv2.imshow('color_frame', stitch)
        key = cv2.waitKey(1)

        if key == ord('q'):
           break 
        elif key == ord('c'):
            images.append(im_frame)
            depths.append(depth_frame)
    with open('testdata.npy', 'wb') as f:
        for i in range(len(images)):
            np.save(f, images[i])
            np.save(f, depths[i])
    print('There are these many images: ', len(images))
    print('Done')
