import numpy as np
import pyrealsense2 as rs
import cv2
from posePnP import calcPose


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def create_output(vertices, colors, filename):
	vertices = np.hstack([vertices,colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def triangulate(depth_map, fx = 5, fy = 5):
    h,w = depth_map.shape
    xx,yy = np.meshgrid(np.arange(w), np.arange(h))
    mask = (depth_map > 0)
    
    proj_x = (xx[mask] - w/2) * depth_map[mask] / fx
    proj_y = (yy[mask] - h/2) * depth_map[mask] / fy
    
    return np.vstack((proj_x.flatten(), proj_y.flatten(), depth_map[mask].flatten())), mask

if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()


    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(depth_scale)
    intr = np.load('../calibration/intrinsics.npy')
    fx = intr[0,0]
    fy = intr[1,1]

    #get initial frame
    frames = align.process( pipeline.wait_for_frames() )
    prev_frame = np.asanyarray(frames.get_color_frame().as_frame().get_data())
    R = np.eye(3)
    t = np.array([[0,0,0]]).T
    try:
        while 1:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            im = frames.get_color_frame()
            depth_frame = np.asanyarray( depth.as_frame().get_data() )
            im_frame = np.asanyarray( im.as_frame().get_data() )

            color = cv2.applyColorMap(  cv2.convertScaleAbs(depth_frame,alpha=0.03) , cv2.COLORMAP_JET)
            stitch = np.hstack((color,im_frame))
            cv2.imshow('color_frame', stitch)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('c'):
                pts3, mask = triangulate(depth_frame*depth_scale, fx,fy)
                color_map = im_frame[mask].reshape(-1,3)
                create_output(pts3.T, color_map, 'test.ply')
    except KeyboardInterrupt:
        pipeline.stop()
