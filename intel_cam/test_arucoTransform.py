import numpy as np
import pyrealsense2 as rs
import cv2

def triangulate(depth_map, fx = 5, fy = 5):
    h,w = depth_map.shape
    xx,yy = np.meshgrid(np.arange(w), np.arange(h))
    mask = (depth_map > 0)
    
    proj_x = (xx[mask] - w/2) * depth_map[mask] / fx
    proj_y = (yy[mask] - h/2) * depth_map[mask] / fy
    
    return np.vstack((proj_x.flatten(), proj_y.flatten(), depth_map[mask].flatten())), mask

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

class arucoTest:
    def __init__(self, frame, intr, dist, markerLength=0.0246153846):
        self.arucoPoseDict = {}

        self.arucoDict = cv2.aruco.Dictionary_get( cv2.aruco.DICT_6X6_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.markerLength = markerLength

        corners, idx, _ = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerLength, intr, dist)

        self.K = intr
        self.D = dist

        for i in range( idx.shape[0] ):
            index = idx[i,0]
            R,_ = cv2.Rodrigues(rvecs[i])
            t = tvecs[i].T
            self.arucoPoseDict[index] = (R,t)
    def getPose(self, frame):
        corners, idx, _ = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.K, self.D)


        avgR = np.zeros((3,3))
        avgT = np.zeros((3,1))
        count = 0

        for i in range(idx.shape[0]):
            index = idx[i,0]
            R1,_ = cv2.Rodrigues(rvecs[i])
            t1 = tvecs[i].T

            if index in self.arucoPoseDict.keys():
                R,t = self.arucoPoseDict[index]
                poseR = R @ R1.T
                poseT = t - (R @ R1.T) @ t1

                avgR += poseR
                avgT += poseT
                count += 1
        if count != 0:
            avgR /= count
            avgT /= count
        return avgR, avgT


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

    aruco_test = None
    pts3_total = []
    colormap_total = []
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
            if aruco_test is None:
                aruco_test = arucoTest(im_frame, intr, dist, 0.04445 / 0.0002500000118743628) # init

            pts3, mask = triangulate(depth_frame, fx, fy)
            R,t = aruco_test.getPose(im_frame)

            colormap = im_frame[:,:,::-1][mask].reshape(-1,3)

            pts3_total.append(R @ pts3 + t)
            colormap_total.append(colormap)

    pts3 = pts3_total[0]
    colormap = colormap_total[0]
    for i in range(1,len(pts3_total)):
        pts3 = np.hstack((pts3,pts3_total[i]))
        colormap = np.vstack((colormap, colormap_total[i]))
    create_output(pts3.T, colormap, 'AfterTest.ply')
