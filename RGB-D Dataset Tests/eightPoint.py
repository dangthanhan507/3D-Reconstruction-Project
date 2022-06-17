import cv2
import numpy as np;

from skimage.measure import ransac
from skimage.transform import AffineTransform

def featureMatching1(im1,im2):
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    
    #ransac
    model, inliers = ransac( (src_pts, dst_pts), 
                            AffineTransform, min_samples=4, residual_threshold=4, max_trials=40)
    
    
    
    ## beginning of unnecessary
    n_inliers = np.sum(inliers)
    inlier_kp_left = [ cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers] ]
    inlier_kp_right = [ cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers] ]
    placehold_matches = [cv2.DMatch(idx,idx,1) for idx in range(n_inliers)]
    
    plt.figure(figsize=(15,15))
    final_img = cv2.drawMatches( im1, inlier_kp_left, im2, inlier_kp_right, placehold_matches, None )
    plt.imshow(final_img)
    plt.show()
    ## end of unnecessary
    
    
    return src_pts[inliers], dst_pts[inliers]

def featureMatching(im1,im2):
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    
    
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    
    #ransac
    model, inliers = ransac( (src_pts, dst_pts), 
                            AffineTransform, min_samples=4, residual_threshold=8, max_trials=40)
    
    
    return src_pts[inliers], dst_pts[inliers]

def eight_point(src_pts, dst_pts):
    assert src_pts.shape[0] == 2
    assert dst_pts.shape[0] == 2
    assert src_pts.shape[1] == dst_pts.shape[1]
    assert src_pts.shape[1] >= 8
    
    ones = np.ones(shape=src_pts.shape[1])
    
    #projective coords
    start = np.vstack((src_pts,ones))
    end = np.vstack((dst_pts,ones))
    
    end_x = start[0,:] * end
    end_y = start[1,:] * end
    end_1 = start[2,:] * end
    
    A = np.vstack((end_x,end_y,end_1)).T # create homogenous matrix
    
    ATA = A.T@A
    
    U,S,V = np.linalg.svd(A)
    
    #e = V[:,np.argmin(S)]
    e = V[np.argmin(S),:]
    
    F = e.reshape(3,3)
    
    return F

def point8_triangulate(ptsL, ptsR, K, R, t):
    
    qL = ptsL.copy()
    qR = ptsR.copy()
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    c = np.array([[cx,cy]]).T
    
    qL = qL - c
    qL[0,:] = qL[0,:] / fx
    qL[1,:] = qL[1,:] / fy
    
    qR = qR - c
    qR[0,:] = qR[0,:] / fx
    qR[1,:] = qR[1,:] / fy
    
    qL = np.vstack( (qL,np.ones(qL.shape[1])) )
    qR = np.vstack( (qR,np.ones(qR.shape[1])) )
    
    Rlql = qL
    Rrqr = (R @ qR) * -1
    b = t    
    
    zhatL = np.zeros(shape=(1,ptsL.shape[1]))
    zhatR = np.zeros(shape=(1,ptsR.shape[1]))
    for i in range(ptsL.shape[1]):
        R_lidx = Rlql[:,i].reshape(3,1)
        R_ridx = Rrqr[:,i].reshape(3,1)
        A = np.hstack((R_lidx,R_ridx))
        
        zhat = np.linalg.lstsq(A,b,rcond=None)[0]
        zhatL[:,i] = zhat[0]
        zhatR[:,i] = zhat[1]
    PL = zhatL * qL
    PR = zhatR * qR
    
    P1 = PL
    P2 = R @ PR + t
    pts3 = (P1+P2)/2
    return pts3
def decomposeE(F, src_pts, dst_pts, K):
    E = K.T @ F @ K # fundamental
    
    U,S,VT = np.linalg.svd(E)
    s = (S[0] + S[1])/2
    S = np.array([[s,0,0], [0,s,0], [0,0,0]])
    E = U @ S @ VT
    
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    U,S,VT = np.linalg.svd(E)
    
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    R3 = U @ W @ VT
    R4 = U @ W.T @ VT
    
    C1 = U[:,2].reshape(3,1)
    C2 = -U[:,2].reshape(3,1)
    C3 = U[:,2].reshape(3,1)
    C4 = -U[:,2].reshape(3,1)
    
    #check rotation determinants
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4
    pts3_1 = point8_triangulate(src_pts, dst_pts, K, R1.T, -R1.T@C1)
    pts3_2 = point8_triangulate(src_pts, dst_pts, K, R2.T, -R2.T@C2)
    pts3_3 = point8_triangulate(src_pts, dst_pts, K, R3.T, -R3.T@C3)
    pts3_4 = point8_triangulate(src_pts, dst_pts, K, R4.T, -R4.T@C4)
    
    z_1 = R1[2,:]@(pts3_1 - C1)
    z_2 = R2[2,:]@(pts3_2 - C2)
    z_3 = R3[2,:]@(pts3_3 - C3)
    z_4 = R4[2,:]@(pts3_4 - C4)
    
    s1 = (z_1 > 0).sum()
    s2 = (z_2 > 0).sum()
    s3 = (z_3 > 0).sum()
    s4 = (z_4 > 0).sum()
    
    max_sum = max([s1,s2,s3,s4])
    if max_sum == s1:
        return R1.T, -R1.T@C1
    elif max_sum==s2:
        return R2.T, -R2.T@C2
    elif max_sum==s3:
        return R3.T, -R3.T@C3
    else:
        return R4.T, -R4.T@C4


def motion8(im0, im1, K):
    src_pts, dst_pts = featureMatching( np.uint8(im0*255),np.uint8(im1*255) )
    F = eight_point(src_pts.T, dst_pts.T)
    R,t = decomposeE(F, src_pts.T, dst_pts.T, K)
    return R,t
