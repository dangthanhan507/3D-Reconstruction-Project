import cv2
import numpy 


if __name__ == '__main__':

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    '''
    for i in [30,31,32, 33, 34 ,35]:
        im = cv2.aruco.drawMarker(arucoDict, i, 200, 1)
        cv2.imshow('im',im)
        cv2.waitKey(0)
        cv2.imwrite(f'aruco_tag{i}.png', im)
    '''

    grid_board = cv2.aruco.GridBoard_create(4,3, 0.04, 0.01, arucoDict, firstMarker=36)
    im_b = grid_board.draw((1920,1080))
    
    cv2.imwrite('aruco_board.png', im_b)

    cv2.imshow('im',im_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #corners, ids, rejected = cv2.aruco.detectMarkers(im, arucoDict, parameters=arucoParams)

