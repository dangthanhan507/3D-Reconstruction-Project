import cv2
import time
if __name__ == '__main__':
    filename = 'video.avi'
    frames_per_second = 20.0
    W = 480
    H = 1280

    vidType = cv2.VideoWriter_fourcc(*'MJPG')
    cam = cv2.VideoCapture(2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    writer = cv2.VideoWriter(filename, vidType, 25.0 , (1280,480))
    try:
        while 1:
            ret,frame = cam.read()
            if ret:
                writer.write(frame)
                cv2.imshow('frame',frame)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cam.release()
        writer.release()
    cv2.destroyAllWindows()
