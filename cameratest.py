import cv2
def get_img_from_camera_net():
    cap = cv2.VideoCapture("rtsp://admin:mkls1123@192.168.0.64/")#获取网络摄像机
    
    i = 1
    ret = True
    while ret:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        print (str(i))
        cv2.imshow(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
 
# 测试
if __name__ == '__main__':

    get_img_from_camera_net()
