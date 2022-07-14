import cv2
import os

 
videoFile = 'F:/ygszfx/lanxiumei/lxm/20220606-lxm-whq-1-qudou.mp4'
cap = cv2.VideoCapture(videoFile)
frameNum = 0
savepath='F:/ygszfx/lanxiumei/lxm/qudou/'
folder=os.path.exists(savepath)
if not folder:
    os.makedirs(savepath)
while (cap.isOpened()):
    ret, frame = cap.read()
    frameNum = frameNum + 1
    if frameNum % 10 == 0:  # 调整帧数
        if ret:
            new_img = cv2.resize(frame,None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(savepath+str(frameNum//10) + ".jpg", new_img)  # 保存图片
        # cv2.namedWindow("resized", 0)  # 0可以改变窗口大小了
        # # cv2.resizeWindow("resized", 640, 480) # 设置固定大小不能该，上面和一起使用
        # cv2.imshow("resized", frame)  # 显示视频
        else:
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release
cv2.destroyAllWindows()

