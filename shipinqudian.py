import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.optimize import curve_fit
import random
import math
import sys

font = {'family' : 'MicroSoft YaHei',
        'weight' : 'bold',
        'size'   : 30}

def func1(x, x0,t,b,c):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:t, 
                                   lambda x:np.exp(-b*x0)*c-np.exp(-b*x)*c+t])
def ransac1(x,y,rate,Iter_Number,DisT,ProT,csz):
    chongcaiyang=0
    N=len(x)
    best_inliers_number =0
    i=0
    p0=50,csz,0.1,1
    while True:
        x_2c= random.sample(list(x), int(N*rate))
        y_2c=y[x_2c]
        popt, pcov = curve_fit(func1, x_2c, y_2c,p0,maxfev=500000)
        X_inliers = []
        Y_inliers = []
        inliers_current = 0 #当前模型的内点数量
        i+=1
        for j in range(N):
            x_current = x[j]
            y_current = y[j]
            dis_current = np.abs(y_current-func1(x_current,*popt))
            if (dis_current <= DisT):
                inliers_current += 1
                X_inliers.append(x_current)
                Y_inliers.append(y_current)
            #print("当前内点数量={}, 最佳内点数量={},最佳内点比例={}".format(inliers_current, best_inliers_number,best_inliers_number / N))
        if (inliers_current > best_inliers_number):
            i=0
            Pro_current = inliers_current / N       #当前模型的内点比例Pro_current
            best_inliers_number = inliers_current   #更新最优内点的数量
            best_x0,best_t,best_b,best_c = popt  #更新模型参数
            print("更新结果：x0={}, t={}, b={}, c={},当前内点比例={}, 最佳内点比例={}"
                .format(best_x0,best_t,best_b,best_c, Pro_current, best_inliers_number/N))
        if ((best_inliers_number / N) > ProT):
            print("更新结果：终止：内点比例=", best_x0,best_t,best_b,best_c,(best_inliers_number / N), "大于期望内点比例=", ProT)
            chongcaiyang=1
            break
        if i>Iter_Number:
            print("i>Iter_Number")
            x0_qz=math.ceil(best_x0)
            y1=y[:x0_qz+1]
            y2=y[x0_qz+1:int(0.45*N)]
            y=np.append(y1,y2[y2>0])
            N=len(y)
            x=np.arange(N)
        if i>2*Iter_Number:
            chongcaiyang=0
            break
    return best_x0,best_t,best_b,best_c,chongcaiyang,best_inliers_number / N

def OnMouseAction(event, x, y, flags, param):
    global frame, position,gs,position_line,position_line1,position_line2,position_line3,d,jxk
    if event == cv2.EVENT_RBUTTONDOWN:
        position_line = (x,y)
        cv2.circle(frame, position_line, 2, (0,0,0), -1)
    # elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
    #     cv2.line(frame, position_line, (x,y), (0,0,0), 2)
    elif event == cv2.EVENT_RBUTTONUP:
        position_line1 = (x,y)
        d=math.sqrt((position_line[0]-position_line1[0])**2+(position_line[1]-position_line1[1])**2)
        cv2.line(frame, position_line, position_line1, (0,0,0), 2)
        cv2.putText(frame,f"{chizi}mm" , (position_line[0]+2,position_line[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    elif event == cv2.EVENT_MBUTTONDOWN:
        position_line2 = (x,y)
        cv2.circle(frame, position_line2, 2, (255,0,0), -1)
    elif event == cv2.EVENT_MBUTTONUP:
        z=1
        position_line3 = (x,y)
        jxk=np.arange(position_line3[1],position_line2[1],-5*(d/int(chizi)))
        jxk=[int(jxk[i]) for i in range(len(jxk))]
        for j,i in enumerate(jxk):
            cv2.rectangle(frame, position_line2, (position_line3[0],i), (255,0,0), 1)
        # d1=math.sqrt((position_line2[0]-position_line3[0])**2+(position_line2[1]-position_line3[1])**2)
        # cv2.line(frame, position_line2, position_line3, (255,0,0), 2)
            cv2.putText(frame,f"{j*5}mm" , (position_line3[0]+2,i+2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        cv2.putText(frame,f"{round((int(chizi)/d)*(position_line3[1]-position_line2[1]),1)}mm" , (position_line3[0]-1,position_line2[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        x_pinjun=(position_line3[0]+position_line2[0])/2
        jxk_pinjun=[jxk[i]-int(2.5*(d/int(chizi))) for i in range(len(jxk))]
        print(x_pinjun)
        for wz in jxk_pinjun:
            hl=int(wz-2)
            hh=int(wz+3)
            wl=int(x_pinjun-2)
            wh=int(x_pinjun+3)
            print(wl,wh)
            point=np.zeros(shape=(len(dirs)))
            p=0
            for i in range(len(dirs)):
                for j in range(hl,hh):
                    for h in range(wl,wh):
                        if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                            p+=list1[j,h,1,i]
                p1=p/25
                point[i]=p1
                p=0
            print('2')
            
            y_med = signal.medfilt(point, kernel_size=13)
            y_med=np.array(y_med)
            y_med=y_med.flatten()
            #y_med=y_med[y_med>0]
            if y_med[y_med>0].size == 0:
                #y_med=np.zeros((len(dirs)))
                x=np.arange(len(dirs))
                plt.rc("font", **font)
                plt.figure(figsize=(40,20), dpi=80)
                plt.plot(x, y_med, '-p', color='grey',
                    marker = 'o',
                    markersize=5, linewidth=8,
                    markerfacecolor='red',
                    markeredgecolor='red',
                    markeredgewidth=2)
                plt.xlabel('帧数', fontsize=35)
                plt.ylabel('荧光强度', fontsize=35)
                plt.title(f'第{gs}个点的ICG曲线', fontsize=50)
                cv2.circle(frame, center=position, radius=2,
                  color=(255, 0, 0), thickness=-1)
                cv2.putText(frame,f"{gs}" , (position[0]+1,position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")
                plt.show()
                z=0
            if z:
                #y_med=np.insert(y_med,0,0)
                csz=np.mean(y_med[0:10])
                x=np.arange(len(y_med))
                x0,t,b,c,ccy1,ratio1=ransac1(x,y_med,0.8,1000,3,0.9,csz)
                ratio2=0
                ratio3=0
                if ccy1==0:
                    print("chongcaiyang!!!!")
                    hl=int(wz-2)
                    hh=int(wz+3)
                    wl=int((x_pinjun+position_line2[0])/2-2)
                    wh=int((x_pinjun+position_line2[0])/2+3)
                    print(position_line2[0])
                    print(wl,wh)
                    point=np.zeros(shape=(len(dirs)))
                    p=0
                    for i in range(len(dirs)):
                        for j in range(hl,hh):
                            for h in range(wl,wh):
                                if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                    p+=list1[j,h,1,i]
                        p1=p/25
                        point[i]=p1
                        p=0
                    print('2')
                    y_med = signal.medfilt(point, kernel_size=13)
                    y_med=np.array(y_med)
                    y_med=y_med.flatten()
                    csz=np.mean(y_med[0:10])
                    x=np.arange(len(y_med))
                    x01,t1,b1,c1,ccy2,ratio2=ransac1(x,y_med,0.8,1000,3,0.9,csz)
                if ccy2==0:
                    print("Again chongcaiyang!!!!")
                    hl=int(wz-2)
                    hh=int(wz+3)
                    wl=int((x_pinjun+position_line3[0])/2-2)
                    wh=int((x_pinjun+position_line3[0])/2+3)
                    print(position_line2[0])
                    print(wl,wh)
                    point=np.zeros(shape=(len(dirs)))
                    p=0
                    for i in range(len(dirs)):
                        for j in range(hl,hh):
                            for h in range(wl,wh):
                                if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                                    p+=list1[j,h,1,i]
                        p1=p/25
                        point[i]=p1
                        p=0
                    print('2')
                    y_med = signal.medfilt(point, kernel_size=13)
                    y_med=np.array(y_med)
                    y_med=y_med.flatten()
                    csz=np.mean(y_med[0:10])
                    x=np.arange(len(y_med))
                    x02,t2,b2,c2,ccy3,ratio3=ransac1(x,y_med,0.8,1000,3,0.9,csz)
                if ratio3 > ratio2:
                    x01,t1,b1,c1=x02,t2,b2,c2
                    print("again success:chongcaiyang!!!!")
                if ratio2> ratio1:
                    x0,t,b,c=x01,t1,b1,c1
                    print("success:chongcaiyang!!!!")

                #y_med1 = preprocess1.fit_transform(y_med)
                #y_med2 = preprocess1.fit_transform(y_med)
                tmax=(np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                #thalfmax=(np.log(0.5*(np.exp(-b*x0)+t/c))/-b)
                print(position_line3[0]+2,wz)
                score=int(100-2*(tmax-x0))
                if score <=0:
                    score=0
                #cv2.putText(frame,f'Tmax:{round((tmax-x0)/3,1)}s,Fmax:{round(0.95*(np.exp(-b*x0)*c+t),1)}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.putText(frame,f'Score:{score}' , (position_line3[0]+30,wz), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                #plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")


    elif event == cv2.EVENT_LBUTTONDOWN:
        gs+=1
        z=1                                          
        position = (x,y)
        print(x,y)
        if position != None:
            hl=position[1]-2
            hh=position[1]+3
            wl=position[0]-2
            wh=position[0]+3
            print("1")
            point=np.zeros(shape=(len(dirs)))
            p=0
            for i in range(len(dirs)):
                for j in range(hl,hh):
                    for h in range(wl,wh):
                        if list1[j,h,0,i]>=35 and list1[j,h,0,i]<=77:
                            p+=list1[j,h,1,i]
                p1=p/25
                point[i]=p1
                p=0
            print('2')
            
            y_med = signal.medfilt(point, kernel_size=13)
            y_med=np.array(y_med)
            y_med=y_med.flatten()
            #y_med=y_med[y_med>0]
            if y_med[y_med>0].size == 0:
                #y_med=np.zeros((len(dirs)))
                x=np.arange(len(dirs))
                plt.rc("font", **font)
                plt.figure(figsize=(40,20), dpi=80)
                plt.plot(x, y_med, '-p', color='grey',
                    marker = 'o',
                    markersize=5, linewidth=8,
                    markerfacecolor='red',
                    markeredgecolor='red',
                    markeredgewidth=2)
                plt.xlabel('帧数', fontsize=35)
                plt.ylabel('荧光强度', fontsize=35)
                plt.title(f'第{gs}个点的ICG曲线', fontsize=50)
                cv2.circle(frame, center=position, radius=2,
                  color=(255, 0, 0), thickness=-1)
                cv2.putText(frame,f"{gs}" , (position[0]+1,position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")
                plt.show()
                z=0
            if z:
                #y_med=np.insert(y_med,0,0)
                csz=np.mean(y_med[0:10])
                x=np.arange(len(y_med))
                x0,t,b,c,ccy3,ratio3=ransac1(x,y_med,0.8,1000,3,0.9,csz)
                #y_med1 = preprocess1.fit_transform(y_med)
                #y_med2 = preprocess1.fit_transform(y_med)
                tmax=round(((np.log(0.01*(np.exp(-b*x0)+t/c))/-b)-x0)/3,2)
                t12max=round(((np.log(0.5*(np.exp(-b*x0)+t/c))/-b)-x0)/3,2)
                fmax=round(0.99*(np.exp(-b*x0)*c+t),2)
                slope=round((fmax-t)/tmax,2)
                tr=round(t12max/tmax,3)
                print('3')
                plt.rc("font", **font)
                plt.figure(figsize=(40,20), dpi=80)
                plt.scatter(x,y_med)
                plt.plot(x, func1(x,x0,t,b,c), '-p', color='red',
                    marker = 'o',
                    markersize=3, linewidth=5,
                    markerfacecolor='red',
                    markeredgecolor='red',
                    markeredgewidth=2)
                plt.xlabel('帧数', fontsize=35)
                plt.ylabel('荧光强度', fontsize=35)
                plt.title(f'第{gs}个点的ICG曲线', fontsize=50)
                plt.text(x=0.5*len(x),#文本x轴坐标 
                    y=0.7*(np.exp(-b*x0)*c+t), #文本y轴坐标
                    s=f'Tmax:{tmax}s,Fmax:{fmax},T1/2max:{t12max}s\nSlope:{slope},TR:{tr}', #文本内容
                    fontdict=dict(fontsize=30, color='r',family='MicroSoft YaHei',),#字体属性字典
                    #添加文字背景色
                    bbox={'facecolor': '#74C476', #填充色
                        'edgecolor':'b',#外框色
                        'alpha': 0.5, #框透明度
                        'pad': 8,#本文与框周围距离 
                        }
                    )
                plt.axhline(fmax)
                plt.axvline(x=x0)
                plt.axvline(x=np.log(0.01*(np.exp(-b*x0)+t/c))/-b)
                cv2.circle(frame, center=position, radius=2,
                  color=(255, 0, 0), thickness=-1)
                cv2.putText(frame,f"{gs}" , (position[0]+1,position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                plt.savefig(f"F:/ygszfx/mfx/2whq{gs}.jpg")
                plt.show()
                


 
    # elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:      #按住左键拖曳不放开
    #     position2 = (x,y)
        
    # elif event == cv2.EVENT_LBUTTONUP:                                          #放开左键
    #     position2 = (x,y)  



if __name__ == '__main__':
    #chizi = sys.argv[1]
    chizi=100
    gs=0
    dirs=os.listdir("F:/ygszfx/lanxiumei/lxm/whq")
    dirs.sort(key=lambda x: int(x.split('.')[0]))
    frame=cv2.imread("F:/ygszfx/lanxiumei/lxm/whq/1.jpg")
    h,w,c=frame.shape
    list1=np.zeros(shape=(h,w,c,len(dirs)))
    for i in range(len(dirs)):
        path=os.path.join("F:/ygszfx/lanxiumei/lxm/whq/",dirs[i])
        img_BGR = cv2.imread(path)
        # 使用bgr转化hsv
        hsv = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
        list1[:,:,:,i]=hsv
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', OnMouseAction)
    while True:
        cv2.imshow('frame',frame)
    # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("F:/ygszfx/lanxiumei/lxm/biaojiwhq2.jpg", frame)
            break
    
        #plt.savefig(f'F:/ygszfx/ycl/220330-h-1-qudou-peizhun1-icg-junzhi-resize1/{i}.jpg')
        