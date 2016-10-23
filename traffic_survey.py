import numpy as np
import cv2
import identification
width = 640
hight = 480
face_f =(97,97)
track_window=[None]*10
roi_hist=[None]*10
count=0
face_count=[1]*500
check=0
xg_t=[0]*10
yg_t=[0]*10
ch=[0]*10
trackcheck_map=[0]*10

def detect(img,cascade,minsize):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=minsize,
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")#カスケードファイルのパス
    cascade = cv2.CascadeClassifier(cascade_fn)
    #cap = cv2.VideoCapture('Output2.avi')　#動画ファイルをキャプチャする場合
    cap = cv2.VideoCapture(0)               #webカメラを使用する場合
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    first_flag=1
    while(True):
        ret, frame = cap.read()
        
        if ret == True:
            frame = cv2.resize(frame,(width,hight))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = detect(gray, cascade,face_f)
            vis = frame.copy()
            
            draw_rects(vis, rects, (0, 255, 0))
            face_flag=0
            for x1, y1, x2, y2 in rects:
                xg=(x1+x2)/2
                yg=(y1+y2)/2
                face_flag=1
                flag=1
                roi = frame[y1:y2, x1:x2]
                if ((10<x1) and(width-10>x2)and(hight-10>y2)and (10<y1)):
                    flag=0
                    for j in xrange(0,10):
                        if track_window[j]:
                            (x,y,w,h) = track_window[j]
                            if((x-30<xg)and(xg<30+x+w)and(y-30<yg)and(yg<30+y+h)):
                                flag=1
                                
                                if face_count[trackcheck_map[j]]<10 :
                                    face_count[trackcheck_map[j]]+=1
                                    face = "face/face%d_%d.jpg" %(trackcheck_map[j],face_count[trackcheck_map[j]]) 
                                    cv2.imwrite(face,roi)#あらかじめ対応するフォルダを作成しておく
                                track_window[j] = (x1, y1, x2-x1, y2-y1)

                                roi = frame[y1:y2, x1:x2]
                                
                                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
                                img_mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    
                                roi_hist[j] = cv2.calcHist([hsv_roi], [0], img_mask, [180], [0,180])
        
                                cv2.normalize(roi_hist[j], roi_hist[j], 0, 255, cv2.NORM_MINMAX)
                                break
                    if flag!=1:
                        check+=1
                        
                        face = "face/face%d_1.jpg" %(check)
                        cv2.imwrite(face,roi)
                        
                        print (check)

                        for count in xrange(0,10):
                            if track_window[count]==None:
                                track_window[count] = (x1, y1, x2-x1, y2-y1)
                                trackcheck_map[count]= check
                                ch[count]=str(check)+"_HUMAN"
                                roi = frame[y1:y2, x1:x2]

                                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
                                img_mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    
                                roi_hist[count] = cv2.calcHist([hsv_roi], [0], img_mask, [180], [0,180])
   
                                cv2.normalize(roi_hist[count], roi_hist[count], 0, 255, cv2.NORM_MINMAX)
                                #print (track_window)
                                break
            for i in xrange(0, 10):
                if track_window[i]:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
                    dst = cv2.calcBackProject([hsv],[0],roi_hist[i],[0,180], 1)
 
            
                    ret, track_window[i] = cv2.meanShift(dst, track_window[i], term_crit)
                       
                    x,y,w,h = track_window[i]
                    xg_t[i]=(x+x+w)/2
                    yg_t[i]=(y+y+h)/2
                    if not ((5<x) and(width-5>x+w)and(hight-5>y+h)and (5<y)or(face_flag)):
                        track_window[i]=None
                    for i2 in xrange(0,10):
                        if track_window[i2]:
                            (xl,yl,wl,hl) = track_window[i2]
                            if(xl<xg_t[i])and(xg_t[i]<xl+wl)and(yl<yg_t[i])and(yg_t[i]<yl+hl)and(i!=i2):
                                track_window[i2]=None

                    if track_window[i]:
                        x,y,w,h = track_window[i]
                        cv2.rectangle(vis, (x,y), (w+x, h+y), 255, 2)
                        cv2.putText(vis,ch[i],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255))
            real = "a total of "+str(check)+" people"
            cv2.putText(vis,real,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,100,255))
            cv2.imshow('SHOW MEANSHIFT IMAGE', vis)
 
           
            k = cv2.waitKey(1)
            if k == ord('q'):
                print("end")
                
                test.sosu(check,face_count)
                f = open("face/total_people.txt","w")
                f.write(str(check))
                f.close()
                break
        else:
            print("end")
            test.sosu(check,face_count)
            f = open("face/total_people.txt","w")
            f.write(str(check))
            f.close()
            break

