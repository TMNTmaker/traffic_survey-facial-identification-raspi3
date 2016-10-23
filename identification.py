import numpy as np
import cv2
descripters = cv2.BRISK_create()
descripter_test = cv2.BRISK_create()
surf = cv2.xfeatures2d.SURF_create()
sift = cv2.xfeatures2d.SIFT_create()
#latch = cv2.xfeatures2d.LATCH_create()
#brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#daisy = cv2.xfeatures2d.DAISY_create()
#freak = cv2.xfeatures2d.FREAK_create()
orb = cv2.ORB_create()
af = cv2.AgastFeatureDetector_create()
#fast = cv2.FastFeatureDetector_create()
#mser = cv2.MSER_create()
#akaze = cv2.AKAZE_create()
#kaze = cv2.KAZE_create()
matchers = cv2.BFMatcher()
print ("stanby")

def calc(detector,descripter,matcher,face_image,flag,d2,d3):
    k1,d1 = detector.detectAndCompute(face_image, None)
    min_dist = 100000
    try:
        if flag == 0:
            matches = matcher.match(d1,d2)
        else:
            matches = matcher.match(d1,d3)
    except:
        return min_dist
    dist = [m.distance for m in matches]
    if len(dist) < 3:
        return min_dist
    sorted_dist = sorted(dist)
    top3_min_dist = 0
    top3_min_dist = sorted_dist[0]+sorted_dist[1]+sorted_dist[2]+sorted_dist[3]      
    min_dist = min(top3_min_dist / 4, min_dist)
	return min_dist


def sosu(M,c):
    z = [[0 for j in range(M+1)] for i in range(M+1)]
    max_image=cv2.imread("face/face1_1.jpg")
    match_ch=[0]*M
    K = M+1
    keypoints2_max=[0]*K
    for i in xrange(1,K):
        max_k=0
        for u in xrange(1,c[i-1]):
            test_file = "face/face%d_%d.jpg" %(i,u)
            test_image = cv2.imread(test_file)
            keypoints2 = af.detect(test_image)
            if(max_k<len(keypoints2)):
                max_k = len(keypoints2)
                keypoints2_max[i] = keypoints2
                max_image = test_image
        test_file = "face/face%d.jpg" %i
        cv2.imwrite(test_file,max_image)
    for i in xrange(1,K-1):
	    test_file = "face/face%d.jpg" %i 
	    test_image = cv2.imread(test_file)
	    k2,d2 = sift.detectAndCompute(test_image, None)
	    k3,d3 = surf.detectAndCompute(test_image,None)
        img2 = cv2.drawKeypoints(test_image,k4 , None)
        key_file = "face_key/keypoints%d.jpg" %i
        cv2.imwrite(key_file,img2)
        for p in xrange(i+1,K):
	        face = "face/face%d.jpg" %p
            images = cv2.imread(face)
            t_f=0
            resut1 = calc(sift, descripters, matchers,images,t_f,d2,d3)
            t_f=1
            resut2 = calc(surf, descripter_test, matchers,images,t_f,d2,d3)
            if (resut1 < 200 and resut2 < 0.20 ):
                z[i][p] = 1
                print(resut1,resut2,'match',i,' & ',p) 
            else:
                print(resut1,resut2,'not match',i,' & ',p)
    h = 1
    for i in xrange(1,K-1):
        flag=0
        for p in xrange(i+1,K):
            if( z[i][p] == 1):
                flag=1
        if(flag==0):
            h=h+1
    print(z)
    print('sousu :',h)
