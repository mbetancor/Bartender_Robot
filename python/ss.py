from PIL import Image
import cv2
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import os


########### TRAINING DATA: ##########

def find_contours(a,b,c):
  r = cv2.findContours(a,b,c)
  if len(r) == 2:
    return r
  return r[1:]

signs = {"Call":"b",
         "Alcoholic":"a",
         "Yes":"y",
         "No":"d",
         "Wait":"w",
         "Plate":"p",
         "Soft":"c",}
         


def sample(filename):
  smp = cv2.imread(filename, 1)
  smp = cv2.cvtColor(smp,cv2.COLOR_BGR2GRAY)  
  smp_th = cv2.adaptiveThreshold(smp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
               cv2.THRESH_BINARY,11,2)  
  contours_smp,_ = find_contours(smp_th,cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
  numberofpoints_smp = 0
  for contour in contours_smp:
    for point in contour:
      numberofpoints_smp = numberofpoints_smp +1 
      
  allcountours_smp = np.zeros((numberofpoints_smp,1,2), dtype=np.int32)
  count_smp = 0
  for contour in contours_smp:
    for point in contour:
      allcountours_smp[count_smp][0] = [point[0][0],point[0][1]]
      count_smp = count_smp + 1
    cnt_smp = allcountours_smp
  return cnt_smp

def load_samples():
  samples = {}
  for sign,prefix in signs.items():
    for filename in os.listdir("samples"):
      if filename.startswith(prefix):
        print(filename)
        if not sign in samples:
          samples[sign] = []
        samples[sign].append(sample("samples/"+filename))
  return samples


"""
def test():
        cv2.imwrite("result_p6.jpg",p6_th)
        cv2.imwrite("result_p5.jpg",p5_th)
        cv2.imwrite("result_p4.jpg",p4_th)
        cv2.imwrite("result_p3.jpg",p3_th)
        cv2.imwrite("result_p2.jpg",p2_th)
        cv2.imwrite("result_p1.jpg",p1_th)
        return cv2.matchShapes(cnt_p4,cnt_p5,1,0.0)
"""

def match(contour, samples, limit = 0.02):
  rank = []
  if (len(contour) >2):
    for name,contours in samples.items():
      for c in contours:
        score = cv2.matchShapes(c,contour,1,0.0)
        if (score < limit):
          rank.append((score,name))
  if len(rank):
    return (sorted(rank))[0]

        
def main_loop(samples):
  cap = cv2.VideoCapture(0)
  
  
  yes_found = True
  disagree_found = True
  wait_found = True

  robot_found=False
  cola_found = False
  plate_found = False
  alcoholic_found = False
  
  
  while(cap.isOpened):
    _, frame = cap.read()
    frame=cv2.medianBlur(frame,11)
    
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(im_gray)

    cropped_1 = array(pil.crop((640,0,1280,720)))
    cropped_2 = array(pil.crop((0,0,640,720)))

    thresh1 = cv2.adaptiveThreshold(cropped_1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,11,2)  
    thresh2 = cv2.adaptiveThreshold(cropped_2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,11,2)  
    contours1, hierarchy1 = find_contours(thresh1,cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE) 
    contours2, hierarchy2 = find_contours(thresh2,cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE) 

    numberofpoints = 0
    for contour in contours1:
      for point in contour:
        numberofpoints = numberofpoints +1 

    allcountours = np.zeros((numberofpoints,1,2), dtype=np.int32)
    count = 0
    for contour in contours1:
      for point in contour:
        allcountours[count][0] = [point[0][0],point[0][1]]
        count = count + 1
    cnt1 = allcountours 

    numberofpoints2 = 0
    for contour in contours2:
      for point in contour:
        numberofpoints2 = numberofpoints2 +1 

    allcountours2 = np.zeros((numberofpoints2,1,2), dtype=np.int32)
    count2 = 0
    for contour in contours2:
      for point in contour:
        allcountours2[count2][0] = [point[0][0],point[0][1]]
        count2 = count2 + 1
    cnt2 = allcountours2 

    limit = 0.05

    m = match(cnt1,samples)
    if m:
      print "CNT1: " + str(m)
    m = match(cnt2,samples)
    if m:
      print "CNT2: " + str(m)


    cv2.imshow('::float:: frame',thresh2)
    cv2.waitKey(30)
   


def lock():
        
        yes_found = False
        disagree_found = False

        robot_found=True
        cola_found = True
        wait_found = True
        plate_found = True
        alcoholic_found = True

def unlock():
        
        yes_found = True
        disagree_found = True

        robot_found=False
        cola_found = False
        wait_found = False
        plate_found = False
        alcoholic_found = False

print "loading samples:"
samples = load_samples()
print "Ready and looking"
main_loop(samples)
