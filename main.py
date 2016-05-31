from PIL import Image
import cv2
import numpy as np
import os
from numpy import array
from matplotlib import pyplot as plt
import time



########### TRAINING DATA: ##########


def load_data(letter):
	arr = []
	list_of_files = os.listdir(os.getcwd())
	for each_file in list_of_files:
		if each_file.startswith(letter):
			arr.append(each_file)
	return arr


def contours_for_th_frame(th):
	contours, _ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	numberofpoints = 0
	for contour in contours:
		for point in contour:
			numberofpoints = numberofpoints +1

	allcountours = np.zeros((numberofpoints,1,2), dtype=np.int32)
	count = 0
	for contour in contours:
		for point in contour:
			allcountours[count][0] = [point[0][0],point[0][1]]
			count = count + 1
		cnt = allcountours


def define_contours(image):
	im = cv2.imread(image, 1)
	img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	contours, _ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	numberofpoints = 0
	for contour in contours:
		for point in contour:
			numberofpoints = numberofpoints +1

	allcountours = np.zeros((numberofpoints,1,2), dtype=np.int32)
	count = 0
	for contour in contours:
		for point in contour:
			allcountours[count][0] = [point[0][0],point[0][1]]
			count = count + 1
		cnt = allcountours

	cv2.imwrite("result.jpg",th)
	return cnt

def compare_contours(img1,img2):
	return cv2.matchShapes(define_contours(img1),define_contours(img2),1,0.0)


def arr_contours(letter):
	ans = []
	arr = load_data(letter)
	for item in arr:
		ans.append(define_contours(item))

	return ans

def secure_shapes(letter):
	ans = []
	arr = load_data(letter)
	for x in range(0,len(arr)-1):
		for y in range(x+1,len(arr)):
			ans.append(compare_contours(arr[x],arr[y]))
	return ans




a_arr = arr_contours('a')
b_arr = arr_contours('b')
c_arr = arr_contours('c')
d_arr = arr_contours('d')
w_arr = arr_contours('w')
y_arr = arr_contours('y')
id_arr = arr_contours('id')
p_arr = arr_contours('p')



yes_found = True
disagree_found = True
		
robot_found=False
cola_found = True
wait_found = True
plate_found = True
alcoholic_found = True




######### LOCKS ##########

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
	robot_found=True

	cola_found = False
	wait_found = False
	plate_found = False
	alcoholic_found = False


def get_smallest(arr,th_frame):
	minimum = 3.0
	for index in arr:
		shape = cv2.matchShapes(index,th_frame,1,0.0)
		minimum = min(shape,minimum)
	return minimum

def find_contours(a,b,c):
  r = cv2.findContours(a,b,c)
  if len(r) == 2:
	return r
  return r[1:]

def start():
	
	cap = cv2.VideoCapture(0)

	while(cap.isOpened):
		_, frame = cap.read()
		#frame=cv2.medianBlur(frame,5)

		im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		pil = Image.fromarray(im_gray)
		cropped = array(pil.crop((200,200,600,600))) # 200 and 600


		cv2.rectangle(frame,(200,200),(600,600),(155,255,0),2)

		thresh = cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
		#_,thresh = cv2.threshold(cropped,170,255,0)


		contours, hierarchy1 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
 
		numberofpoints = 0
		for contour in contours:
		  for point in contour:
			numberofpoints = numberofpoints +1 

		allcountours = np.zeros((numberofpoints,1,2), dtype=np.int32)
		count = 0
		for contour in contours:
		  for point in contour:
			allcountours[count][0] = [point[0][0],point[0][1]]
			count = count + 1
		cnt = allcountours 
			
		
		if ( len(cnt) > 2 and (yes_found==False) and (get_smallest(y_arr,cnt)<0.05)) :
			print "Yes!"
			unlock()
		
		if ( len(cnt) > 2 and (wait_found==False) and (get_smallest(w_arr,cnt)<0.05) ) :
			print "Cash received"
			lock()
				
		if ( len(cnt) > 2 and (robot_found==False) and (get_smallest(b_arr,cnt)<0.05)) :
			print "Hi! I'm the SUPER bartender Robot :) "
			unlock()
		
		if ( len(cnt) > 2 and (cola_found==False) and (get_smallest(c_arr,cnt)<0.05)) :
			print "Soft drink added"
			lock()



		cv2.imshow('frame',thresh)
 		cv2.waitKey(25)

cv2.destroyAllWindows()


