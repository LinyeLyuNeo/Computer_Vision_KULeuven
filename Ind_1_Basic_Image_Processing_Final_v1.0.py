# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:17:40 2021

author: Linye Lyu, r0481422

Last Modified: March 11th, 2021

Course: Computer Vision

Master of AI, KU Leuven

Individual Assignment 1

"""



# import libraries 
import cv2
import numpy as np

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# function to show debug text on the video frame
def debugFrame(frame):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text='Debug Frame',org=(10,500),fontFace=font, fontScale=2, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
    
    return frame

# function to convert a color frame to gray frame
def colorToGray(frame):
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # need to convert gray image to color image again so it can be written to the video
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

    
    return frame

# function to apply a Guassian filter based on the kernel size
def gaussian(frame, ksize=5):
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.GaussianBlur(frame,(ksize,ksize),10)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'GaussianBlur, ksize= ' + str(ksize)
    cv2.putText(frame, text,org=(10,500),fontFace=font, fontScale=1.5, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
    
    return frame

# function to apply a Bilateral filter based on the corner size 
def bilateral(frame, ksize=5):
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)           
    frame = cv2.bilateralFilter(frame,ksize,75,75)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'bilateralFilter, filter size= ' + str(ksize)
    cv2.putText(frame, text,org=(10,500),fontFace=font, fontScale=1.5, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
    
    return frame


# function to grab a black object in RGB color space 
def grabRGB(frame):
    
    framesize = frame.shape

    width = framesize[0]
    length = framesize[1]
    
    # Create a blank frame size red image
    red_img = np.zeros((width, length, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    red_img[:] = (0, 0, 255)
    
    
    # define the BGR limits to grab
    
    lower_bgr = np.array([0,0,0])
    upper_bgr = np.array([60,55,50])
    
    mask = cv2.inRange(frame, lower_bgr, upper_bgr)
    
    
    frame = cv2.bitwise_and(red_img, red_img, mask = mask)
    
    # morphological operations to improve the grab
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.erode(frame,kernel,iterations = 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text='Grab with RGB',org=(10,500),fontFace=font, fontScale=2, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
    
    return frame

# function to grab a black object in HSV color space 
def grabHSV(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    framesize = frame.shape

    width = framesize[0]
    length = framesize[1]

    # define the HSV limits to grab
    
    lower_hsv = np.array([0,0,0])
    upper_hsv = np.array([180,255,60])
    
    mask = cv2.inRange(frame, lower_hsv, upper_hsv)
    
    
    # Create a blank frame size red image
    red_img = np.zeros((width, length, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    red_img[:] = (0, 0, 255)

    frame = cv2.bitwise_and(red_img, red_img, mask = mask)
    
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.erode(frame,kernel,iterations = 1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text='Grab with HSV',org=(10,500),fontFace=font, fontScale=2, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA) 
    
    return frame


# function to detect edges using sobel edge detector 
def sobelEdge(frame,ksize,scale):
    
    # apply smoothing to the frame
    frame= cv2.GaussianBlur(frame, (15,15), 0)
    
    # convert it into gray scale for edge detection
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        
    
    sobelx = cv2.Sobel(frame,cv2.CV_32F,1,0,ksize=ksize,scale=scale)
    sobely = cv2.Sobel(frame,cv2.CV_32F,0,1,ksize=ksize,scale=scale)
    
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    frame = cv2.addWeighted(src1=abs_sobelx,alpha=0.5,src2=abs_sobely,beta=0.5,gamma=0)
    
    # morphological operations to improve the edge detection
    kernel = np.ones((5,5),np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Sobel Edge Detection, ksize= ' + str(ksize) + ', scale= ' +str(scale)
    cv2.putText(frame,text,org=(10,500),fontFace=font, fontScale=1, color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
    
    return frame


# function to detect circle edges using hough circle detector 
def houghCircle(frame, param1=100, param2=30, minRadius=1, maxRadius=30):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    gray = cv2.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Hough Circle, param1= ' + str(param1) + ', param2= ' + str(param2) + ', maxR= ' +str(maxRadius)
    cv2.putText(frame,text,org=(10,500),fontFace=font, fontScale=1, color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
        
    return frame

# function to detect a object using template matching 
def objectDetection(frame, template='C:/Users/ASUS/Documents/CV_Assignments/part.png'):
    
    full = frame.copy()
    
    part = cv2.imread(template)
    
    method = cv2.TM_CCOEFF
    
    res = cv2.matchTemplate(full, part, method)
    
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    height,width, channels = part.shape
    
    bottom_right = (top_left[0]+width,top_left[1]+height)
    
    cv2.rectangle(full,top_left,bottom_right,(0,0,255),5)
    
    
    
    frame = full
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Template Matching to Find Tequilla!'
    cv2.putText(frame,text,org=(10,500),fontFace=font, fontScale=1, color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
    
    return frame


# function to show the likelihood map of the template matching 
def objectLikelihood(frame,frame_w,frame_h,template='C:/Users/ASUS/Documents/CV_Assignments/part.png'):
    
    full = frame.copy()
    
    part = cv2.imread(template)
    
    method = cv2.TM_CCOEFF
    
    res = cv2.matchTemplate(full, part, method)
    
    likelihood_map = res
    
    likelihood_map = cv2.cvtColor(cv2.normalize(likelihood_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR)
    likelihood_map = cv2.resize(likelihood_map, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
     
    frame = likelihood_map
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Likelihood Map to Find Tequilla!'
    cv2.putText(frame,text,org=(10,500),fontFace=font, fontScale=1, color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
    
    return frame

# function to show the corner detection using Good Features to Track
def cornerDetection(frame):
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray_frame,80,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)
    
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Corner Detections: Good Features to Track!'
    cv2.putText(frame,text,org=(10,500),fontFace=font, fontScale=1, color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
    
    
    return frame


# function that use face tracking API to track face moving. 
def faceTracking(frame,tracker):
    
    # Update tracker
    success, roi = tracker.update(frame)
    
    # roi variable is a tuple of 4 floats
    # We need each value and we need them as integers
    (x,y,w,h) = tuple(map(int,roi))
    
    # Draw Rectangle as Tracker moves
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
    else :
        # Tracking failure
        cv2.putText(frame, "Failure to Detect Tracking!!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # Display tracker type on frame
    cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3);

    
    return frame

tracker = cv2.TrackerMIL_create()
tracker_name = str(tracker).split()[0][1:]



input_video_file = 'C:/Users/ASUS/Documents/CV_Assignments/mysupervideo_1.mp4'
output_video_file = 'C:/Users/ASUS/Documents/CV_Assignments/mysupervideo_temp.mp4'

cap = cv2.VideoCapture(input_video_file)
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4

out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

# while loop where the real work happens
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
      
        #################################
        # Part 1: Basic Image Processing# 
        #################################
        # task 1: switching the video between color and grayscale for few times: 0-4s
  
        if between(cap, 0, 1000) or between(cap, 2000, 3000):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text='Color',org=(10,500),fontFace=font, fontScale=2, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
            
        if between(cap, 1000, 2000) or between(cap, 3000, 4000):           
            frame = colorToGray(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text='Grayscale',org=(10,500),fontFace=font, fontScale=2, color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)

        # task 2: smoothing using guassian and bi-lateral filter, 
        # increase the effect of blurring by widen the filter kernel. : 4-12s
        
        # Guassian Filter
 
        if between(cap, 4000, 6000):
            ksize = 7
            frame = gaussian(frame,ksize)
        if between(cap, 6000, 8000):
            ksize = 15
            frame = gaussian(frame,ksize)
            
        # Bilateral Filter
        if between(cap, 8000, 10000):         
            ksize = 7       
            frame = bilateral(frame,ksize)
 
        if between(cap, 10000, 12000):
            ksize = 15       
            frame = bilateral(frame,ksize)
        
        # task 3: Grab your object in RGB and HSV color space. 
        #Show binary frames with the foreground object in white and background in black. : 12-20s

            
        # grab in RGB
        if between(cap, 12000, 16000):
            frame = grabRGB(frame)
            
            
        # grab in HSV
        if between(cap, 16000, 20000): 
            frame = grabHSV(frame)
            
        #################################    
        # Part 2 Object Detection 20-40s#
        #################################
        
        # task 1: Sobel edge detector: 20-25s
        if between(cap, 20000, 21250):
            ksize = 5
            scale = 5
            frame = sobelEdge(frame,ksize,scale)
        if between(cap, 21250, 22500):
            ksize = 5
            scale = 3
            frame = sobelEdge(frame,ksize,scale)
        if between(cap, 22500, 23750):
            ksize = 3
            scale = 5
            frame = sobelEdge(frame,ksize,scale)
        if between(cap, 23750, 25000):
            ksize = 3
            scale = 3
            frame = sobelEdge(frame,ksize,scale)
        
        # Hough transform to detect circular shape 25s-35s
        if between(cap, 25000, 27000):
            param1 = 100
            param2 = 30
            minRadius = 1
            maxRadius = 0
            frame = houghCircle(frame, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if between(cap, 27000, 29000):
            param1 = 150
            param2 = 30
            minRadius = 1
            maxRadius = 0
            frame = houghCircle(frame, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if between(cap, 29000, 31000):
            param1 = 50
            param2 = 30
            minRadius = 1
            maxRadius = 50
            frame = houghCircle(frame, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if between(cap, 31000, 33000):
            param1 = 180
            param2 = 30
            minRadius = 1
            maxRadius = 100
            frame = houghCircle(frame, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)     
        if between(cap, 33000, 35000):
            param1 = 180
            param2 = 30
            minRadius = 1
            maxRadius = 200
            frame = houghCircle(frame, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)  
        
#         Object Detection 35s-40s
        if between(cap, 35000, 37000):
            frame =objectDetection(frame)
            
        if between(cap, 37000, 40000):
            frame =objectLikelihood(frame,frame_width,frame_height)    
        
        
        ################################# 
        # Part 3 Carte blanche: 40 - 60s#
        ################################# 


#       Corner Detection 40s-45s
        if between(cap, 40000, 45000):
            
            frame = cornerDetection(frame)
           
        # Face Detection using Tracking API 50s-60s
        if between(cap, 45000, 60000):
            
            frame = faceTracking(frame,tracker)
        
        
    
        
        # (optional) display the resulting frame
        cv2.imshow('Frame', frame)

        # write frame that you processed to output
        out.write(frame)

        

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture and writing object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
