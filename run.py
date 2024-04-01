
from ultralytics import YOLO
import math
import cv2
import numpy as np
import speech_recognition as sr
import os, shutil

window_name = 'Image'

def line_finder(x1,y1,x2,y2,x3,y3,x4,y4):
    
    len = math.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    wid = math.sqrt(pow((x1-x4),2) + pow((y1-y4),2))
    
    if len < wid :
        xx1 = x1 + (x2-x1)/2
        xx2 = x4 + (x3-x4)/2
        yy1 = y1 + (y2-y1)/2
        yy2 = y4 + (y3-y4)/2
    else :
        xx1 = x1 + (x4-x1)/2
        xx2 = x2 + (x3-x2)/2
        yy1 = y1 + (y4-y1)/2
        yy2 = y4 + (y3-y4)/2
    
    
    
    
    
    
    return xx1,xx2,yy1,yy2

def draw_midpoint(image,index_of_obb_check, obb_coords,index_to_check):
    if index_of_obb_check != 100:    
        x1 = obb_coords[index_of_obb_check][0][0].item()
        x2 = obb_coords[index_of_obb_check][1][0].item()
        x3 = obb_coords[index_of_obb_check][2][0].item()
        x4 = obb_coords[index_of_obb_check][3][0].item()
        y1 = obb_coords[index_of_obb_check][0][1].item()
        y2 = obb_coords[index_of_obb_check][1][1].item()
        y3 = obb_coords[index_of_obb_check][2][1].item()
        y4 = obb_coords[index_of_obb_check][3][1].item()
        
        
        xx1,xx2,yy1,yy2 = line_finder(x1,y1,x2,y2,x3,y3,x4,y4)
        
        slopp = slope_finder(xx1,yy1,xx2,yy2)
        
        
        
        if index_to_check == 0 or   index_to_check == 1:
            center_point = ( int(x1 + (x3-x1)/2 ) ,int(y2 + (y4-y2)/2))
        else:
            
            center_point = draw_screwdriver(image,index_of_obb_check,obb_coords)
            
        image = cv2.circle(image, center_point, radius = 10, color =(0,0,0), thickness=10)
        
    else:
        slopp = 0
    
    return image,slopp

def draw_screwdriver(image,index_of_obb_check,obb_coords):
    

    
    x1 = obb_coords[index_of_obb_check][0][0].item()
    x2 = obb_coords[index_of_obb_check][1][0].item()
    x3 = obb_coords[index_of_obb_check][2][0].item()
    x4 = obb_coords[index_of_obb_check][3][0].item()
    y1 = obb_coords[index_of_obb_check][0][1].item()
    y2 = obb_coords[index_of_obb_check][1][1].item()
    y3 = obb_coords[index_of_obb_check][2][1].item()
    y4 = obb_coords[index_of_obb_check][3][1].item()
    
    


    # Find the minimum and maximum x and y coordinates to create the bounding rectangle
    x_min = int(min(x1, x2, x3, x4))
    x_max = int(max(x1, x2, x3, x4))
    y_min =int(min(y1, y2, y3, y4))
    y_max = int(max(y1, y2, y3, y4))

    # Crop the image to the bounding box of the screwdriver
    screwdriver_region = image[y_min:y_max, x_min:x_max]

    # Convert the cropped image to grayscale
    gray_screwdriver = cv2.cvtColor(screwdriver_region, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (adjust the threshold value as needed)
    _, thresholded = cv2.threshold(gray_screwdriver, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the handle is the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        handle_coordinates = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        # Draw the contour on the original image for visualization (optional
        # Extract the handle region
        x, y, w, h = cv2.boundingRect(max_contour)
        handle_region = screwdriver_region[y:y+h, x:x+w]
    
        
        
    else:
        print("No contour found for the handle.")

    
    
    img = cv2.cvtColor(screwdriver_region, cv2.COLOR_BGR2GRAY) 
    ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    kernel = np.ones((10, 10), np.uint8) 
    
    # The first parameter is the original image, 
    # kernel is the matrix with which image is 
    # convolved and third parameter is the number 
    # of iterations, which will determine how much 
    # you want to erode/dilate a given image. 

    image = cv2.dilate(thresh1, kernel, iterations=1) 



    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find black pixels (value = 0) and calculate their coordinates
    black_pixels = np.where(binary_image == 0)
    black_coordinates = list(zip(black_pixels[1], black_pixels[0]))  # (x, y) format

    # Calculate the average coordinates
    average_x = int(sum(x for x, _ in black_coordinates) / len(black_coordinates))
    average_y = int(sum(y for _, y in black_coordinates) / len(black_coordinates))
    cente = (average_x + x_min, average_y + y_min)
    return cente

def slope(x1, y1, x2, y2): 
    return (float)(y2-y1)/(x2-x1) 
    

def slope_finder(x1,y1,x2,y2):
    
    slopp = math.atan(slope(x1,y1,x2,y2)) * 180 / math.pi
    
    
    return slopp


r = sr.Recognizer()

# Reading Audio file as source
# listening the audio file and store in audio_text variable

with sr.AudioFile('openspanner.wav') as source:
    
    audio_text = r.listen(source)
    
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        
        # using google speech recognition
        tt = r.recognize_google(audio_text)
    except:
         print('Sorry.. run again...')

if "screwdriver" in tt:
    print('Going to pick screwdriver')
    index_to_check = 2
elif "open spanner" or "openspanner" in tt:
    print('Going to pick open spanner')
    index_to_check = 0
elif "ring spanner" or "ringspanner" in tt:
    print('Going to pick ring spanner')
    index_to_check = 1
else:
    print('Sorry, I couldn\'t recognize the command')

model = YOLO('best.pt')





cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    obb_coords = results[0].obb.xyxyxyxy
    obb_cls = results[0].obb.cls.numpy()
    annotated_frame = results[0].plot()
    

    index_of_obb_check = 100

    for i in range(len(obb_cls)):
        if int(obb_cls[i]) == index_to_check:
            index_of_obb_check = i
   

    final_image,slopp = draw_midpoint(annotated_frame, index_of_obb_check, obb_coords, index_to_check)
    final_image = cv2.putText(final_image, text=str(slopp), org=(1000,200), fontFace =cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 255), thickness=2,lineType=cv2.LINE_AA) 

    cv2.imshow('Frame', final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


