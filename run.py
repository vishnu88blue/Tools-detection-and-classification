from ultralytics import YOLO
import math
import cv2
import numpy as np
import os, shutil


window_name = 'Image'


def draw_midpoint(image,index_of_obb_check, obb_coords,index_to_check):
    x1 = obb_coords[index_of_obb_check][0][0].item()
    x2 = obb_coords[index_of_obb_check][1][0].item()
    x3 = obb_coords[index_of_obb_check][2][0].item()
    x4 = obb_coords[index_of_obb_check][3][0].item()
    y1 = obb_coords[index_of_obb_check][0][1].item()
    y2 = obb_coords[index_of_obb_check][1][1].item()
    y3 = obb_coords[index_of_obb_check][2][1].item()
    y4 = obb_coords[index_of_obb_check][3][1].item()
    
    
    
    
    
    
    
    
    if index_to_check == 0 or   index_to_check == 1:
        center_point = ( int(x1 + (x3-x1)/2 ) ,int(y2 + (y4-y2)/2))
    else:
        
        center_point = draw_screwdriver(image,index_of_obb_check,obb_coords)
        
    image = cv2.circle(image, center_point, radius = 20, color =(0,0,0), thickness=20)
    
    
    
    return image



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

    






inp = input("Hello, What can i do for you?")
print(inp)

if "screwdriver" in inp:
    print('going to pick screwdriver')
    index_to_check = 2
elif "open spanner" in inp:
    print('going to pick open spanner')
    index_to_check = 0
elif "ring spanner" in inp:
    print('going to pick ring spanner')
    index_to_check = 1
else:
    print('Soory i coudnt recognize the command')







model = YOLO('best.pt')
results=  model.predict('test3.jpg', save=True, conf=0.5, classes=[index_to_check])
obb_coords = results[0].obb.xyxyxyxy  
obb_cls = results[0].obb.cls.numpy()
for i in range(len(obb_cls)):
    if int(obb_cls[i]) == index_to_check :
        index_of_obb_check = i
        

path = "runs/obb/predict/test3.jpg"
image = cv2.imread(path)

final_image = draw_midpoint(image, index_of_obb_check,obb_coords,index_to_check)

image = cv2.resize(final_image,(1024,768))
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()


shutil.rmtree("C:/Projects/Object detection/runs/obb/predict2")


