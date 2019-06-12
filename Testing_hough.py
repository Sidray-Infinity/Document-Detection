from __future__ import division
import numpy as np
import cv2
import time
from scipy.misc import toimage
from PIL import Image
import matplotlib.pyplot as plt

RAD_LOW = 5 
RAD_LOW_ACUTE = 85 
RAD_HIGH_ACUTE = 95 
RAD_LOW_OBTUSE = 175 

RAD_ISO_LOW = 80 
RAD_ISO_HIGH = 100

RATIO_LOW = 1.3
RATIO_HIGH = 1.6


def show_image(winName, image):
    cv2.imshow(winName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def det_angle(lb1, lb2):
    return abs(lb1[0][1] - lb2[0][1])

def searchLine(set, line):
    try:
        rho = line[0][0]
        theta = line[0][1]
        for line_block in set:
            if(line_block[0][0] == rho and line_block[0][1] == theta):
                return False
        return True
    except:
        pass

def isolate_lines(lines):
    new_lines = []
    try:
        for i in lines:
            for j in lines:
                if(det_angle(i, j) > np.deg2rad(RAD_ISO_LOW) and det_angle(i, j) < np.deg2rad(RAD_ISO_HIGH)):
                    if searchLine(new_lines, i):
                        new_lines.append(i)
                    if searchLine(new_lines, j):
                        new_lines.append(j)

        return new_lines
    except:
        pass

def seg_hor_ver(lines):

    ''' Using Good ol' 'if else' statements.
        Assumption: Image is algined properly. '''

    hor_lines = []
    ver_lines = []

    try:
        for line_block in lines:
            theta = line_block[0][1]
            if((theta < np.deg2rad(RAD_LOW)) or (theta > np.deg2rad(RAD_LOW_OBTUSE))):
                ver_lines.append(line_block)
            elif(theta > np.deg2rad(RAD_LOW_ACUTE) and theta < np.deg2rad(RAD_HIGH_ACUTE)): # Between 75 and 105
                hor_lines.append(line_block)

        return hor_lines, ver_lines
    except:
        pass

def print_lines(img, lines, color=(255,0,0)):
    temp_img = np.copy(img)
    for line_block in lines:
        r = line_block[0][0]
        theta = line_block[0][1]
        a = np.cos(theta) 
        b = np.sin(theta) 
        x0 = a*r 
        y0 = b*r 
        x1 = int(x0 + 1000*(-b)) 
        y1 = int(y0 + 1000*(a)) 
        x2 = int(x0 - 1000*(-b)) 
        y2 = int(y0 - 1000*(a)) 
        cv2.line(temp_img,(x1,y1), (x2,y2), color,2)

    return temp_img

def sort_lines(set):
    def sort_on_rho(line_bloc):
        return abs(line_bloc[0][0])

    set.sort(key = sort_on_rho)
    return set

def extract_from_frame(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    image = np.asarray(image)
    
    doc_image = image[y:y+h, x:x+w]

    return doc_image

def cal_aspect_ratio(points):
    w1 = abs(points[2][0] - points[1][0])
    w2 = abs(points[3][0] - points[0][0])
    w = min(w1, w2)

    h1 = abs(points[0][1] - points[1][1])
    h2 = abs(points[2][1] - points[3][1])
    h = min(h1, h2)

    ratio = float(w)/float(h)
    #print(ratio)

    if(ratio > RATIO_LOW and ratio < RATIO_HIGH):
        return True
    else:
        return  False

def intersectionPoint(line1, line2):
    """
    Determining intersection point b/w two lines of the form r = xcos(R) + ysin(R)
    """

    y = (line2[0][0]*np.cos(line1[0][1]) - line1[0][0]*np.cos(line2[0][1]))/(np.sin(line2[0][1])*np.cos(line1[0][1]) - np.sin(line1[0][1])*np.cos(line2[0][1]))
    x = (line1[0][0] - y*np.sin(line1[0][1]))/np.cos(line1[0][1])
    return [x,y]

def polygon_area(points):  
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:  
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2

cap = cv2.VideoCapture(0)
flag = True
flag1 = True
count = 0

while(True):

    t1 = time.time()

    ret, frame = cap.read()
    orig = frame.copy()
    
    x = 150
    w = 181
    y = 23
    h = 427

    frame = frame[x:x+w, y:y+h, :]

    frame_area = w*h

    cv2.imshow("ORIG FRAME", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edge = cv2.Canny(blur, 20, 30)
    #frame = np.asarray(frame) 

    cv2.imshow("Canny", edge)

    lines = cv2.HoughLines(edge, rho=1, theta=np.pi/180.0, threshold=90)

    try:
        new_lines = isolate_lines(lines)
        hor_lines, ver_lines = seg_hor_ver(new_lines)   
    
        if(len(hor_lines) != 0 and len(ver_lines) != 0):
            hor_lines = sort_lines(hor_lines)
            ver_lines = sort_lines(ver_lines)

            hor = print_lines(frame, hor_lines)
            ver = print_lines(frame, ver_lines)

            final_lines = [] # Follows Clockwise rotation
            final_lines.append(hor_lines[0])
            final_lines.append(ver_lines[-1])
            final_lines.append(hor_lines[-1])
            final_lines.append(ver_lines[0])

            fin_lines = print_lines(frame, final_lines)
            cv2.imshow("Final", fin_lines)

            # Using linear algebra to determine points

            if(len(final_lines) == 4):
                points = []

                for i in range(0, len(final_lines)-1, 1):
                    x = intersectionPoint(final_lines[i], final_lines[i+1])
                    points.append(x)
                x = intersectionPoint(final_lines[3], final_lines[0]) # For the final two lines
                points.append(x)

                poly_area = abs(polygon_area(points))

                zero_flag = True
                for i in points:
                    if(i[0] == 0.0 or i[1] == 0.0):
                        zero_flag = False

                if(poly_area!=0 and zero_flag):

                    for i in points:
                        print(i)
                    print('----------------------------------')

                    if(cal_aspect_ratio(points)):
                        points = np.array(points, np.int32)
                        points = points.reshape((-1,1,2))
                        if(poly_area/frame_area > 0.5):
                            frame = cv2.polylines(frame, [points], True, (0,255,0), 2)
                        else:
                            frame = cv2.polylines(frame, [points], True, (0,0,255), 2)

        cv2.imshow("FRAME", frame)
    
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
