import cv2 
import glob
import numpy as np

# constants 
BROWN_LOWER = (5, 0, 0)
BROWN_UPPER = (33, 255, 255)
GREEN_LOWER = (32, 0, 0)
GREEN_UPPER = (86, 255,255)
GLARE_MIN = np.array([0, 0, 50],np.uint8)
GLARE_MAX = np.array([0, 0, 225],np.uint8)
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])

'''Function to reduce glare by using CLAHE'''
def reduceGlare(gray, frame_hsv):
    # reduce glare by using CLAHE 
    grayimg = gray
    #HSV
    frame_threshed = cv2.inRange(frame_hsv, GLARE_MIN, GLARE_MAX)
    #INPAINT
    mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    #CLAHE
    clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    claheCorrecttedFrame = clahefilter.apply(grayimg)
    #COLOR 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #INPAINT + HSV
    result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 
    #INPAINT + CLAHE
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    #HSV+ INPAINT + CLAHE
    lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    lab_planes1 = list(cv2.split(lab1))
    clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes1[0] = clahe1.apply(lab_planes1[0])
    lab1 = cv2.merge(lab_planes1)
    clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    return clahe_bgr1

'''algorithm for brown spot -- daisy'''
def checkBrownSpot(img, name):
    # blurred = cv2.blur(img, (3, 3)) 
    # imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # calculate surface area of the leaf 
    # by subtracting the number of black pixels from the entire image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sa = cv2.countNonZero(gray)  
    # Convert the image to HSV:
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # reduce the glare 
    glare_reduced = reduceGlare(gray, frame_hsv)
    frame_hsv = cv2.cvtColor(glare_reduced, cv2.COLOR_BGR2HSV)
    # mask of yellow
    yellow_mask = cv2.inRange(frame_hsv, YELLOW_LOWER, YELLOW_UPPER)
    # get rid of the yellow part of the leaf 
    imask = ~yellow_mask>0
    not_yellow = np.zeros_like(img, np.uint8)
    # not_yellow is part of the image that is not yellow
    not_yellow[imask] = glare_reduced[imask]
    # mask of green
    green_mask = cv2.inRange(frame_hsv, GREEN_LOWER, GREEN_UPPER)
    # get rid of the green part of the leaf 
    imask = ~green_mask>0
    not_green = np.zeros_like(img, np.uint8)
    # not_green is part of the image that is not yellow
    not_green[imask] = not_yellow[imask]
    # brown color
    # Create the brown HSV mask
    brown_mask = cv2.inRange(frame_hsv, BROWN_LOWER, BROWN_UPPER)
    # slice the brown
    bmask = brown_mask>0
    brown = np.zeros_like(img, np.uint8)
    # brown is part of the image that is brown
    brown[bmask] = glare_reduced[bmask]
    # combine the masks and detect areas of brown in non-green regions 
    target = cv2.bitwise_and(not_yellow, brown)
    target = cv2.bitwise_and(target, not_green)
    cv2.imshow(name, target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            

'''algorithm for yellowing of leaf -- daisy''' 
def checkYellowing(img, name): 
    pass

'''main'''
# load all images from the Library directory
images = {}
for filename in glob.glob("./library/*.png"):
    img = cv2.imread(filename)
    if img is not None:
        images[filename] = img
        
for name, img in images.items(): 
    checkBrownSpot(img, name)
    checkYellowing(img, name)