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
YELLOW_LOWER = np.array([10, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])

'''Function to reduce glare by using CLAHE'''
def reduceGlare(gray, img, name):
    # dst = cv2.GaussianBlur(gray,(7,7),cv2.BORDER_DEFAULT)
    # # reduce glare by using CLAHE 
    # grayimg = dst
    # #HSV
    # frame_threshed = cv2.inRange(frame_hsv, GLARE_MIN, GLARE_MAX)
    # #INPAINT
    # mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    # #CLAHE
    # clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    # claheCorrecttedFrame = clahefilter.apply(grayimg)
    # #COLOR 
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # lab_planes = list(cv2.split(lab))
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # #INPAINT + CLAHE
    # grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    # mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    # result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA) 
    # cv2.imwrite(name+'xglare.png',result2)
    # return result2
    # threshold
    hh, ww = img.shape[:2]
    lower = (150,150,150)
    upper = (240,240,240)
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)

    # floodfill the outside with black
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = morph.copy()
    mask = cv2.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]

    # use mask with input to do inpainting
    result1 = cv2.inpaint(img, mask, 101, cv2.INPAINT_TELEA)
    result2 = cv2.inpaint(img, mask, 101, cv2.INPAINT_NS)

    # write result to disk
    #cv2.imwrite("tomato_inpaint2.jpg", result2)

    # display it
    cv2.imshow("RESULT2", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    glare_reduced = reduceGlare(gray, img, name[:-4])
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
    not_green[imask] = glare_reduced[imask]
    # brown color
    # Create the brown HSV mask
    brown_mask = cv2.inRange(frame_hsv, BROWN_LOWER, BROWN_UPPER)
    # slice the brown
    bmask = brown_mask>0
    brown = np.zeros_like(img, np.uint8)
    # brown is part of the image that is brown
    brown[bmask] = glare_reduced[bmask]
    # combine the masks and detect areas of brown in non-green regions 
    target = cv2.bitwise_and(not_yellow, not_green)
    target = cv2.bitwise_and(target, brown)
    # cv2.imshow(name, target)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
            

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