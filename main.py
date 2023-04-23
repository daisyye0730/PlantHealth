import cv2 
import glob
import numpy as np

# constants 
BROWN_LOWER = (5, 0, 0)
BROWN_UPPER = (33, 255, 255)
GREEN_LOWER = (32, 0, 0)
GREEN_UPPER = (86, 255,255)
YELLOW_LOWER = np.array([10, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])

'''Function to reduce glare by using CLAHE'''
def reduceGlare(img, name):
    #COLOR 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #INPAINT + CLAHE
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA)
    return result2

'''algorithm for brown spot -- daisy'''
def checkBrownSpot(img, name):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # find the green color 
    # mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
    # # find the brown color
    # mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # # find the yellow color in the leaf
    # mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

    # # find any of the three colors(green or brown or yellow) in the image
    # mask = cv2.bitwise_or(mask_green, mask_brown)
    # mask = cv2.bitwise_or(mask, mask_yellow)

    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(img,img, mask= mask)
    # # blurred = cv2.blur(img, (3, 3)) 
    # # imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # cv2.imshow(name, res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #calculate surface area of the leaf 
    #by subtracting the number of black pixels from the entire image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sa = cv2.countNonZero(gray) 
    # reduce the glare 
    glare_reduced = reduceGlare(img, name[:-4])
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
    target = cv2.bitwise_and(target, target, mask=brown_mask)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    brown_count = cv2.countNonZero(gray) 
    ratio = brown_count/sa
    print(name, ratio)
            

'''algorithm for yellowing of leaf -- daisy''' 
def checkYellowing(img, name): 
    pass


'''algorithm for discoloration spots -- raymond'''
def checkDiscoloration(img, name):
    # Check for anything that is not green (or black/background)
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)               # Convert to HSV space
    green_mask = cv2.inRange(frame_hsv, GREEN_LOWER, GREEN_UPPER)  # Create mask for green regions
    not_green_pos = ~green_mask > 0                                # Find which pixels are not green
    not_green = np.zeros_like(img, np.uint8)                       # Prepare new image matrix
    not_green[not_green_pos] = img[not_green_pos]                  # Set each pixel

    # Get image dimensions
    pixels_leaf = 0
    pixels_discolored = 0
    height, width, _ = not_green.shape

    # Going through each pixel of a large image can take a few seconds...
    print("[Discoloration] Calculating...")

    # Go through each pixel
    for r in range(height):
        for c in range(width):
            if any(img[r][c]):        # If pixel in image is not (0,0,0) = black
                pixels_leaf += 1
            if any(not_green[r][c]):  # If pixel under not-green mask is not (0,0,0) = black
                pixels_discolored += 1
                not_green[r][c] = (0, 0, 255)  # Mark the discolored parts in red

    # Print stats
    print(f"[Discoloration] {pixels_discolored:,} discolored pixels / {pixels_leaf:,} total pixels = {pixels_discolored / pixels_leaf * 100}% discolored")

    # Make a new showing discolored parts on left, original image on right
    horiz = np.concatenate((not_green, img), axis=1)
    cv2.imshow(name, horiz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    checkDiscoloration(img, name)