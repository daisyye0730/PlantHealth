import cv2 
import glob
import numpy as np

# constants 
BROWN_LOWER = [6, 50, 0]
BROWN_UPPER = [23, 255, 81]
GREEN_LOWER = (32, 0, 0)
GREEN_UPPER = (86, 255,255)

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
    print(sa)
    # Convert the image to HSV:
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask of green
    mask = cv2.inRange(frame_hsv, GREEN_LOWER, GREEN_UPPER)
    # slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    # green is part of the image that is green
    green[imask] = img[imask]
    # brown color
    # lower_values = np.array(BROWN_LOWER)
    # upper_values = np.array(BROWN_UPPER)
    # # Create the HSV mask
    # mask = cv2.inRange(frame_hsv, lower_values, upper_values)
    # count_brown = 0
    # for i in range (0, len(mask)):
    #     for j in range (0, len(mask[0])):
    #         if mask[i][j].all() == 255: 
    #             count_brown += 1 
    # cv2.imshow(name, green)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
            

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