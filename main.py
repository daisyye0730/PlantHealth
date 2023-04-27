import cv2
import glob
import numpy as np

# constants
# if not commented, the range is in hsv 
BROWN_LOWER = (0, 70, 0)  
BROWN_UPPER = (100, 255, 255)
GREEN_LOWER = (27, 10, 0)
GREEN_UPPER = (86, 255, 240)
GREY_LOWER = (0, 0, 0)
GREY_UPPER = (255, 55, 240)
YELLOW_LOWER1 = (0, 107, 127)
YELLOW_UPPER1 = (30, 255, 255)
YELLOW_LOWER2 = (15, 0, 0)
YELLOW_UPPER2 = (30, 126, 126)
GLARE_LOWER = (220, 220, 220) # rgb
GLARE_UPPER = (255, 255, 255) # rgb
BLUE_LOWER1 = (86, 20, 90) 
BLUE_UPPER1 = (138, 255, 255) 
BLUE_LOWER2 = (86, 0, 0) 
BLUE_UPPER2 = (138, 126, 126) 
AREA_MIN = 300 # constant to determine which brown spots are large enough to be important 

'''Function to reduce glare'''
def reduceGlare(img, name):
    no_glare = cv2.inRange(img, GLARE_LOWER, GLARE_UPPER)
    not_glare_pos = ~no_glare > 0
    # Prepare new image matrix
    not_glare = np.zeros_like(img, np.uint8)
    # Set each pixel
    not_glare[not_glare_pos] = img[not_glare_pos]
    return not_glare


'''function that blurs the edge of the leaf so that it can be ignored later on'''
def blurEdge(img, name): 
    blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
    mask = np.zeros(img.shape, np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255,255,255),5)
    output = np.where(mask==np.array([255, 255, 255]), blurred_img, img)
    return output 


'''Function that filters out the grey regions'''
def filterGrey(frame_hsv):
    # Check for anything that is not grey (or black/background)
    # Create mask for grey regions
    grey_mask = cv2.inRange(frame_hsv, GREY_LOWER, GREY_UPPER)
    # Find which pixels are not grey
    not_grey_pos = ~grey_mask > 0
    # Prepare new image matrix
    not_grey = np.zeros_like(frame_hsv, np.uint8)
    # Set each pixel
    not_grey[not_grey_pos] = frame_hsv[not_grey_pos]
    return not_grey


'''Function that filters out the blue regions'''
def filterBlue(frame_hsv):
    # Check for anything that is not blue (or black/background)
    # we need to slice blue twice because it is a parabolic slice on the hsv color space
    # Create mask for blue regions
    blue_mask1 = cv2.inRange(frame_hsv, BLUE_LOWER1, BLUE_UPPER1)
    not_blue_pos = ~blue_mask1 > 0
    not_blue = np.zeros_like(frame_hsv, np.uint8)
    not_blue[not_blue_pos] = frame_hsv[not_blue_pos]
    blue_mask2 = cv2.inRange(not_blue, BLUE_LOWER2, BLUE_UPPER2)
    # Find which pixels are not blue
    not_blue_pos = ~blue_mask2 > 0
    # Prepare new image matrix
    not_blue2 = np.zeros_like(frame_hsv, np.uint8)
    # Set each pixel
    not_blue2[not_blue_pos] = not_blue[not_blue_pos]
    return not_blue


'''This function ignores all the brown pixels on the edge of the leaf'''
def ignorePixelOnEdge(brownImg, original_img): 
    # first detect the contour of the leaf 
    original_img = cv2.blur(original_img, (5, 5))
    dilated = cv2.dilate(original_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    cont_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    copy = original_img.copy()
    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    for ele in c:
        if ele[0][0] < brownImg.shape[0] and ele[0][1] < brownImg.shape[1] and brownImg[ele[0][0]][ele[0][1]].all() != 0: 
            brownImg[ele[0][0]][ele[0][1]] = 0
    cv2.drawContours(copy, c, -1, (0,255,0))
    horiz = np.concatenate(
        (img, copy, brownImg), axis=1)
    cv2.imshow(name, horiz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''This function detects where the brown regions are relative to the leaf'''
def detectlocation(brown_img, name):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(brown_img, cv2.MORPH_CLOSE, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Finding contours of white square:
    conts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    for cnt in conts:
        area = cv2.contourArea(cnt)

        #filter more noise
        if area > AREA_MIN: 
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2 = x1 + w                   # (x1, y1) = top-left vertex
            y2 = y1 + h                   # (x2, y2) = bottom-right vertex
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


'''algorithm for brown spot -- daisy'''
def checkBrownSpot(img, name):
    # calculate surface area of the leaf
    # by subtracting the number of black pixels from the entire image
    blurred = blurEdge(img, name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sa = cv2.countNonZero(gray)
    # reduce the glare
    glare_reduced = reduceGlare(blurred, name[:-4])
    # filter green, grey, blue from images
    no_green = filterGreen(glare_reduced)
    not_yellow = filterYellow(no_green)
    no_green_grey = filterGrey(not_yellow)
    # no_green_grey_blue = filterBlue(no_green_grey)
    # brown color
    # Create the brown HSV mask
    brown_mask = cv2.inRange(no_green_grey, BROWN_LOWER, BROWN_UPPER)
    brown = cv2.bitwise_and(no_green_grey, no_green_grey, mask=brown_mask)
    # horiz = np.concatenate(
    #     (img, brown), axis=1)
    # cv2.imshow(name, horiz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(brown, cv2.COLOR_BGR2GRAY)
    brown_count = cv2.countNonZero(gray)
    ratio = brown_count/sa
    print(f"[Brown Spot] {brown_count:,} brown pixels / {sa:,} total pixels = {round(ratio,3) * 100}% brown")
    detectlocation(brown, name)


'''algorithm for yellowing of leaf -- daisy'''
def checkYellowing(img, name, isBrown):
    pass


'''function that looks at only non-yellow parts '''
def filterYellow(img):
    # Check for anything that is not yellow (or black/background)
    # Convert to HSV space
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create mask for yellow regions
    yellow_mask = cv2.inRange(frame_hsv, YELLOW_LOWER1, YELLOW_UPPER1)
    yellow_mask2 = cv2.inRange(frame_hsv, YELLOW_LOWER2, YELLOW_UPPER2)
    mask = cv2.bitwise_or(yellow_mask, yellow_mask2)
    # Find which pixels are not yellow
    not_yellow_pos = ~mask > 0
    # Prepare new image matrix
    not_yellow = np.zeros_like(img, np.uint8)
    # Set each pixel
    not_yellow[not_yellow_pos] = img[not_yellow_pos]
    return not_yellow

'''function that looks at only non-green parts '''
def filterGreen(img):
    # Check for anything that is not green (or black/background)
    # Convert to HSV space
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create mask for green regions
    green_mask = cv2.inRange(frame_hsv, GREEN_LOWER, GREEN_UPPER)
    # Find which pixels are not green
    not_green_pos = ~green_mask > 0
    # Prepare new image matrix
    not_green = np.zeros_like(img, np.uint8)
    # Set each pixel
    not_green[not_green_pos] = img[not_green_pos]
    return not_green


'''algorithm for discoloration spots -- raymond'''
def checkDiscoloration(img, name):
    not_green = filterGreen(img)
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
                # Mark the discolored parts in red
                not_green[r][c] = (0, 0, 255)

    # Print stats
    print(f"[Discoloration] {pixels_discolored:,} discolored pixels / {pixels_leaf:,} total pixels = {pixels_discolored / pixels_leaf * 100}% discolored")

    # Make a new showing discolored parts on left, original image on right
    horiz = np.concatenate((not_green, img), axis=1)
    cv2.imshow(name, horiz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''algorithm for detecting holes in leaf -- raymond'''
def checkHoles(img, name):
    mask_img = img.copy() # Create a copy so we do not edit the actual image itself (drawContours() will do that)

    # Make the image grayscale so we can apply at threshold for anything that is not the background (black)
    NOT_BLACK_LOWER_THRESHOLD = 50  # Ignore anything that is 0-49 in black & white
    grayscale_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(grayscale_img, NOT_BLACK_LOWER_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Now, find the largest contour (the leaf)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Draw the border of the leaf in red
    mask_color = (0, 0, 255)
    red_leaf = cv2.drawContours(mask_img, contours=[max_contour], contourIdx=-1, color=mask_color, thickness=cv2.FILLED)  # Creates a red mask of the leaf

    # Going through each pixel of a large image can take a few seconds...
    print("[Hole detection] Calculating...")

    # Preparing to create an image of the leaf with blue background instead of black
    blue_bg = np.zeros_like(red_leaf)
    height, width, _ = blue_bg.shape
    for r in range(height):
        for c in range(width):
            if red_leaf[r][c][0] == 0 and red_leaf[r][c][1] == 0 and red_leaf[r][c][2] == 255:  # If pixel inside leaf contour
                blue_bg[r][c] = img[r][c]                                                       #   Restore to original leaf image (including holes)
            else:                            # Else: Outside leaf contour = background
                blue_bg[r][c] = (255, 0, 0)  #   Change to blue

    # The image is now the original leaf (incl. holes) with a blue background instead of black
    # Now the only thing that is black are the holes within the leaf

    # We will find contours of the black holes to show the user
    # First, convert to black & white
    NOT_BW_BLUE_LOWER_THRESHOLD = 28
    grayscale_img = cv2.cvtColor(blue_bg, cv2.COLOR_BGR2GRAY)
    # Since the holes themselves are black, we can't make a normal mask since the entire image will be black
    # Use an inverted mask instead
    _, binary_img = cv2.threshold(grayscale_img, NOT_BW_BLUE_LOWER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find contours (= holes)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = list(filter(lambda ct: cv2.contourArea(ct) > 10, contours))  # Filter out the contours with too low of an area (probable noise))

    # Draw boxes around contours
    box_color = (0,0,255)
    orig = img.copy()
    holes = cv2.drawContours(orig, contours=filtered_contours, contourIdx=-1, color=box_color, thickness=10)  # Draws red boxes around holes
            
    cv2.imshow(name, holes)
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
    print(name)
    # checkBrownSpot(img, name)
    #checkYellowing(img, name, False)
    #checkDiscoloration(img, name)
    checkHoles(img, name)
