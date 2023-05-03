import cv2
import glob
import numpy as np
import os 
import sys

# constants
# if not commented, the range is in hsv 
BROWN_LOWER = (0, 20, 0)  
BROWN_UPPER = (100, 255, 240)
GREEN_LOWER = (25, 10, 0)
GREEN_UPPER = (86, 255, 240)
GREY_LOWER = (0, 0, 70)
GREY_UPPER = (180, 20, 240)
# these yellows are for analyzing brown spots 
YELLOW_LOWER1 = (0, 180, 170)
YELLOW_UPPER1 = (30, 255, 255)
YELLOW_LOWER2 = (15, 20, 170)
YELLOW_UPPER2 = (30, 126, 255)
# YELLOW_LOWER1 = (0, 107, 127)
# YELLOW_UPPER1 = (30, 255, 255)
# YELLOW_LOWER2 = (15, 0, 0)
# YELLOW_UPPER2 = (30, 126, 126)
# these yellows are for analyzing yellowing 
YELLOW_LOWER3 = (15, 127, 127)
YELLOW_UPPER3 = (30, 255, 255)
# this is used to reduce glare 
GLARE_LOWER = (240, 240, 240) #bgr
GLARE_UPPER = (255, 255, 255) #bgr
BLUE_LOWER1 = (85, 20, 90) 
BLUE_UPPER1 = (138, 255, 255) 
BLUE_LOWER2 = (85, 0, 0) 
BLUE_UPPER2 = (138, 126, 126) 
WHITE_LOWER = (200, 200, 200) #bgr
WHITE_UPPER = (255, 255, 255) #bgr
BLACK_LOWER = (0, 0, 0) #bgr
BLACK_UPPER = (20, 20, 20) #bgr
# constant to determine which brown spots are large enough to be important 
AREA_MIN = 300 

CURRENT_SA = 0 # global variable for the current surface area 

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
def filterGrey(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_h = cv2.inRange(h, 0, 180)
    mask_s = cv2.inRange(s, 0, 55)
    mask_v = cv2.inRange(v, 0, 240)
    res = cv2.bitwise_and(mask_h, mask_s)
    res2 = cv2.bitwise_and(res, mask_v)
    not_res = cv2.bitwise_not(res2)
    res = cv2.bitwise_or(hsv, hsv, mask=not_res)
    return cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
 
'''Function that filters out the white regions'''
def filterWhite(rgb):
    # Check for anything that is not white (or black/background)
    # Create mask for white regions
    white_mask = cv2.inRange(rgb, WHITE_LOWER, WHITE_UPPER)
    no_white = cv2.bitwise_not(white_mask)
    res = cv2.bitwise_or(rgb, rgb, mask=no_white)
    return res


'''Function that filters out the blue regions'''
def filterBlue(rgb):
    # Check for anything that is not blue (or black/background)
    # we need to slice blue twice because it is a parabolic slice on the hsv color space
    # Create mask for blue regions
    frame_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
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
    return cv2.cvtColor(not_blue, cv2.COLOR_HSV2BGR)


'''This function detects the contour of the leaf'''
def detectContour(original_img):  
    original_img = cv2.blur(original_img, (5, 5))
    dilated = cv2.dilate(original_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    cont_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    return c 


'''This function detects where the brown regions are relative to the leaf'''
def detectlocation(brown_img, name, original_img, sentences):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(brown_img, cv2.MORPH_CLOSE, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    contour_leaf = detectContour(original_img)
    x,y,w,h = cv2.boundingRect(contour_leaf)
    #cv2.drawContours(original_img, conts, -1, (0,255,0), 3)
    d = {"upper left": 0, "lower left": 0, "upper right": 0, "lower right": 0}
    for cnt in conts:
        area = cv2.contourArea(cnt)
        #filter more noise
        if area > AREA_MIN: 
            cv2.drawContours(original_img, cnt, -1, (0,0,255), 3)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # check where the center of mass is positioned regarding the entire contour of leaf
                if cx < (x+w/2): 
                    if cy < (y+h/2): 
                        d["upper left"] += area 
                    else: 
                        d["lower left"] += area
                else: 
                    if cy < (y+h/2): 
                        d["upper right"] += area
                    else: 
                        d["lower right"] += area
    if d["upper left"] == 0 and d["lower left"] == 0 and d["upper right"] == 0 and d["lower right"] == 0:
        sentences.append("\tNo brown spot significant enough is detected")
    else:
        most_freq = max(d, key = d.get)
        sentences.append("\tThe brown spots are mostly in the "+most_freq+" corner.")
    # cv2.imshow(name, original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return original_img


'''This function filters out pixels that are almost black'''
def turnblack(rgb):
    black_mask = cv2.inRange(rgb, BLACK_LOWER, BLACK_UPPER)
    no_black= cv2.bitwise_not(black_mask)
    res = cv2.bitwise_or(rgb, rgb, mask=no_black)
    return res


'''algorithm for brown spot -- daisy'''
def checkBrownSpot(img, name):
    # calculate surface area of the leaf
    # by subtracting the number of black pixels from the entire image
    global CURRENT_SA
    blurred = blurEdge(img, name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    CURRENT_SA = cv2.countNonZero(gray)
    # reduce the glare
    glare_reduced = reduceGlare(blurred, name[:-4])
    # filter green, grey, blue from images
    no_green = filterGreen(glare_reduced)
    not_yellow = filterYellow(no_green)
    no_white = filterWhite(not_yellow)
    no_grey = filterGrey(no_white)
    no_blue = filterBlue(no_grey)
    # turn pixels that are almost black into completely black 
    change_to_black = turnblack(no_blue)
    # brown color
    # Create the brown HSV mask
    hsv = cv2.cvtColor(change_to_black, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    brown = cv2.bitwise_and(hsv, hsv, mask=brown_mask)
    new_brown = cv2.cvtColor(brown, cv2.COLOR_HSV2BGR)
    # horiz = np.concatenate(
    #     (img, no_green, not_yellow, no_white, no_grey, no_blue, new_brown), axis=1)
    # cv2.imshow(name+"+ xgree + xyellow + xwhite + xgrey + xblue + brown", horiz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(new_brown, cv2.COLOR_BGR2GRAY)
    brown_count = cv2.countNonZero(gray)
    ratio = brown_count/CURRENT_SA
    sentences = []
    sentences.append(f"\t[Brown Spot] {brown_count:,} brown pixels / {CURRENT_SA:,} total pixels = {round(ratio,3) * 100}% brown")
    score = 0
    if ratio < 0.1: 
        sentences.append("\t[Brown Spot Score] Healthy -- Score 5")
        score = 5
    elif ratio < 0.2: 
        sentences.append("\t[Brown Spot Score] Relatively Healthy -- Score 4")
        score = 4
    elif ratio < 0.3: 
        sentences.append("\t[Brown Spot Score] Okay -- Score 3")
        score = 3
    elif ratio < 0.4: 
        sentences.append("\t[Brown Spot Score] Relatively Unhealthy -- Score 2")
        score = 2
    else:
        sentences.append("\t[Brown Spot Score] Unhealthy -- Score 1")
        score = 1
    final_img = detectlocation(new_brown, name, img, sentences)
    cv2.imwrite(name[:-4]+"brownspot.png", final_img)
    return (name[:-4]+"brownspot.png", sentences, score)


'''algorithm for yellowing of leaf -- daisy'''
def checkYellowing(img, name):
    # calculate surface area of the leaf
    # by subtracting the number of black pixels from the entire image
    blurred = blurEdge(img, name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reduce the glare
    glare_reduced = reduceGlare(blurred, name[:-4])
    # filter out the green
    no_green = filterGreen(glare_reduced)
    # only look at the yellow 
    # Convert to HSV space
    frame_hsv = cv2.cvtColor(no_green, cv2.COLOR_BGR2HSV)
    # Create mask for yellow regions
    mask = cv2.inRange(frame_hsv, YELLOW_LOWER3, YELLOW_UPPER3)
    # yellow_mask2 = cv2.inRange(frame_hsv, YELLOW_LOWER4, YELLOW_UPPER4)
    # mask = cv2.bitwise_or(yellow_mask, yellow_mask2)
    # Find which pixels are yellow
    yellow_pos = mask > 0
    # Prepare new image matrix
    yellow = np.zeros_like(img, np.uint8)
    # Set each pixel
    yellow[yellow_pos] = no_green[yellow_pos]
    # horiz = np.concatenate(
    #     (img, yellow))
    # cv2.imshow(name, horiz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
    yellow_count = cv2.countNonZero(gray)
    ratio = yellow_count/CURRENT_SA
    sentences = []
    sentences.append(f"\t[Yellowing] {yellow_count:,} yellow pixels / {CURRENT_SA:,} total pixels = {round(ratio,3) * 100}% yellow")
    score = 0
    if ratio < 0.1: 
        sentences.append("\t[Yellowing Score] Healthy -- Score 5")
        score = 5
    elif ratio < 0.2: 
        sentences.append("\t[Yellowing Score] Relatively Healthy -- Score 4")
        score = 4
    elif ratio < 0.3: 
        sentences.append("\t[Yellowing Score] Okay -- Score 3")
        score = 3
    elif ratio < 0.4: 
        sentences.append("\t[Yellowing Score] Relatively Unhealthy -- Score 2")
        score = 2
    else:
        sentences.append("\t[Yellowing Score] Unhealthy -- Score 1")
        score = 1
    cv2.imwrite(name[:-4]+"yellowing.png", yellow)
    return (name[:-4]+"yellowing.png", sentences, score)
        

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
    img = turnblack(img)

    # Going through each pixel of a large image can take a few seconds...
    #print("[Discoloration] Calculating...")

    # Go through each pixel
    score = 0
    for r in range(height):
        for c in range(width):
            if any(img[r][c]):        # If pixel in image is not (0,0,0) = black
                pixels_leaf += 1
            if any(not_green[r][c]):  # If pixel under not-green mask is not (0,0,0) = black
                pixels_discolored += 1
                # Mark the discolored parts in red
                not_green[r][c] = (0, 0, 255)
    sentences = []
    # Print stats
    sentences.append(f"\t[Discoloration] {pixels_discolored:,} discolored pixels / {CURRENT_SA:,} total pixels = {pixels_discolored / CURRENT_SA * 100}% discolored")

    # Make a new showing discolored parts on left, original image on right
    # horiz = np.concatenate((not_green, img), axis=1)
    # cv2.imshow(name, horiz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ratio = pixels_discolored / CURRENT_SA
    if ratio < 0.1: 
        sentences.append("\t[Discoloration Score] Healthy -- Score 5")
        score = 5
    elif ratio < 0.2: 
        sentences.append("\t[Discoloration Score] Relatively Healthy -- Score 4")
        score = 4
    elif ratio < 0.3: 
        sentences.append("\t[Discoloration Score] Okay -- Score 3")
        score = 3
    elif ratio < 0.4: 
        sentences.append("\t[Discoloration Score] Relatively Unhealthy -- Score 2")
        score = 2
    else:
        sentences.append("\t[Discoloration Score] Unhealthy -- Score 1")
        score = 1
    cv2.imwrite(name[:-4]+"discoloration.png", not_green)
    return (name[:-4]+"discoloration.png", sentences, score)

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
    hole_pixels = 0
    for cnt in filtered_contours:
        hole_pixels += cv2.contourArea(cnt)
    # cv2.imshow(name, holes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sentences = []
    score = 0
    # Print stats
    sentences.append(f"\t[Holes] {hole_pixels:,} pixels / {CURRENT_SA:,} total pixels = {hole_pixels / CURRENT_SA * 100}% holes")
    ratio = hole_pixels / CURRENT_SA
    if ratio * 100 < 0.2: 
        sentences.append("\t[Hole Detection Score] Healthy -- Score 5")
        score = 5
    elif ratio * 100 < 0.6: 
        sentences.append("\t[Hole Detection Score] Relatively Healthy -- Score 4")
        score = 4
    elif ratio * 100 < 0.8: 
        sentences.append("\t[Hole Detection Score] Okay -- Score 3")
        score = 3
    elif ratio * 100 < 5: 
        sentences.append("\t[Hole Detection Score] Relatively Unhealthy -- Score 2")
        score = 2
    else:
        sentences.append("\t[Hole Detection Score] Unhealthy -- Score 1")
        score = 1
    cv2.imwrite(name[:-4]+"hole.png", holes)
    return (name[:-4]+"hole.png", sentences, score)
        

'''The function that writes the result of the images to an html page'''
def writeToHtml(name, brown_spot_img, sent_brown, yellow, sent_yellow, discolor, sent_disc, hole, sent_hole):
    with open('result.html', 'a') as the_file:
        the_file.write('''<span style="font-weight: bolder">'''+name+'''</span>\n''')
        the_file.write('''<img src = "''' + name + '''" width = "200px" height = "200px">\n''')
        the_file.write('''<br>\n''')
        the_file.write('''<span style="font-weight: bolder">Brown Spot</span>\n''')
        the_file.write('''<img src = "''' + brown_spot_img + '''" width = "200px" height = "200px">\n''')
        the_file.write('''<br>\n''')
        for ele in sent_brown: 
            the_file.write('''<span> ''' + ele + '''</span>\n''')
            the_file.write('''<br>\n''')
        the_file.write('''<br>\n''')
        the_file.write('''<span style="font-weight: bolder">Yellowing</span>\n''')
        the_file.write('''<img src = "''' + yellow + '''" width = "200px" height = "200px">\n''')
        the_file.write('''<br>\n''')
        for ele in sent_yellow: 
            the_file.write('''<span> ''' + ele + '''</span>\n''')
            the_file.write('''<br>\n''')
        the_file.write('''<br>\n''')
        the_file.write('''<span style="font-weight: bolder">Discoloration</span>\n''')
        the_file.write('''<img src = "''' + discolor + '''" width = "200px" height = "200px">\n''')
        the_file.write('''<br>\n''')
        for ele in sent_disc: 
            the_file.write('''<span> ''' + ele + '''</span>\n''')
            the_file.write('''<br>\n''')
        the_file.write('''<br>\n''')
        the_file.write('''<span style="font-weight: bolder">Hole Detection</span>\n''')
        the_file.write('''<img src = "''' + hole + '''" width = "200px" height = "200px">\n''')
        the_file.write('''<br>\n''')
        for ele in sent_hole: 
            the_file.write('''<span> ''' + ele + '''</span>\n''')
            the_file.write('''<br>\n''')
        the_file.write('''<br>\n''')
        the_file.write('''<br>\n''')


'''main'''
# prepping the directory and cleaning old data 
if os.path.exists("result.html"):
    os.remove("result.html")
    
dir_name = "./library/"
dir = os.listdir(dir_name)

for item in dir:
    if item.endswith("brownspot.png") or item.endswith("discoloration.png") or item.endswith("yellowing.png") or item.endswith("hole.png"):
        os.remove(os.path.join(dir_name, item))

# check if we are doing command line argument or running the entire database
if len(sys.argv) > 1: 
    for name in sys.argv:
        if name == "main.py":
            continue
        img = cv2.imread(name)
        brown_spot_img, sent_brown, score_brown = checkBrownSpot(img.copy(), name)
        yellow, sent_yellow, score_yellow = checkYellowing(img.copy(), name)
        discolor, sent_disc, score_disc = checkDiscoloration(img, name)
        hole, sent_hole, score_hole = checkHoles(img, name)
        writeToHtml(name, brown_spot_img, sent_brown, yellow, sent_yellow, discolor, sent_disc, hole, sent_hole)

else:
    # load all images from the Library directory
    images = {}
    for filename in glob.glob("./library/*.png"):
        img = cv2.imread(filename)
        if img is not None:
            images[filename] = img

    for name, img in images.items():
        brown_spot_img, sent_brown, score_brown = checkBrownSpot(img.copy(), name)
        yellow, sent_yellow, score_yellow = checkYellowing(img.copy(), name)
        discolor, sent_disc, score_disc = checkDiscoloration(img, name)
        hole, sent_hole, score_hole = checkHoles(img, name)
        writeToHtml(name, brown_spot_img, sent_brown, yellow, sent_yellow, discolor, sent_disc, hole, sent_hole)
