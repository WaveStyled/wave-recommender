import requests
import cv2 as cv
import numpy as np

def main_loop():
    while True:
        x = input()
        if(x == "calibrate"):
            return
        if(x == "recommend"):
            return

# Calls server startup, loads the wardrobe csv
def bootup():
    # call bootup link
    r = requests.put("http://localhost:5001/start")

def calibrate_start():
    print("How many calibration outfits would you like to see?\n")
    print("Note: More you calibrate, more the model will understand your likes:\n")
    num_calibrate = 1
    r = requests.put("http://localhost:5001/calibrate_start/"+str(num_calibrate))
    fits = r.json()
    # display images
    ratings, fit, attr = displayFit(fits[0], fits[1], '../matts_wardrobe_jpeg')
    # send ratings back
    r = requests.put("http://localhost:5001/calibrate_end/",json=ratings)

def displayFit(outfit, conditions, path):
        ratings = []
        i = 0
        for im in outfit:
            if im:
                images = []
                overall_shape = None
                for item in im:
                    if item:
                        image = cv.imread(f'{path}/{item}.jpeg')
                        image = cv.resize(image, (0, 0), None, .1, .1)
                        if not overall_shape:
                            overall_shape = image.shape
                        elif image.shape != overall_shape: # adjust image if not oriented properly
                            image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
                        images.append(image)
                ims = np.hstack((images[0], images[1]))  # puts the images side-by-side
                for rest in images[2:]:
                    ims = np.hstack((ims,rest))
                cv.imshow(f'outfit: {str(im)}, attr: {str(conditions[i])}', ims)
                while True: # user inputs
                    like_dislike = cv.waitKey(0) & 0xFF - 48
                    print(like_dislike)
                    if not like_dislike or like_dislike == 1:
                        break
                    if like_dislike == 65:
                        cv.destroyAllWindows()
                        return ratings, outfit[:i], conditions[:i]
                ratings.append(like_dislike)
                cv.destroyAllWindows()  # if you want to remove window on key press
            i +=1
        cv.destroyAllWindows()
        return ratings, outfit, conditions

bootup()
calibrate_start()


#main_loop()