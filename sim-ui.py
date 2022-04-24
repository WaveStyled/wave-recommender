from anyio import ConditionStatistics
import requests
import cv2 as cv
import numpy as np

def main():
    bootup(1000)
    print("WAVESTYLED UI SIMULATION")
    # choice = int(input("Start? (1/0): "))
    # #if choice == 1:
    # #    calibrate(1000)
    r = requests.delete("http://localhost:5001/delete/?userid=1000", json={"PK": 150})
    print(r.text)
        
        #train_recommender(1000)
        #oc_mappings = ["oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors", "oc_comfy"]  ## maps occasion to integer (id)
        #we_mappings = ["we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical"]
        #while True:          
        #print("RECOMMENDATIONS:")
        #occasion = int(input("What occasion? (formal (0), semi_formal (1), casual (2), workout (3), outdoors (4), comfy (5) ): "))
        #weather = int(input("What occasion? (cold (0), hot (1), rainy (2), snowy (3), typical (4) ): "))
        #quit = 0
        #fits = recommend(f'{oc_mappings[occasion]}', f'{we_mappings[weather]}', 1000) ## display fits
        #for f in fits:
        #    print(f)

    # bootup(47)
    # print("WAVESTYLED UI SIMULATION ROUND 2 MFFFFFFF")
    # choice = int(input("Start? (1/0): "))
    # if choice == 1:
    #     choice2 = int(input("Calibrate Model? (1/0): "))
    #     if choice2 == 1:
    #         calibrate(47)
    #     train_recommender(47)
    #     oc_mappings = ["oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors", "oc_comfy"]  ## maps occasion to integer (id)
    #     we_mappings = ["we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical"]
    #     #while True:          
    #     print("RECOMMENDATIONS:")
    #     occasion = int(input("What occasion? (formal (0), semi_formal (1), casual (2), workout (3), outdoors (4), comfy (5) ): "))
    #     weather = int(input("What occasion? (cold (0), hot (1), rainy (2), snowy (3), typical (4) ): "))
    #     quit = 0
    #     fits = recommend(f'{oc_mappings[occasion]}', f'{we_mappings[weather]}', 47) ## display fits
    #     if fits:
    #         for f in fits:
    #             print(f)   

    print("NOW LETS SEE IF THIS WORKED")
    r = requests.get("http://localhost:5001/user_info/")
    print(r.json()['data'])
    
            

# Calls server startup, loads the wardrobe csv
def bootup(u = 1000):
    # call bootup link
    requests.put(f"http://localhost:5001/start/?userid={u}")

def calibrate(u = 1000):
    print("How many calibration outfits would you like to see?\n")
    print("Note: More you calibrate, more the model will understand your likes:\n")
    num_calibrate = int(input('Number: '))
    r = requests.put(f"http://localhost:5001/calibrate_start/?num_calibrate={num_calibrate}&userid={u}")
    fits, conditions = r.json()

    # display images
    ratings, fit, attr = displayFit(fits, conditions, '../matts_wardrobe_jpeg')
    cv.destroyAllWindows()  # if you want to remove window on key press
    # print(ratings, fit, attr)

    # send ratings back
    requests.put(f"http://localhost:5001/calibrate_end/?userid={u}", json=[ratings, fit, attr])

def train_recommender(u = 1000):
    print("CALIBRATING RECOMMENDER MODEL...\n")
    requests.post(f"http://localhost:5001/recommend_train/?userid={u}")

def recommend(occasion, weather, u = 1000):
    r = requests.get(f"http://localhost:5001/recommend/?userid={u}&occasion={occasion}&weather={weather}")
    return r.json()


#def getColorName(R,G,B):
#    colors = pd.DataFrame()
#    df = colors[["R", "G", "B"]]
#    dist = lambda r,g,b : abs(R - r) + abs(B - b) + abs(G - g)
#    distances = np.array([dist(color.R, color.G, color.B) for color in df.itertuples()])
#    return colors.iloc[np.argmin(distances), 'color_name']

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


if __name__ == '__main__':
    main()



#main_loop()