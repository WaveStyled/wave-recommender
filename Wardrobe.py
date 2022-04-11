##########
#
# Represents a Wardrobe, which is a collection of Items
#
##########
from Item import Item
import pandas as pd
from random import randint
import cv2 as cv
import numpy as np

class Wardrobe:

    def __init__ (self):
        self.dt = pd.DataFrame(columns= ['pieceid', "type", "color", "recent_date_worn", "times_worn", "rating", 
            "oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors",
            "oc_comfy", "we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical", "dirty"])
    
    def from_csv(self, path):
        self.dt = pd.read_csv(path)
    
    def addItem(self, clothing_item): # index 4 is the type and its just checking the last letter
        self.dt.loc[self.dt.shape[0]] = clothing_item
    
    def getItem(self, primary_key):     
        item = self.dt.loc[self.dt['pieceid'] == primary_key].to_records(index=False)  
        return item[0] if item else None
    
    def getItemObj(self, primary_key):
        return Item(self.getItem(primary_key))

    def deleteItem(self, primary_key):
        self.dt.drop([primary_key-1], axis=0, inplace=True)

    def getWardrobe(self):
        return self.dt.to_records(index=False)
    
    def filter(self, sub_attr, attribute="type"):
        return self.dt.loc[self.dt[attribute] == sub_attr].to_records()
    
    def getdf(self):
        return self.dt

    def gen(self, occasion, weather, ends):
        x = self.dt.loc[(self.dt["type"].str.endswith(ends)) & (self.dt["oc_"+occasion] == 1) & (self.dt["we_"+weather] == 1) ]
        if (len(x.index)==0):
            return -1
        chosen = x.sample()
        return int(chosen["pieceid"])

    def gen_random(self, occasion, weather):
        top = ""
        shorts = ""
        shoes = ""
        under =""
        bot = ""
        hat = ""
        fit = [0,0,0,0,0,0,0]
        if (weather == "hot"):
            hat_chance = randint(1,3)
            
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            top = self.gen(occasion,weather,"S")
            fit[1] = top
            bot = self.gen(occasion,weather,"H")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "cold"):
            hat_chance = randint(1,3)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            undershirt_chance = randint(1,4)
            if(undershirt_chance == 1):
                under =	self.gen(occasion,weather,"S")
                fit[1] = under
            top =  self.gen(occasion,weather,"T")
            fit[2] = top
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "rainy"):
            
            hat_chance = randint(1,2)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            shirt_or_sweat = randint(1,4)
            if(shirt_or_sweat == 1):
                # shirt
                top = self.gen(occasion,weather,"S")
                fit[1] = top
            else:
                top =  self.gen(occasion,weather,"T")
                fit[2] = top
            
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
            jacket = self.gen(occasion, weather,"C")
            fit[3] = jacket
        elif (weather == "typical"):
            shirt_or_sweat = randint(1,2)
            if(shirt_or_sweat == 1):
                # shirt
                top = self.gen(occasion,weather,"S")
                fit[1] = top
            else:
                top =  self.gen(occasion,weather,"T")
                fit[2] = top
            
            shorts_or_pants =randint(1,2)

            if(shorts_or_pants == 1):
                bot = self.gen(occasion,weather,"H")
                fit[4] = bot
            else:
                bot = self.gen(occasion,weather,"P")
                fit[4] = bot
            
            hat_chance = randint(1,4)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "snowy"):
            hat_chance = randint(1,2)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            
            top =  self.gen(occasion,weather,"T")
            fit[1] = top
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
            jacket = self.gen(occasion, weather,"C")
            fit[3] = jacket
        return fit

    def getRandomFit(self, num_fits=1):
        occasions = ["formal","semi_formal","casual","workout","outdoors","comfy"]
        weather = ["hot","cold","rainy","snowy","typical"]
        fits = []
        oc_we = []
        for _ in range(0,num_fits,1):
            oc = occasions[randint(0,len(occasions)-1)]
            we = weather[randint(0,len(weather)-1)]
            fit = self.gen_random(oc,we)
            
            if -1 not in fit:
                fits.append(fit)
                oc_we.append([oc,we])
        
        #fits.sort()
        #list(fits for fits,_ in itertools.groupby(fits))
        return [fits,oc_we]

    def displayFit(self, outfit, conditions, path):
        ratings = []
        i = 0
        for im in outfit:
            if im:
                images = []
                overall_shape = None
                for item in im:
                    if item:
                        image = cv.imread(f'{path}/{item}.jpeg')
                        image = cv.resize(image, (0, 0), None, .20, .20)
                        if not overall_shape:
                            overall_shape = image.shape
                        elif image.shape != overall_shape:
                            image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
                        images.append(image)
                ims = np.hstack((images[0], images[1]))
                for rest in images[2:]:
                    ims = np.hstack((ims,rest))
                cv.imshow(f'outfit: {str(im)}, attr: {str(conditions[i])}', ims)
                while True:
                    like_dislike = cv.waitKey(0) & 0xFF - 48
                    if not like_dislike or like_dislike == 1:
                        break
                ratings.append(like_dislike)
                # cv.destroyAllWindows()  if you want to remove window on key press
        i +=1
        cv.destroyAllWindows()
        return ratings


    def __getitem__ (self, clothing_type):  ## allows for [] notation with the object
        return self.dt.loc[(self.dt["type"].str.endswith(clothing_type))].to_records()

    def __str__ (self):
        return str(self.dt)

    def __del__ (self):
        print("Wardrobe eliminated")