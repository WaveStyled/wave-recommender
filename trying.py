import requests  ## DUMMY TESTER CLASS

import requests
from random import randint
import pandas as pd
#from Wardrobe import Wardrobe
import itertools
from Wardrobe import Wardrobe
#from PIL import Image
data = pd.read_csv('./good_matts_wardrobe.csv') 
url = "http://localhost:5000/add"


def post_csv(w):
    for row in data.itertuples(index=False):
        d = list(row)
        d.insert(3, None)
        d.insert(4, 1)
        d.insert(5, 0.5)
        d.append(0)
        # print(d)
        w.addItem(d)
                #w.addItem(row)
		# r = requests.post(url,json=myobj)
    return w

w = Wardrobe()
post_csv(w)
print(w.getdf())


fit = w.gen_random("casual", "hot")
print(fit)

# post_csv()

# response = requests.get(url="http://localhost:5001/wardrobedata")
# print(response.text)