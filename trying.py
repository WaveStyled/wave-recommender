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



def post_csv():

	for row in data.itertuples(index=False):
		myobj = {'COLOR': row.color,
				'TYPE': row.type,
				'PIECEID': row.pieceid,
				'OC_SEMI_FORMAL': row._4,
				'OC_FORMAL': row.formal,
				'OC_CASUAL': row.casual,
				'OC_WORKOUT': row.workout,
				'OC_OUTDOORS': row.outdoors,
				'OC_COMFY': row.comfy,
				'WE_HOT': row.hot,
				'WE_COLD': row.cold,
				'WE_RAINY': row.rainy,
				'WE_SNOWY': row.snowy,
				'WE_AVERAGE': row.typical
								}
		r = requests.post(url,json=myobj)

['pieceid', "color", "type", "recent_date_worn", "times_worn", "rating", 
                                        "oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors",
                                        "oc_comfy", "we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical", "dirty"]

w = Wardrobe()
input = [1, "red", "BRUH", None, 2, 0.5, 0, 1, 1, 1, 1, 2, 3, 4, 2, 1, 1, 1]
w.addItem(input)
x = w.getWardrobe()
print(x)

# post_csv()

# response = requests.get(url="http://localhost:5001/wardrobedata")
# print(response.text)