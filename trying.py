import requests  ## DUMMY TESTER CLASS

import requests
from random import randint
import pandas as pd
#from Wardrobe import Wardrobe
import itertools
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

post_csv()




response = requests.get(url="http://localhost:5001/wardrobedata")
print(response.text)