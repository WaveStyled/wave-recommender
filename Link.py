##########
#
# REST API that runs the Python server connection
#
# INSTALL: python3 -m pip install fastapi uvicorn[standard]
#
# See Wardrobe class for any further installation commands if necessary
##########

# Library imports
from cv2 import fitEllipse
import uvicorn
import sys
from fastapi import FastAPI
from typing import Optional, List
from Wardrobe import Wardrobe
from Recommender import Recommender

# Creates app and wardobe instance
app = FastAPI()
wardrobe = Wardrobe()
recommender = Recommender()


@app.put("/start")
async def boot():
    if(wardrobe.logged_in == False):
        wardrobe.from_csv("./good_matts_wardrobe.csv")
        wardrobe.logIn()
        return 200
    else:
        return 404

"""
Function: 
Python server - add item

Desc: 
When an item has been added to the node server, node pings python and tells it to add the
new item to pythons item object.

Inputs:
 - Dictionary of the new items attributes
 - UserID(default none) : Not implemented fully

Returns:
 - 200 or 404 to the Node server
"""
@app.put("/add")
async def update(item: dict, userid: Optional[int] = None):
    # If item exists 200, otherwise 404
    success = 200 if item else 404
    
    # Adds item to wardrobe object
    wardrobe.addItem(item.get("data"))
    
    # returns success message(200 or 404)
    return success


"""
Function: 
Python server - delete item: INCOMPLETE

Desc: 
When an item has been deleted from the app, node removes from the SQL db and pings the python server. 
Python removes the item from the wardrobe object.


Inputs:
 - PieceID as a dictionary object
 - UserID(default none) : Not implemented fully

Outputs:
 - 200 or 404 to Node server
"""
@app.put("/delete/")
async def delete(item: dict, userid: Optional[int] = None):
    # If item exists 200, otherwise 404
    success = 200 if item else 404
    #TODO: Logic for deleteing item
    
    # returns success message(200 or 404)
    return success

"""
Function: 
Python server - recommend: INCOMPLETE

Desc: 
When a user wants to get an outfit, Node will ping the server to send some outfits back to the server in the form of PieceID's

Inputs:
 - Occasion, Weather

Outputs:
 - List of different outfits(pieceIDs)
"""
@app.post("/create_recommender/")
async def recommend(userid : Optional[int] = None):
    recommender.fromDB()
    return 200

@app.get("/recommender_train/")
async def recommend(userid : Optional[int] = None):
    #print(wardrobe)
    recommender.addColors(wardrobe)
    recommender.encode_colors()
    recommender.normalize()

    # training
    train, labels = recommender.create_train()
    recommender.buildModel()
    recommender.train(train,labels)
    return 200

@app.get("/recommend/")
async def recommend(occasion : str, weather : str, userid : Optional[int] = None):
    #recommender.fromDB()
    # run prediction
    print(occasion, weather)
    return 200

"""
Function: 
Python server - calibrate_start: COMPLETE

Desc: 
User chooses to calibrate, node pings python to generate some random outfits from the users DB

Inputs:
 - None

Outputs:
 - List of different randomly generated outfits(pieceIDs)
"""
@app.put("/calibrate_start/{num_calibrate}")
async def calibrate_start(num_calibrate: int):
    fits = wardrobe.getRandomFit(num_calibrate)
    return fits

@app.put("/calibrate_end/")
async def calibrate_end(metadata: list):
    wardrobe.outfitToDB(ratings=metadata[0],outfits=metadata[1],attrs=metadata[2])
    return 200

"""
Function: 
Python server - get_wardrobe

Desc: 
Returns wardrobe object as a dictionary

Inputs:
 - None

Outputs:
 - Dictionary of wardrobe
"""
@app.get("/getwardrobe")
async def getwardrobe():
    # Returns dict str()
    return {"data": str(wardrobe)}

"""
Function: 
Python server - kill_server: INCOMPLETE

Desc: 
Kills server Maybe?

Inputs:
 - None

Outputs:
 - None
"""
@app.get("/end")
async def killServer():

    
    def signal_handler(sig, frame):  # only runs when the user actually does the ctrl + C
        print('You pressed Ctrl+C!')
        
    print("Node server has triggered shutdown")
    #del wardrobe
    # sys.exit(0)
    return 1

# On bootup start the server on Localhost at port 5001 
if __name__ == '__main__':
    model = None
    uvicorn.run(app, host='127.0.0.1', port=5001)