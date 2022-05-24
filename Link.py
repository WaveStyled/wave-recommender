##########
#
# REST API that runs the Python server connection
#
# INSTALL: python3 -m pip install fastapi uvicorn[standard]
#
# See Wardrobe class for any further installation commands if necessary
##########

# Library imports
from pydantic import DataclassTypeError
import uvicorn
import sys
from fastapi import FastAPI
from typing import Optional, List
from User import UserBase


# Creates app and wardobe instance
app = FastAPI()
USERBASE = UserBase()

@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
def shutdown_event():
    #recommender.savemodel('') # how would we know which model to close?
    pass

@app.put("/start/")
async def boot(userid: Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    return 200
    # if user.wardrobe_init("./csv_files/good_matts_wardrobe.csv"):
    #     #print(user.getWD())
    #     return 200
    # else:
    #     return 404
    

"""
Function: 
Python server - add item

Desc: 
When an item has been added to the node server, node pings python and tells it to add the
new item to pythons item object.

Inputs:
 - Dictionary of the new items attributes
 - UserID(default 999) : Id of the user

Returns:
 - 200 or 404 to the Node server
"""
@app.put("/add")
async def update(item: dict, userid: Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    # If item exists and is successfully added 200, otherwise 404
    success = 200 if (item is not None and user.addWDItem(item.get("data"))) else 404
    return success

"""
Function:
Python server -- change Item

Desc:
Takes in an item representation and updates that corresponding item in the Python
Dataframe (if it exists). Handles Wardrobe updating on the Python end

Inputs: 
- Dictionary containing a list of the updated state of the item
- UserID (default 999)

Returns: 
 - 200 (need to implement erorr handling)
"""
@app.post("/change/")
async def change(data : dict, userid : Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    tochange = data['data']
    user.updateWDItem(tochange)
    return 200


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
@app.delete("/delete/")
async def delete(id: int, userid: Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    user.removeWDItem(id)
    return 200

"""
Function: 
Python server - recommen train

Desc: 
Loads the user's recommendation model and updates the model's
predictions based on the new data

Inputs:
 - retrain BOOL (Default True) --> whether the model should be retrained
 - userid (Default 999) --> id of the User

Outputs:
 - 200 if successful otherwise 404
"""
@app.post("/recommend_train/")
async def recommend_train(retrain : bool = True, userid : Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    user.load_model()
    user.update_preferences(new_data=True, train_again=True) # when buffer set train_again to True
    return 200

"""
Function:
Python server - recomend

Desc:
For a given occasion and weather, generates recommendations for the specified
user

Inputs: (all query parameters)
 - occasion STR --> the desired occasion
 - weather STR --> the desired weather
 - userid (Default 999) --> id of the User

Returns:
List of tuples that represent the recommended fits for the user
"""
@app.get("/recommend/")  ## query parameters done in the link itself (see sim-ui recommend() for examples)
async def recommend(occasion : str, weather : str, userid : Optional[str] = "999"):
    recommendations = USERBASE.get_user(userid).get_recommendations(occasion=occasion, weather=weather)
    print(recommendations)
    return recommendations
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
@app.put("/calibrate_start/")
async def calibrate_start(num_calibrate: int, userid : Optional[str] = "999"):
    user = USERBASE.get_user(userid)
    fits = user.begin_calibration(num_calibrate)
    return fits

"""
Function:
Python Server - End the calibrate phase

Desc:
Ends the calibration phase, which involves sending outfits to the Outfits DB
and initializing the recommender/model

Inputs: 
 - data DICTIONARY --> holds the ratings, outfits, and occasion/weather tuples from the
    calibration
 - userid (Default 999) --> id of the User

Returns:
 - 200 if operation is successful otherwise 404
"""
@app.put("/calibrate_end/")
async def calibrate_end(data: dict, userid : Optional[str] = "999"):
    metadata = data['data']
    print(metadata, userid)
    USERBASE.get_user(userid).end_calibration(ratings=metadata[0],outfits=metadata[1],attrs=metadata[2])
    return 200

"""
Function: 
Python Server -- set outfit of the day (OOTD)

Desc:
Sets the OOTD for the specified user

Inputs:
 - data DICTIONARY --> 'outfit' key contains the OOTD fit, 'weather' contains the weather,
    'occasion' contains the occasion string, optional key 'date' for the date
 - userid (Default 999) --> id of the User

Returns
 - 200 if no errors happen otherwise 404
"""
@app.put("/OOTD/")
async def add_ootd(data : dict, userid : Optional[str] = "999"):
    outfit = data['outfit']
    weather = data['weather']
    occasion = data['occasion']
    user = USERBASE.get_user(userid)
    user.chooseOOTD(outfit, weather, occasion)
    return 200


"""
Function:
Python Server -- Get Outfit of the Day (OOTD)

Desc:
returns the OOTD fit for the desired user, weather, and occasion

Inputs;
 - data DICTIONARY --> holds the weather, occasion, date metadata
 - UserID INT (default 999) --> id of the user

Returns:
 - LIST --> fit of the day (empty if none exists)
"""
@app.get("/OOTD/")
async def get_ootd(userid : Optional[str] = "999"):
    data = {}
    weather = data.get('weather', "")
    occasion = data.get('occasion', "")
    date = data.get('date', "")
    user = USERBASE.get_user(userid)
    ootd = user.getOOTD(weather, occasion, date)
    print(ootd)
    return ootd


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
@app.get("/getwardrobe/")
async def getwardrobe(userid : Optional[str] = "999"):
    # Returns dict str()
    return {"data": str(USERBASE.get_user(userid).getWD())}

"""
Function:
Python server - get user info

Desc:
Returns the wardrobe, OOTD, pieceids, outfit table of all the users
in the USERBASE

Inputs: None

Returns: None
"""
@app.get("/user_info/")
async def getinfo():
    return {"data": str(USERBASE)}


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

