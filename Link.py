from sre_constants import SUCCESS
import uvicorn
from fastapi import FastAPI
from typing import Optional
from Wardrobe import Wardrobe
from Recommender import Recommender
import signal

app = FastAPI()

@app.put("/add/")
async def update(item: dict, userid: Optional[int] = None):
    success = 200 if item else 404
    wardrobe.addItem(item.get("data"))
    #if model:
    #    model.update(item)
    # model.update(item)
    return userid, item.get("data")

@app.put("/delete/")
async def delete(item: dict, userid: Optional[int] = None):
    success = 200 if item else 404
    #TODO: Logic for deleteing item
    return userid

@app.get("/recommend")
async def recommend():
    # model.recommend(occaison, weather, data)
    pass ## pass wardrobe into the Recommender

@app.get("/screening_phase")
async def begin():
    model = Recommender(wardrobe)
    return 200

@app.get("/wardrobedata")
async def getwardrobe():
    return str(wardrobe)

@app.get("/shutdown")
async def killServer():
    print("Node server has triggered shutdown")
    del wardrobe
    signal.signal(signal.SIGINT)
    return 1

if __name__ == '__main__':
    wardrobe = Wardrobe()
    model = None
    uvicorn.run(app, host='127.0.0.1', port=5001)
