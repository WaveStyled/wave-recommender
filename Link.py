import signal
import uvicorn
import sys
from fastapi import FastAPI
from typing import Optional
from Wardrobe import Wardrobe
from Item import Item
from Recommender import Recommender

app = FastAPI()
global wardrobe
wardrobe = Wardrobe()

@app.put("/add")
async def update(item: dict, userid: Optional[int] = None):
    success = 200 if item else 404
    wardrobe.addItem(item.get("data"))
    #if model:
    #    model.update(item)
    # model.update(item)
    return item.get("data")

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
    return wardrobe.getItem(42)

@app.get("/end")
async def killServer():

    
    def signal_handler(sig, frame):  # only runs when the user actually does the ctrl + C
        print('You pressed Ctrl+C!')
        
    print("Node server has triggered shutdown")
    #del wardrobe
    signal.signal(signal.SIGINT, signal_handler)
    # sys.exit(0)
    return 1

@app.on_event("startup")
def start_event():
    print("App Starting and initializing Fields....")

@app.on_event("shutdown")  # just need to find a way to trigger this using the node
def shutdown_event():
    print("shutting down....")

if __name__ == '__main__':
    model = None
    uvicorn.run(app, host='127.0.0.1', port=5001)