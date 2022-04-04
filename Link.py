import uvicorn
from fastapi import FastAPI
from Wardrobe import Wardrobe
from Recommender import Recommender
import sys 

app = FastAPI()

@app.put("/ping")
async def update(item: dict):
    success = 200 if item else 404
    wardrobe.addItem(item.get("data"))
    #if model:
    #    model.update(item)
    # model.update(item)
    return success

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

@app.get("/kill")
async def killServer():
    sys.stdout.write("Dead")
    del wardrobe
    exit()

if __name__ == '__main__':
    wardrobe = Wardrobe()
    model = None
    uvicorn.run(app, host='127.0.0.1', port=5001)
