import uvicorn
from fastapi import FastAPI
from Wardrobe import Wardrobe
import sys 


app = FastAPI()
wardrobe = Wardrobe()

@app.put("/ping")
async def update(item: dict):
    success = 200 if item else 404
    wardrobe.addItem(item.get("data"))
    return success

@app.get("/recommend")
async def recommend():
    pass ## pass wardrobe into the Recommender

@app.get("/wardrobedata")
async def getwardrobe():
    return {"wardrobe": str(wardrobe.getItem(34))}

@app.get("/kill")
async def killServer():
    sys.stdout.write("Dead")
    exit()

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)
