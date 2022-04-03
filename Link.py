## RUN INSTALLATION python3 -m pip install fastapi uvicorn[standard]
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import Wardrobe

app = FastAPI()

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)
    wardrobe = Wardrobe()

class Row (BaseModel):
    data : tuple

@app.put("/ping")
async def update(item: dict):
    success = 200 if item else 404
    new_item = wardrobe.addItem(item["data"])
    return success

@app.delete("/ping")
async def delete_item(item: dict):
    pass

@app.get("/wardrdrobepython")
async def datastuf():
    
    return 

@app.get("/recommend")
async def recommend():
    pass

