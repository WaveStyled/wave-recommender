import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.put("/ping")
async def update(item: dict):
    success = 200 if item else 404
    
    return item

@app.get("/recommend")
async def recommend():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)