## RUN INSTALLATION python3 -m pip install fastapi uvicorn[standard]
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

class DummyItem (BaseModel):
    update: int

app = FastAPI()

@app.put("/ping/")
async def update(item: DummyItem):
    item.update({"response": "what's up"})
    return item

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)