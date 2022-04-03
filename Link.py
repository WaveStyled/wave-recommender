## RUN INSTALLATION python3 -m pip install fastapi uvicorn[standard]
import psycopg2 as psqldb
import uvicorn
from fastapi import FastAPI

HOSTNAME = 'localhost'
DATABASE  = 'wavestyled'
USER = 'postgres'
PASS = 'cse115'
PORT = 5432

app = FastAPI()

@app.put("/ping")
async def update(item: dict):
    primary_key = item.get("PK")

    success = 200
    try: 
        with psqldb.connect(
                host = HOSTNAME,
                dbname = DATABASE,
                user = USER,
                password = PASS,
                port = PORT) as conn:

            with conn.cursor(cursor_factory=psqldb.extras.DictCursor) as curs:
                query = "SELECT * FROM Wardrobe WHERE PIECEID = {}".format(primary_key)
                curs.execute(query)
                tuple = curs.fetchall()[0]
                conn.commit() ## save transactions into the database
    except Exception as error:
        success = 404
    finally:
        if conn:
            conn.close()   ## close the connection -- wraps each SQL call

    return {"success":success, "tuple":tuple}

@app.get("/recommend")
async def recommend():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)