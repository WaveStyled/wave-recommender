## must run python3 -m pip install psycopg2-binary  
import psycopg2 as psqldb

HOSTNAME = 'localhost'
DATABASE  = 'wavestyled'
USER = 'postgres'
PASS = 'cse115'
PORT = 5432

try: 
    with psqldb.connect(   ## open the connection
            host = HOSTNAME,
            dbname = DATABASE,
            user = USER,
            password = PASS,
            port = PORT) as conn:

        with conn.cursor(cursor_factory=psqldb.extras.DictCursor) as curs:
    
            # create_script = '''  '''
            # curs.execute(create_script)

            ### insert
            # insert_script = 'INSERT INTO .....'
            # insert_values = [(1,2,3,4,4)]
            # for iv in insert_values:
            #    curs.execute(insert_script, iv)

            # delete_script = 'DELETE FROM Wardrobe WHERE name = %s'
            # delete_record = ('Brih',)
            # cur.execute(delete_script, delete_record)

            ## get data
            curs.execute('SELECT * FROM Wardrobe')
            for item in curs.fetchall():
                pass
            conn.commit() ## save transactions into the database
except Exception as error:
    print(error)
finally:
    if conn:
        conn.close()   ## close the connection -- wraps each SQL call