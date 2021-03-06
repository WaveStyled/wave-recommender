##########
#
# Reference file for Python SQL connection and queries
#
# INSTALL:  run python3 -m pip install psycopg2-binary  
##########

import psyscopg2 as psqldb

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

        with conn.cursor(cursor_factory=psqldb.extras.DictCursor) as curs:  # cursor_factory optional
    
            # create_script = '''  '''
            # curs.execute(create_script)   -- GENERAL SCRIPT FORMAT

            ###
            # insert_script = 'INSERT INTO .....'
            # insert_values = [(1,2,3,4,4)]
            # for iv in insert_values:
            #    curs.execute(insert_script, iv)  -- GENERAL INSERT FORMAT

            # delete_script = 'DELETE FROM Wardrobe WHERE name = %s'
            # delete_record = ('Brih',)
            # cur.execute(delete_script, delete_record)   -- GENERAL DELETE FORMAT

            ## get data
            curs.execute('SELECT * FROM Wardrobe')
            for item in curs.fetchall():      # --- GENERAL SELECT FORMAT
                pass

            query = f"SELECT * FROM Wardrobe WHERE PIECEID = {curs}"
            curs.execute(query)
            tup = dict(curs.fetchone())   ## want to then invoke the Recommender class object
            conn.commit() ## save transactions into the database
except Exception as error:
    print(error)
finally:
    if conn:
        conn.close()   ## close the connection -- wraps each SQL call
