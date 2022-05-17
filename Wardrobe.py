##########
#
# Wardrobe Class represented as a PANDAS Dataframe
#
# Stores all the clothing items and generates random outfits
#
# INSTALL: python3 -m pip install pandas 
#          python3 -m pip install opencv-python
#          python3 -m pip install psycopg2-binary
#
# Note: pandas should install NumPy for you, but in case: python3 -m pip install numpy
##########
from Item import Item
import pandas as pd
from random import randint
import cv2 as cv
import numpy as np
import psycopg2 as psqldb

class Wardrobe:
    
    oc_mappings = {"oc_formal": 1, "oc_semi_formal": 2, "oc_casual": 3, "oc_workout": 4, "oc_outdoors": 5, "oc_comfy": 6}  ## maps occasion to integer (id)
    we_mappings = {"we_cold": 1, "we_hot": 2, "we_rainy": 3, "we_snowy": 4, "we_typical": 5}                               ## maps weather to integer (id)

    """
    Function: 
    Wardrobe constructor

    Desc: 
    Initializes an empty Pandas dataframe that represents the inputted wardrobe items

    Inputs:
    - cols [STR] --> the column names of the wardrobe - DEFAULT the column names of the Wardrobe
    
    Returns: None
    """
    def __init__ (self, cols=['pieceid', "color", "type", "date_added", "times_worn", 
            "rating", "oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors",
            "oc_comfy", "we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical", "dirty"]
                    ):
        self.logged_in = False
        self.dt = pd.DataFrame(columns=cols)
        self.cols = cols
    
    """
    Function: 
    Wardrobe load data from a given csv file

    Desc: 
    Fills the Wardrobe dataframe with the data from the csv
    
    ** ENSURE csv is consistent with the Wardrobe format **

    Inputs:
    - path STR --> the pathname of the csv file as a string
    
    Returns: None
    """
    def from_csv(self, path):
        self.dt = pd.read_csv(path)
        self.logged_in = True
    
    """
    Function: 
    Wardrobe - Add item

    Desc: 
    Adds a new item to the wardrobe by appending a new row to the Dataframe
    Invoked by the Python Server when a PUT request is recieved

    Inputs:
    - clothing_item --> 18-tuple that stores the data for a single wardrobe item
                * See column names to see which tuple entry corresponds to which
    
    Returns: None
    """
    def addItem(self, clothing_item):
        self.dt.loc[self.dt.shape[0]] = clothing_item

    
    """
    Function: 
    Wardrobe - Get Item

    Desc: 
    Outputs the desired item with the given 'piece_id'

    Inputs:
    - primary key INT --> 'piece_id' of item
    - ind    BOOL     --> bool that dictates whether the Dataframe index should be returned with the tuple
    
    Returns:
    TYPE: <nd.recarray> object, which is a dictionary-like object that can reference specific tuple-entries based on column name
        EX: to get the type of an item X, the code X.type outputs the string id of the item
    
    - The corresponding tuple of the item. None if the given 'piece_id' does not exist in the dataframe
    - Ensures that only 1 item is returned as primary keys are unique
    """
    def getItem(self, primary_key, ind=False):     
        item = self.dt.loc[self.dt['pieceid'] == primary_key].to_records(index=ind)
        return item[0] if item else None

    
    """
    Function:
    Wardobe - update Item

    Desc:
    Updates the given item in its respective place in the wardrobe

    Inputs:
    - item --> item tuple to be update

    Returns:
    None

    It finds the pieceid of the desired item using the item tuple and merges the two
    in the dataframe
    """
    def updateItem(self, item):
        if isinstance(item, list):
            val = self.dt.loc[self.dt['pieceid'] == item[0]].to_records(index=True)
            if len(val) > 0:
                withoutindex = list(val[0])[1:]
                index = val[0][0]
                updated = []
                for old, new in zip(withoutindex, item):
                    if old is not None and new is None:
                        updated.append(old)
                    else:
                        updated.append(new)
                # print(withoutindex)
                # print(item)
                # print(updated)
                self.dt.iloc[index] = updated

    """
    Function: 
    Wardrobe - Get Item as Item Object -- DEPRECATED

    Desc: 
    Outputs the desired item with the given 'piece_id' as an ITEM object

    Inputs:
    - primary key INT  --> 'piece_id' of item
    
    Returns:
    TYPE: Item
    
    - The corresponding tuple of the item. None if the given 'piece_id' does not exist in the dataframe
    - Ensures that only 1 item is returned as primary keys are unique
    """
    def getItemObj(self, primary_key):
        return Item(self.getItem(primary_key))

    """
    Function: 
    Wardrobe - Delete Item

    Desc: 
    Deletes the desired item with the given 'piece_id' if it exists in the Dataframe

    Inputs:
    - primary key --> 'piece_id' of item
    
    Returns: None
    """
    def deleteItem(self, primary_key):
        item = self.getItem(primary_key=primary_key, ind=True)
        if item:
            self.dt.drop([item[0]], axis=0, inplace=True)
            self.dt.reset_index(drop=True, inplace=True)

    """
    Function: 
    Wardrobe - Get Wardrobe

    Desc: 
    Outputs the desired item with the given 'piece_id' as an ITEM object

    Inputs:
    - primary key --> 'piece_id' of item
    
    Returns:
    TYPE: List of <nd.recarray> -- SEE GetItem description for explanation on nd.recarray
    """
    def getWardrobe(self):
        return self.dt.to_records(index=False)
    
    """
    Function: 
    Wardrobe - Filter

    Desc: 
    Filters the dataframe based on the attribute (column name) and returns all rows with the given type of attribute
    
    Inputs:
    - sub_attr STR --> the value to be searched for. MUST be a subset of the given attribute
    - attribute STR --> the general column to be searched. DEFAULT on 'type'
    
    Returns:
    TYPE: list of <nd.recarray> that represent rows that meet the given conditions
    """
    def filter(self, sub_attr, attribute="type"):
        return self.dt.loc[self.dt[attribute] == sub_attr].to_records(index=False)
    

    """
    Function: 
    Wardrobe - get Dataframe

    Desc: 
    Returns the Pandas dataframe that represents the wardrobe

    Inputs: None
    
    Returns:
    TYPE: Pandas Dataframe object that has all the current data of the Wardrobe
    """
    def getdf(self):
        return self.dt

    """
    Function: 
    Wardrobe - Gen

    Desc: 
    Generates a random Item given the occasion and weather
    Gives the 'piece_id' of a random item in the Wardrobe that can be worn on provided occasion and weather (both 1s in DataFrame)

    Inputs:
    - occasion STR --> occasion label    -- must be a valid column name in dataframe
    - weather STR  --> weather label
    - ends STR/CHAR --> the character the item ends with
    
    Returns:
    TYPE: INT
    The 'piece_id' of the item generated, -1 if no items exist in that category
    """
    def gen(self, occasion, weather, ends):
        clean_wardrobe = self.dt[self.dt['dirty'] == False]
        x = clean_wardrobe.loc[(clean_wardrobe["type"].str.endswith(ends)) & (clean_wardrobe[occasion] == 1) & (clean_wardrobe[weather] == 1) ]
        if (len(x.index)==0):
            return -1
        chosen = x.sample()
        return int(chosen["pieceid"])

    """
    Function: 
    Wardrobe - Gen Random (Outfit)

    Desc: 
    Generates a random outfit given the weather and occasion. Each type of weather has a different outfit generation
    procedure. (Ex. for a hot day, don't need to consider sweatshirts)

    Inputs:
    - occasion STR --> occasion label    -- must be a valid column name in dataframe
    - weather STR  --> weather label
    
    Returns:
    TYPE: [INT] with exactly 7 elements containing the 'piece_ids' of the generated items
    """
    def gen_random(self, occasion, weather):
        top = ""
        shorts = ""
        shoes = ""
        under =""
        bot = ""
        hat = ""
        fit = [0,0,0,0,0,0,0]
        if (weather == "we_hot"):
            hat_chance = randint(1,4)
            
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            top = self.gen(occasion,weather,"S")
            fit[1] = top
            bot = self.gen(occasion,weather,"H")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "we_cold"):
            hat_chance = randint(1,3)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            undershirt_chance = randint(1,4)
            if(undershirt_chance == 1):
                under =	self.gen(occasion,weather,"S")
                fit[1] = under
            top =  self.gen(occasion,weather,"T")
            fit[2] = top
            jacket_chance = randint(1,4)
            if(jacket_chance == 1):
                jacket = self.gen(occasion, weather,"C")
                fit[3] = jacket
            
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "we_rainy"):
            
            hat_chance = randint(1,2)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            shirt_or_sweat = randint(1,4)
            if(shirt_or_sweat == 1):
                # shirt
                top = self.gen(occasion,weather,"S")
                fit[1] = top
            else:
                top =  self.gen(occasion,weather,"T")
                fit[2] = top
            
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
            jacket = self.gen(occasion, weather,"C")
            fit[3] = jacket
        elif (weather == "we_typical"):
            shirt_or_sweat = randint(1,2)
            if(shirt_or_sweat == 1):
                # shirt
                top = self.gen(occasion,weather,"S")
                fit[1] = top
            else:
                top =  self.gen(occasion,weather,"T")
                fit[2] = top
            
            shorts_or_pants =randint(1,2)

            if(shorts_or_pants == 1):
                bot = self.gen(occasion,weather,"H")
                fit[4] = bot
            else:
                bot = self.gen(occasion,weather,"P")
                fit[4] = bot
            
            hat_chance = randint(1,4)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
        elif (weather == "we_snowy"):
            hat_chance = randint(1,2)
            if(hat_chance == 1):
                hat = self.gen(occasion,weather,"A")
                fit[0] = hat
            undershirt_chance = randint(1,4)
            if(undershirt_chance == 1):
                under =	self.gen(occasion,weather,"S")
                fit[1] = under
            top =  self.gen(occasion,weather,"T")
            fit[2] = top
            bot = self.gen(occasion,weather,"P")
            fit[4] = bot
            shoes = self.gen(occasion,weather,"O")
            fit[5] = shoes
            jacket = self.gen(occasion, weather,"C")
            fit[3] = jacket
        return fit

    """
    Function: 
    Wardrobe - Gen Random (Outfit)

    Desc: 
    Generates a random outfit given the weather and occasion. Each type of weather has a different outfit generation
    procedure. (Ex. for a hot day, don't need to consider sweatshirts). Can input probabilities of generating fits 
    that fit specific weather patterns (Ex. in Santa Cruz one expects higher typical weather than snowy)

    Inputs:
    - num_fits INT --> the number of fits to be generated
    - oc_prob [FLOAT] --> 6 element list of probabilities corresponding to the occasion labels
    - we_prob [FLOAT] --> 5 element list of probabilities corresponding to the weather labels
    
    Returns:
    TYPE: [[INT], [STR]]
    Returns a list of the generated fit and the corresonding weather & occasion parameters
    Both lists empty if no fit can be generated
    """
    def getRandomFit(self, num_fits=1, oc_prob=[0.05, 0.07, 0.5, 0.15, 0.13, 0.1], we_prob=[0.3, 0.25, 0.1, 0.03, 0.32]):
        occasions = ["oc_formal","oc_semi_formal","oc_casual","oc_workout","oc_outdoors","oc_comfy"]
        weather = ["we_hot","we_cold","we_rainy","we_snowy","we_typical"]
        fits = []
        oc_we = []
        oc_ind = np.arange(6)
        we_ind = np.arange(5)
        for _ in range(0,num_fits,1):
            rand_oc = np.random.choice(oc_ind, p = oc_prob)
            rand_we = np.random.choice(we_ind, p = we_prob)
            oc = occasions[rand_oc]
            we = weather[rand_we]
            fit = self.gen_random(oc,we)
            
            if -1 not in fit: # if outfit is possible
                fits.append(fit)
                oc_we.append([oc,we])

        return [fits,oc_we]

    """
    Function: 
    Wardrobe - Display Fit

    Desc: 
    Displays the Items of a generated Outfit using local storage of the image files. 
    Presents any valid outfit on the screen in a side-by-side format. Runs through
    a series of outfits that the user can 'rate'
    
    HOW TO USE THE OPENCV image display:
    Press 1 to register a 'LIKE'  --> 1
    Press 0 to register a 'DISLIKE' --> 0
    Press 'q' to quit the showcase and return all given ratings
     
    *** ALL IMAGES MUST BE NAMED AFTER THE 'piece_id' OF THE ITEM IT CORRESPONDS TO 
        AND MUST BE IN JPEG FORMAT (SEE ./wave-server/testing_files/heic_to_jpeg.py TO 
                                    CONVERT IPHONE PICS TO JPEG) ***

    Inputs:
    - outfit [[INT]] --> List of 7-element lists of 'piece_id' of outfit items
    - conditions [STR] --> 2 element list of occasion (first) and weather (second)
    - path STR  --> the path string of the images folders (NO NEED TO INCLUDE FINAL '/' )
    
    Returns:
    TYPE: ([INT], [[INT]], [[STR]])
    Returns the ratings, outfits and occasion/weather of given fits that the user considered
    ** Possible that not all outfits inputted will be rated, so the output list is a subset **
    """
    def displayFit(self, outfit, conditions, path):
        ratings = []
        i = 0
        for im in outfit:
            if im:
                images = []
                overall_shape = None
                for item in im:
                    if item:
                        image = cv.imread(f'{path}/{item}.jpeg')
                        image = cv.resize(image, (0, 0), None, .1, .1)
                        if not overall_shape:
                            overall_shape = image.shape
                        elif image.shape != overall_shape: # adjust image if not oriented properly
                            image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
                        images.append(image)
                ims = np.hstack((images[0], images[1]))  # puts the images side-by-side
                for rest in images[2:]:
                    ims = np.hstack((ims,rest))
                cv.imshow(f'outfit: {str(im)}, attr: {str(conditions[i])}', ims)
                while True: # user inputs
                    like_dislike = cv.waitKey(0) & 0xFF - 48
                    print(like_dislike)
                    if not like_dislike or like_dislike == 1:
                        break
                    if like_dislike == 65:
                        cv.destroyAllWindows()
                        return ratings, outfit[:i], conditions[:i]
                ratings.append(like_dislike)
                cv.destroyAllWindows()  # if you want to remove window on key press
            i +=1
        cv.destroyAllWindows()
        return ratings, outfit, conditions

    """
    Function: 
    Wardrobe - Send Outfit info to the Outfits table in PSQL Database

    Desc: 
    Sends the outfits, correponding ratings and [occasion/weather] attributes and inputs it into
    the Outfits table in the PSQL Database, designating a unique Primary Key 'outfit_id' on entry

    Inputs:
    - outfits [[INT]] --> list of outfits to send
    - ratings [INT]  --> list of ratings for each outfit (each index corresponds to the rating of outfit at same index)
    - attrs [[STR]]  --> list of corresponding occasion/weather
    - THE REST ARE ALL DATABASE ATTRIBUTES THAT ARE DEFAULTED TO THE LOCAL PSQL
    
    Returns: None
    """
    def outfitToDB(self, outfits, ratings, attrs, HOSTNAME='localhost', DATABASE='wavestyled', USER='postgres', PASS='cse115', PORT=5432):
        if len(outfits) == len(ratings) == len(attrs):
            conn = None
            try: 
                with psqldb.connect(   ## open the connection
                        host = HOSTNAME,
                        dbname = DATABASE,
                        user = USER,
                        password = PASS,
                        port = PORT) as conn:

                    with conn.cursor() as curs:
                        curs.execute('SELECT COUNT(*) FROM outfits')
                        pk = curs.fetchone()[0] + 1

                        print("DATABASE CONNECTED")

                        insert_script = ("INSERT INTO outfits " 
                                        "(outfit_id, hat, shirt, sweater, jacket, bottom_layer, "
                                        "shoes, misc, occasion, weather, liked) "
                                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                        
                        for i, outfit in enumerate(outfits):
                            inputs = [pk+i] + outfit + [Wardrobe.oc_mappings.get(attrs[i][0]), Wardrobe.we_mappings.get(attrs[i][1])] + [bool(ratings[i])]
                            curs.execute(insert_script, inputs)
                        conn.commit() ## save transactions into the database
            except Exception as error:
                print(error)
            finally:
                if conn:
                    conn.close()   ## close the connection
        
    """
    Function: 
    Intializer from the DB

    Desc: 
    Initializes the dataframe from the given datatable
    DEFAULTs to Outfits table, so must provide table name if want to load wardrobe
    Inputs: (optional) the info for the database connection
    
    Returns: None
    """
    def fromDB(self, table='outfits', HOSTNAME='localhost', DATABASE='wavestyled', USER='postgres', PASS='cse115', PORT=5432):
        conn = None
        try: 
            with psqldb.connect(   ## open the connection
                    host = HOSTNAME,
                    dbname = DATABASE,
                    user = USER,
                    password = PASS,
                    port = PORT) as conn:

                with conn.cursor() as curs:
                    curs.execute(f'SELECT * FROM {table}')
                    rows = curs.fetchall()
                    cols = self.dt.columns.values.tolist()[:len(rows[0] if rows[0] else 14)]
                    self.dt = pd.DataFrame(rows, columns=cols)

        except Exception as error:
            print(error)
        finally:
            if conn:
                conn.close()   ## close the connection

    def logIn(self):
        self.logged_in = True
    
    def logOut(self):
        self.logged_in = False

    """
    Function: 
    Wardrobe - Get all Items of a specific category usig obj[] notation

    Desc: 
    Given the terminating character of a clothing item code, returns all items in the wardrobe
    Allows to retrieve items for queries like 'GET ALL SHIRTs' or 'GET ALL PANTs' in thw Wardrobe

    Inputs:
    - clothing_type CHAR --> Type of item to retrieve
    
    Returns:
    TYPE: List of <nd.recarray> which are the tuples of all items of that type
    """
    def __getitem__ (self, clothing_type):  ## allows for [] notation with the object
        return self.dt.loc[(self.dt["type"].str.endswith(clothing_type))].to_records(index=False)

    def __str__ (self): # To String
        return str(self.dt)

    def __len__ (self): # Get the number of elements in the wardrobe
        return len(self.dt)

    def __del__ (self): # destructor procedure
        #print("Wardrobe eliminated")
        return