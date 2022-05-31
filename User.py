##########
#
# User and Userbase classes that respresent User and set of User abstractions
#
# Encapsulates User data into an object (Wardrobe, id, Recommender) and 
# also lays foundation of Userbase which is a list of User objects
#
##########
from Wardrobe import Wardrobe
from Recommender import Recommender
from os.path import exists
from hashlib import md5
from datetime import date

class User:
    
    """
    Function: 
    User constructor

    Desc: 
    Initializes User object which stores an empty Wardrobe and empty Recommender

    Inputs: id INT --> user identifier
    
    Returns: None
    """
    def __init__ (self, id=999):
        self.id = id
        self.wd = Wardrobe()
        self.rec = Recommender()
        self.loaded = False
        self.ootd = {}

    def authenticate(self, id):
        return self.id == id

    """
    Function: 
    User - add Item to Wardrobe

    Desc: 
    Adds the item tuple to the User's wardrobe object

    Inputs: Item -- 14-tuple that corresponds to a row in the 
    user's Wardrobe table in the database name
    
    Returns: None
    """
    def addWDItem(self, item): # item - tuple
        self.wd.addItem(item)
        return True
    
    """
    Function: 
    User - delete Item from Wardrobe

    Desc: 
    Deletes the item tuple from the User's wardrobe object.
    Note: Side Effect is that no further Recommendations will consider
    that item but the outfits that have previously been recommended will
    still exist

    Inputs: 
    - primary_key INT --> the primary key corresponding to the item that must be deleted
    from the Wardrobe
    
    Returns: None
    """
    def removeWDItem(self, primary_key):
        self.wd.deleteItem(primary_key)



    """
    Function:
    User - update Item in wardrobe

    Desc:
    Given the item, updates the corresponding item in the User's wardrobe

    Inputs:
    - item LIST --> list representation of the item to be updated
    
    Returns: None
    """    
    def updateWDItem(self, item):
        self.wd.updateItem(item)

    ## CALIBRATION
    """
    Function: 
    User - Begin Calibration

    Desc: 
    Generate the specified random fits in the calibration phase
    Returns "num" randomly generated fits 

    Inputs: 
    - num INT --> the number of fits that the User wants to see in the calibration phase
    
    Returns:
     - List of fits of random occasions and weather patterns in their primary key representations
    """
    def begin_calibration(self, num):
        return self.wd.getRandomFit(num)

    """
    Function: 
    User - End calibration

    Desc: 
    Send all collected outfits and corresponding ratings, attributes to the outfits database

    Inputs: 
    - ratings List(INT) --> ratings (1 or 0) for each outfit
    - outfit List(List(INT)) --> outfits to send to DB
    - attr List((INT, INT)) --> list of occasion, weather tuples 
    
    NOTE that all lengths must be equal as each element in each corresponds to each other
    
    Returns: None
    """
    def end_calibration(self, ratings, outfits, attrs):
        self.wd.outfitToDB(outfits=outfits, ratings=ratings, attrs=attrs, userid=self.id)

    """
    Function: 
    User - Intialize Wardrobe using a CSV file

    Desc: 
    Returns the Wardrobe object associated to the User

    Inputs: None
    
    Returns: Wardrobe object that stores the User's items
    """
    def wardrobe_init(self, path='./csv_files/good_matts_wardrobe.csv'):
        self.wd.from_csv(path)
        return True

    ## MODEL STUFF
    """
    Function: 
    User - Load Model

    Desc: 
    Builds the Recommendation model or loads the model if one already exists

    Inputs: None
    
    Returns: None
    """
    def load_model(self):
        self.loaded = exists(f'{self.id}.h5')
        if not self.loaded:
            self.rec.buildModel()
        else:
            self.rec.load_model(f'{self.id}')
    
    """
    Function: 
    User - Train Model

    Desc: 
    Generates training data from the recommender object and uses it to train the recommender's
    model to learn the user's preferences

    Inputs: None
    
    Returns: None
    """
    def train_model(self):
        train, labels = self.rec.create_train()
        self.rec.train(train, labels)
    
    """
    Function: 
    User - Update the User's preferences

    Desc: 
    Reloads the Recommender dataframe to store the outfits in the Database, performs the color
    encoding, then retrains the model with new data included, expanding the learning scope of the 
    model with new data

    Inputs:
    - new_data BOOL --> whether there has been new data to retrain
    - train_again BOOL --> actually train the model again or not 
    
    Returns: None
    """
    def update_preferences(self, new_data = True, train_again=False):
        if not new_data:
            self.rec.from_csv('./csv_files/outfits.csv')
        self.rec.fromDB(userid=self.id)  ## need another parameter to dictate which DB to draw from
        self.rec.addColors(self.wd)
        self.rec.encode_colors()
        if train_again:
            self.train_model()

    """
    Function:
    User - choose OOTD

    Desc:
    Sets the OOTD (outfit of the day) for the specified weather and occasion

    Input:
    - fit LIST --> fit to be the OOTD
    - weather STR --> weather of the OOTD (in the form 'we_XXX')
    - occasion STR --> occasion of the OOTD (in the form 'oc_XXX')

    Returns: None
    """
    def chooseOOTD(self, fit,  weather, occasion):
        today = date.today()
        # mm/dd/y
        d = today.strftime("%m/%d/%y")
        self.ootd[(d,weather,occasion)] = fit

    """
    Function:
    User - get OOTD 

    Desc: 
    Returns the OOTD for a specific weather, occasion, and date

    Input:
    - weather STR --> weather of the OOTD (in the form 'we_XXX')
    - occasion STR --> occasion of the OOTD (in the form 'oc_XXX')
    - d STR --> Date string for the desired Date
            - Note if "" (empty string) is passed, current date is assumed

    Returns:
    - the fit for the desired date, weather, occasion
     - If the specified occasion, weather, date combo doesnt exist then returns
    empty list
    """
    def getOOTD(self, weather, occasion, d=""):
        if d == "":
            today = date.today()
            # mm/dd/y
            d = today.strftime("%m/%d/%y")
        
        ootd = self.ootd.get((d, weather, occasion), [])
        if not len(ootd) and len(self.ootd.values()):
            ootd = list(self.ootd.values())[-1]
        return ootd
        
    """
    Function: 
    User - Get Recommendation Fits

    Desc: 
    Returns the specified number of recommended fits for the given occasion and weather

    Inputs:
    - occasion STR --> the occasion string ("formal", "casual")
    - weather STR --> the weather string ("hot", "cold")
    - buffer INT --> how many outfits to recommend at once
    
    Returns: List of fits that are most probabilistically likely to be chosen by the user
    """
    def get_recommendations(self, occasion, weather, buffer=5):
        return self.rec.recommend(occasion=occasion, weather=weather, wd=self.wd, buffer=buffer)
        
    """
    Function: 
    User - Save progress

    Desc: 
    Save the Recommender to a file to be reloaded on reboot

    Inputs: None
    
    Returns: None
    """
    def save_progress(self):
        self.rec.save_model(f'{self.id}.h5')

    """
    Function: 
    User - get ID

    Desc: 
    Returns the ID of the User

    Inputs: None
    
    Returns: id INT --> the ID of the user
    """
    def getID(self):
        return self.id
    
    """
    Function: 
    User - get Model

    Desc: 
    Returns the Recommender object of the user

    Inputs: None
    
    Returns: Recommender object associated with the user
    """    
    def getModel(self):
        return self.rec

    """
    Function: 
    User - add Item to Wardrobe

    Desc: 
    Returns the Wardrobe object associated to the User

    Inputs: None
    
    Returns: Wardrobe object that stores the User's items
    """
    def getWD(self):
        return self.wd

    def __str__ (self): ## TO STRING
        return f'ID: {self.id} {self.ootd} \n Wardrobe:\n{str(self.wd)} \n Recommender:\n{str(self.rec)}'

###########

class UserBase:
    
    """
    Function: 
    UserBase constructor

    Desc: 
    Initializes UserBase object that is a dictionary that maps user IDs to their
    user objects

    Inputs: None
    
    Returns: None
    """  
    def __init__ (self):
        self.users = dict()
        
    """
    Function: 
    UserBase Generate Key

    Desc: 
    Generates a unique ID string for a user given its ID

    Inputs: id INT --> user identifier
    
    Returns: None
    """
    def __gen_key (self, id):
        input_str = f'userid{id}'
        return str(md5(input_str.encode('utf8')).hexdigest())

    """
    Function: 
    UserBase - add new User

    Desc: 
    Initializes User object with the given ID and adds it to the UserBase mapping

    Inputs: id INT --> user identifier
    
    Returns: User Object that was created
    """
    def add_new_user(self, id):
        new_user = User(id)
        self.users.update({self.__gen_key(id): new_user})
        return new_user

    """
    Function: 
    UserBase - get User

    Desc: 
    Given the user ID, return the user object associated with it.
    If the user does not exist, UserBase creates a new User with the given ID 

    Inputs: id INT --> user identifier
    
    Returns: None
    """
    def get_user(self, id):
        user = self.users.get(self.__gen_key(id))
        return user if user else self.add_new_user(id)
    
    """
    Function: 
    UserBase - get Userbase

    Desc: 
    Returns the list of User objects that the UserBase object stores

    Inputs: None
    
    Returns: 
    - List of User objects
    """
    def get_userbase(self):
        return self.users.values()
    
    def __str__ (self): # TO STRING
        return "\n\n".join([str(u) for u in self.users.values()])
