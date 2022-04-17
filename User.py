from Wardrobe import Wardrobe
from Recommender import Recommender
from os.path import exists
from uuid import uuid1
import numpy as np

class User:
    def __init__ (self, id=999):
        self.id = id
        self.wd = Wardrobe()
        self.rec = Recommender()
        self.loaded = False

    def authenticate(self, id):
        return self.id == id

    def addWDItem(self, item): # item - tuple
        self.wd.addItem(item)
    
    def removeWDItem(self, primary_key):
        self.wd.deleteItem(primary_key)
    
    def getWD(self):
        return self.wd

    def begin_calibration(self, num):
        return self.wd.getRandomFit(num)

    def end_calibration(self, ratings, outfits, attrs):
        self.wd.outfitToDB(outfits=outfits, ratings=ratings, attrs=attrs)

    def wardrobe_init(self, path='./good_matts_wardrobe.csv'):
        self.wd.from_csv(path)

    ## MODEL STUFF
    def load_model(self):
        self.loaded = exists(f'{self.id}.h5')
        if not self.loaded:
            self.rec.buildModel()
        else:
            self.rec.load_model(f'{self.id}')
    
    def train_model(self):
        train, labels = self.rec.create_train()
        self.rec.train(train, labels)
    
    def update_preferences(self, new_data = True, train_again=False):
        if new_data:
            self.rec.from_csv('./outfits.csv')
        self.rec.fromDB()
        self.rec.addColors(self.wd)
        self.rec.encode_colors()
        if train_again:
            self.train_model()

    def get_recommendations(self, occasion, weather, buffer=5):
        return self.rec.recommend(occasion=occasion, weather=weather, wd=self.wd, buffer=buffer)
        
# duplicate? want to put these in DB

    def save_progress(self):
        self.rec.save_model(f'{self.id}.h5')

    def getID(self):
        return self.id
    
    def getModel(self):
        return self.rec


class UserBase:

    """
    to think about --> how to reference a User in the userbase while maintaining the
    secutiry and consistency?
    """
    def __init__ (self):
        self.users = np.array([], dtype=User)

    def logIn(self, username, pwd):
        users_id = list(filter(lambda u: u.authenticate(username, pwd), self.users))
        user = users_id[0] if len(users_id) > 0 else None 
        if not user:
            id = uuid1()
            user = User(id.int, "wave", "styled")
            np.append(self.users, user)
        return user

    