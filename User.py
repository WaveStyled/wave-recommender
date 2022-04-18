from Wardrobe import Wardrobe
from Recommender import Recommender
from os.path import exists
from uuid import uuid1
import numpy as np
from hashlib import md5

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
        if not new_data:
            self.rec.from_csv('./outfits.csv')
        self.rec.fromDB()  ## need another parameter to dictate which DB to draw from
        self.rec.addColors(self.wd)
        self.rec.encode_colors()
        if train_again:
            self.train_model()

    def get_recommendations(self, occasion, weather, buffer=5):
        return self.rec.recommend(occasion=occasion, weather=weather, wd=self.wd, buffer=buffer)
        
    def save_progress(self):
        self.rec.save_model(f'{self.id}.h5')

    def getID(self):
        return self.id
    
    def getModel(self):
        return self.rec

    def __str__ (self):
        return f'ID: {self.id} \n Wardrobe:\n{str(self.wd)} \n Recommender:\n{str(self.rec)}'

class UserBase:
    def __init__ (self):
        self.users = dict()

    def __gen_key (self, id):
        input_str = f'userid{id}'
        return str(md5(input_str.encode('utf8')).hexdigest())

    def add_new_user(self, id):
        new_user = User(id)
        self.users.update({self.__gen_key(id): new_user})
        return new_user

    def get_user(self, id):
        user = self.users.get(self.__gen_key(id))
        return user if user else self.add_new_user(id)
    
    def get_userbase(self):
        return self.users.values()
    
    def __str__ (self):
        return "\n\n".join([str(u) for u in self.users.values()])

    