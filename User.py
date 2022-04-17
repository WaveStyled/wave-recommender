from Wardrobe import Wardrobe
from Recommender import Recommender
from os.path import exists
from uuid import uuid1
import numpy as np

class User:

    def __init__ (self, id, username, pwd):
        self.id = id
        self.wd = Wardrobe()
        self.rec = Recommender()
        self.username = username
        self.pwd = pwd

    def authenticate(self, username, pwd):
        if self.username == username and self.pwd == pwd:
            return self.id
        else:
            None

    def addWD(self, item):
        pass
    
    def getWD(self):
        return self.wd

    def train(self):
        path = f'{self.id}.h3'
        if exists(path):
            self.rec.load_model(path)
        else:
            pass #train logic here

    def recommend(self):
        pass

    def save_progress(self):
        self.rec.save_model(f'{self.id}.h3')

    def getID(self):
        return self.id


class UserBase:

    def __init__ (self):
        self.users = np.array([], dtype=User)

    def logIn(self, username, pwd):
        ## authentication here
        #account = username in self.users.ids
        account = False
        if not account:
            id = uuid1()
            np.append(self.users, User(id.int, "wave", "styled"))
        else:
            users_id = list(filter(lambda u: u.authenticate(username, pwd), self.users))
            return users_id is not None