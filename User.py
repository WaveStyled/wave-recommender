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

    