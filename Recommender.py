import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
# import tensorflow as tf

# Suppresses the errors that TF gives on import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

class Recommender(Wardrobe):

    mappings = {}

    def __init__ (self):
        super().__init__(['outfit_id','hat','shirt','sweater','jacket','bottom_layer',
                 'shoes','misc','times_worn','recent_date_worn','fit_score','occasion','weather','liked'])
        self.model = None

    def normalize(self, col=['hat','shirt','sweater','jacket','bottom_layer','shoes','misc']):
         for c in col:
             self.dt[c] = self.dt[c] / self.dt[c].abs().max()
             self.dt[c].fillna(0, inplace=True)
             
    def encode_colors(self, col = ["color_hat","color_shirt","color_sweater","color_jacket","color_bottom_layer","color_shoes","color_misc"]):
        colors = set()
        for c in col:
            colors  |= set(self.dt[c].unique())

        mapping = {col: (i + len(Recommender.mappings)) 
                            for i, col in enumerate(colors) if col not in Recommender.mappings}
        Recommender.mappings.update(mapping)
        #copy = {**Recommender.mappings}

        fn = [self.dt.color_hat, self.dt.color_shirt, self.dt.color_sweater,
                self.dt.color_jacket, self.dt.color_bottom_layer, self.dt.color_shoes, self.dt.color_misc]
        
        for x in colors:
            for f in fn:
                f[f == x] = Recommender.mappings[x] # gives some kinda warnin no idea how to fix

    def create_train(self, wd): # return 107 16-tuples
        train = []
        labels = []
        for of in self.dt.itertuples():
            outfit = [of.hat, of.shirt, of.sweater, of.jacket, of.bottom_layer, of.shoes, of.misc]
            colors = [of.color_hat, of.color_shirt, of.color_sweater, of.color_jacket, of.color_bottom_layer, of.color_shoes, of.color_misc]
            liked = [1 if of.liked == 't' else 0]
            train.append(outfit+colors)
            labels.append(liked)
        return np.array(train), np.array(labels)
    
    def addColors(self, wd):
        self.dt['color_hat'] = self.dt.apply(
                lambda row : wd.getItem(row.hat)[2] if wd.getItem(row.hat) else 'null', axis=1)
        self.dt['color_shirt'] = self.dt.apply(
                lambda row : wd.getItem(row.shirt)[2] if wd.getItem(row.shirt) else 'null', axis=1)
        self.dt['color_sweater'] = self.dt.apply(
                lambda row : wd.getItem(row.sweater)[2] if wd.getItem(row.sweater) else 'null', axis=1)
        self.dt['color_jacket'] = self.dt.apply(
                lambda row : wd.getItem(row.jacket)[2] if wd.getItem(row.jacket) else 'null', axis=1)
        self.dt['color_bottom_layer'] = self.dt.apply(
                lambda row : wd.getItem(row.bottom_layer)[2] if wd.getItem(row.bottom_layer) else 'null', axis=1)
        self.dt['color_shoes'] = self.dt.apply(
                lambda row : wd.getItem(row.shoes)[2] if wd.getItem(row.shoes) else 'null', axis=1)
        self.dt['color_misc'] = self.dt.apply(
                lambda row : wd.getItem(row.misc)[2] if wd.getItem(row.misc) else 'null', axis=1)

    def train(self,X,y):
        # Split into training and validation sets
        x_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.2, random_state=144)
        X_train, X_val, y_train, y_val = train_test_split(x_set, y_set, test_size=0.25, random_state=144)

        print("Training ...")
        history = self.model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val)) # batch size?
        
        print("Evaluation ...")
        results = self.model.evaluate(X_test, y_test) #  batch_size=128 from the source
        print(f"test loss {results[0]}, test acc: {results[1]}")

        # predictions = self.model.predict(X_test)
        # perhaps include graphs?
    
    def generate_outfit(self, occasion, weather, fit):  # use tf predict method
        pass

    def buildModel(self):
        self.model = Sequential()
        self.model.add(Dense(units = 14, input_dim=14, activation='relu'))
        self.model.add(Dense(units = 8, activation= 'relu'))
        self.model.add(Dense(units = 1, activation='sigmoid')) 
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def getModel(self):
        return self.model

    def getdf(self):
        return self.dt

def main():
    w = Wardrobe()
    w.from_csv("./good_matts_wardrobe.csv")

    ## setup
    r = Recommender()
    r.from_csv('./outfits.csv')
    r.addColors(w)
    r.encode_colors()
    r.normalize()

    # training
    train, labels = r.create_train(w)
    r.buildModel()
    r.train(train,labels)

    #recommending

if __name__ == '__main__':
     main()

#tf.debugging.set_log_device_placement(True)