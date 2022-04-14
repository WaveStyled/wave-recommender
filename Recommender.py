##########
#
# Recommender Class represented as a PANDAS Dataframe
#
# Stores all the outfits, processes the data and trains a Neural Network to be used
# as the basis for our recommendations
#
# INSTALL: python3 -m pip install tensorflow 
#          Perhaps matplotlib, etc.
#
# Note: Tensorflow is large, so expect slow installation and slow running of programs
# ANY RUN of this program should take around 15-20 seconds to run
##########

import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
import psycopg2 as psqldb 
# import tensorflow as tf

# Suppresses the errors that TF gives on import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

class Recommender(Wardrobe):

    mappings = {} # stores the unique color encodings; shared among all Recommender instances

    """
    Function: 
    Recommender constructor

    Desc: 
    Initializes an empty Pandas dataframe that represents the outfits (without color)
    Recommender inherits the Wardrobe class in order to have extendable functionality
    Initializes the Neural Network Model as well

    Inputs: None
    
    Returns: None
    """
    def __init__ (self):
        super().__init__(['outfit_id','hat','shirt','sweater','jacket','bottom_layer',
                 'shoes','misc','times_worn','recent_date_worn','fit_score','occasion','weather','liked'])
        self.model = None

    """
    Function: 
    Recommender Intializer from the Outfits DB

    Desc: 
    Initializes the Recommender dataframe from the Outfits datatable

    Inputs: (optional) the info for the database connection
    
    Returns: None
    """
    def fromDB(self, HOSTNAME='localhost', DATABASE='wavestyled', USER='postgres', PASS='cse115', PORT=5432):
        conn = None
        try: 
            with psqldb.connect(   ## open the connection
                    host = HOSTNAME,
                    dbname = DATABASE,
                    user = USER,
                    password = PASS,
                    port = PORT) as conn:

                with conn.cursor() as curs:

                    curs.execute('SELECT * FROM outfits')
                    rows = curs.fetchall()
                    for r in rows:
                        self.addItem(r)
                    
                    conn.commit() ## save transactions into the database
        except Exception as error:
            print(error)
        finally:
            if conn:
                conn.close()   ## close the connection

    """
    Function: 
    Recommender - Normalize

    Desc: 
    Normalizes the clothing data points to a value between 0 and 1
    NOTE: DATAFRAME MODIFIED IN PLACE

    Inputs:
    - cols [STR] --> the column names of the outfit table - DEFAULT the column names of the Outfits table
    
    Returns: None (modifies the dataframe in place)
    """
    def normalize(self, col=['hat','shirt','sweater','jacket','bottom_layer','shoes','misc']):
         for c in col:
             self.dt[c] = self.dt[c] / self.dt[c].abs().max()
             self.dt[c].fillna(0, inplace=True)
    
    """
    Function: 
    Recommender - Encode Colors

    Desc: 
    ENCODES COLOR STRINGS TO UNIQUE NUMBERS
    Extracts the unique colors of each item in each row of the Dataframe and adds it to the shared color mapping after
    assigning it a unique value.
    Then, adds the encoded colors to the corresponding entries in the database

    Inputs:
    - cols [STR] --> the column color names of the outfit table - DEFAULT the column color names of the Outfits table
    
    Returns: None (modifies the dataframe in place)
    """   
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
                f[f == x] = Recommender.mappings[x] # gives some kinda warning no idea how to fix

    """
    Function: 
    Recommender - Create Training Dataset

    Desc: 
    Goes through each row in the dataframe and builds a training input for the model
    
    TRAIN INPUT STRUCTURE: 14-element tuple with first 7 being normalized types and second 7 being the colors
    
    Inputs: None
    
    Returns: 2 numpy Arrays with the first containing the training outfit set and the second containing the labels
    """
    def create_train(self): # return 107 16-tuples
        train = []
        labels = []
        for of in self.dt.itertuples():
            outfit = [of.hat, of.shirt, of.sweater, of.jacket, of.bottom_layer, of.shoes, of.misc]
            colors = [of.color_hat, of.color_shirt, of.color_sweater, of.color_jacket, of.color_bottom_layer, of.color_shoes, of.color_misc]
            liked = [1 if of.liked == 't' else 0]
            train.append(outfit+colors)
            labels.append(liked)
        return np.array(train), np.array(labels)
    
    """
    Function: 
    Recommender - Add Colors

    Desc: 
    Goes through each row in the dataframe and adds the colors of each item to the row as the color string specified
    in the given wardrobe instance
        
    Inputs: 
    - wd Wardrobe - instance of a Wardrobe
    
    Returns: None
    """
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
        history = self.model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val)) # batch size?
        
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
    """
    Function: 
    Recommender - Save Model
    Desc: 
    Saves a model given its name. Currently saves to the same directory as the recommender.py 
        
    Inputs: 
    - name: string of what the file should be called
    
    Returns: None
    """
    def save_model(self,name):
        self.model.save(name+".h5")
    
    """
    Function: 
    Recommender - Load model
    Desc: 
    loads a model given its name. Currently loads from the same directory as the recommender.py 
        
    Inputs: 
    - name: string of what the file should is called
    
    Returns: Sets recommender object(self.model) to the loaded model
    """

    def load_model(self,name):
        self.model = load_model(name+".h5")


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
    train, labels = r.create_train()
    r.buildModel()
    r.train(train,labels)

    #recommending

if __name__ == '__main__':
     main()

#tf.debugging.set_log_device_placement(True)