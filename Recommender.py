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

from cv2 import FileStorage_FORMAT_MASK
import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
import psycopg2 as psqldb 
# import tensorflow as tf

# Suppresses the errors that TF gives on import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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
            relevant = [of.occasion, of.weather, 0, 0, 0, 0, 0]
            liked = [1 if (of.liked or of.liked) == 't' else 0]
            train.append([outfit,colors,relevant])
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

    
    def recommend(self, occasion, weather, wd, max_tries=20, buffer=5):  # use tf predict method
        prediction = buffer
        fit = None
        probs = np.array([])
        fits = np.empty(shape=(0,7), dtype=np.int16)
        metadata = [Wardrobe.oc_mappings[occasion], Wardrobe.we_mappings[weather],0,0,0,0,0]
        
        while prediction and max_tries:
            max_tries-=1
            fit = wd.gen_random(occasion, weather)  # here we have to check if outfit hasnt been given before (could do it in gen random)
            if -1 in fit: continue
            colors = []
            for f in fit:
                item = wd.getItem(f)
                color = item[2] if item else item
                color_ind = Recommender.mappings.get(color) if color else Recommender.mappings.get('null')
                if not color_ind:
                    color_ind = len(Recommender.mappings) + 1
                    update_color = color if color else 'null'
                    Recommender.mappings.update({update_color : len(Recommender.mappings) + 1})
                colors.append(color_ind)
            to_predict = np.array([[fit, colors, metadata]])
            pred = self.model.predict(to_predict)
            prediction -= np.argmax(pred)

            probs = np.append(probs, [pred[0][1]])
            fits = np.concatenate((fits, [fit]))

        if -1 in fit: return None
        ## partial sort to buffer elements
        ind = np.argpartition(probs, -1 * buffer)[-1 * buffer:] # get the indices with highest 4 probabilities
        good_fits = fits[ind].tolist()
        final_fits = []
        for fit in good_fits:
            repeats = self.dt.loc[(self.dt['hat']== fit[0]) & (self.dt['shirt']== fit[1]) & (self.dt['sweater']== fit[2]) & (self.dt['jacket']== fit[3]) & (self.dt['bottom_layer']== fit[4]) & (self.dt['shoes']== fit[5]) & (self.dt['misc']== fit[6]) &(self.dt['liked']== 0)].to_numpy().tolist()
            if(len(repeats)  == 0):
                final_fits.append(fit)
        return final_fits
    
    def buildModel(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(3,7)))
        self.model.add(Dense(units = 14, activation='relu'))
        self.model.add(Dense(units = 8, activation= 'relu'))
        self.model.add(Dense(units = 2, activation = "softmax"))
        #self.model.add(Dense(units = 1, activation='sigmoid')) 
        #self.model.compile(loss='binary_crossentropy',
        self.model.compile(loss=SparseCategoricalCrossentropy(),
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

    def dedup(self):
        self.dt.drop_duplicates(in_place=True)


def main():
    w = Wardrobe()
    w.from_csv("./good_matts_wardrobe.csv")

    ## setup
    r = Recommender()
    r.buildModel()
    load = int(input("load model? (1/0): "))
    if load == 1:
        r.load_model('wavestyled')
    else:
        r.from_csv('./outfits.csv')
        #r.fromDB()
        r.addColors(w)
        r.encode_colors()

        # training
        train, labels = r.create_train()
        r.train(train,labels)

    #recommending
    oc_mappings = ["oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors", "oc_comfy"]  ## maps occasion to integer (id)
    we_mappings = ["we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical"]
    #while True:
    while True:       
        print("RECOMMENDATIONS:")
        
        occasion = int(input("What occasion? (formal (0), semi_formal (1), casual (2), workout (3), outdoors (4), comfy (5) ): "))
        weather = int(input("What occasion? (cold (0), hot (1), rainy (2), snowy (3), typical (4) ): "))
        #f = r.generate_outfit(oc_mappings[occasion], we_mappings[weather], w)
        fits = r.recommend(oc_mappings[occasion], we_mappings[weather], w)
        for f in fits:
            print(f)
        save = int(input("save model?: "))
        if save == 1:
            r.save_model('wavestyled')
            break

if __name__ == '__main__':
     main()

#tf.debugging.set_log_device_placement(True)