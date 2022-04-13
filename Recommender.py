import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
# import tensorflow as tf

# Suppresses the errors that TF gives on import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Dense, Input, Activation
#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.callbacks import Callback
#from sklearn.model_selection import train_test_split

class Recommender(Wardrobe):

    colors = {}

    def __init__ (self):
         super().__init__(['outfit_id','hat','shirt','sweater','jacket','bottom_layer',
                 'shoes','misc','times_worn','recent_date_worn','fit_score','occasion','weather','liked'])
    
    def normalize(self, col=['hat','shirt','sweater','jacket','bottom_layer','shoes','misc']):
         for c in col:
             self.dt[c] = self.dt[c] / self.dt[c].abs().max()
             self.dt[c].fillna(0, inplace=True)
 
    def create_train(self, wd): # return 107 16-tuples
        train = []
        for of in self.dt.itertuples():
            outfit = [of.hat, of.shirt, of.sweater, of.jacket, of.bottom_layer, of.shoes, of.misc]
            colors = [of.color_hat, of.color_shirt, of.color_sweater, of.color_jacket, of.color_bottom_layer, of.color_shoes, of.color_misc]
            liked = [1 if of.liked == 't' else 0]
            train.append(np.array(outfit+colors+liked))
        return train
    
    def addColors(self, wd):
        self.dt['color_hat'] = self.dt.apply(
                lambda row : wd.getItem(row.hat)[2] if wd.getItem(row.hat) else None, axis=1)
        self.dt['color_shirt'] = self.dt.apply(
                lambda row : wd.getItem(row.shirt)[2] if wd.getItem(row.shirt) else None, axis=1)
        self.dt['color_sweater'] = self.dt.apply(
                lambda row : wd.getItem(row.sweater)[2] if wd.getItem(row.sweater) else None, axis=1)
        self.dt['color_jacket'] = self.dt.apply(
                lambda row : wd.getItem(row.jacket)[2] if wd.getItem(row.jacket) else None, axis=1)
        self.dt['color_bottom_layer'] = self.dt.apply(
                lambda row : wd.getItem(row.bottom_layer)[2] if wd.getItem(row.bottom_layer) else None, axis=1)
        self.dt['color_shoes'] = self.dt.apply(
                lambda row : wd.getItem(row.shoes)[2] if wd.getItem(row.shoes) else None, axis=1)
        self.dt['color_misc'] = self.dt.apply(
                lambda row : wd.getItem(row.misc)[2] if wd.getItem(row.misc) else None, axis=1)

    def getdf(self):
        return self.dt

    def train(self):

        model = Sequential()
        model.add(Dense(units = 16, input_dim=(16,), activation='relu'))
        model.add(Dense(units = 8, activation= 'relu'))
        model.add(Dense(units = 1, activation='sigmoid')) 
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        history = model.fit(
            X[:14],
            X[15], 
            epochs=50,  
            validation_split=0.2,
                    )
    def update_weights(self):
        pass 

    def add_data(self, item):
        self.wardrobe.addItem(item)
        self.update_weights()
        self.train()

    def one_hot_encode(self, param):
        #convert parameter to a valid NN input
        # for color, we can feed in a shape (1,3) tuple for each input
        pass
    
    def generate_outfit(self, occasion, weather):
        pass

    def buildModel(self, input):
        pass


def enumerate_colors(colors):
  filtered_list = list(filter(lambda ele:ele is not None, colors))
  unique_colors = set(filtered_list)
  return {col: i for i, col in enumerate(unique_colors)}

def add_color_to_mapping(mapping, color):
  if color and color not in mapping:
    val = max(mapping, key=mapping.get)+1
    mapping[color] = val



def main():
    w = Wardrobe()
    w.from_csv("./good_matts_wardrobe.csv")
    print(w.getItem(146))
    r = Recommender()
    r.from_csv('./outfits.csv')
    print(r)
    r.from_csv('./outfits_small.csv')
    r.addColors(w)
    # print(r)
    r.normalize()
    print(r.getdf().head())
    #print(r.getdf().head())
    #print(r.getdf().head())
    print(r.create_train(w))
 
if __name__ == '__main__':
     main()


#tf.debugging.set_log_device_placement(True)
'''
def build_model2(): 
  model = Sequential() 
  model.add(Dense(units = 8, input_dim=2, activation='tanh'))
  # num_parameters = 8 * (2 + 1) = 24 
  model.add(Dense(units = 8, activation='tanh')) 
  # = 8 * (8 + 1) = 72
  model.add(Dense(units = 1, activation='sigmoid')) 
  # = 1 * (8 + 1) = 9
  return model

model = build_model2()
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x,
                    y, 
                    epochs=150, 
                    batch_size=128, 
                    validation_split=0.2,
                    )

'''

## IDEAS:
# the model algorithm is the same, but the occasion/weather can be treated as bias terms
# Bias can be thought of as multipliers that influence
# Can ask the user about "importance" factor, which can increase or decrease that multiplier
# As a apart of the initial screening phase, the user should be prompted to give initial preference data 
# (like favorite color or color combo or type of clothing) so the model can pretrain and give decent initial screening fits
