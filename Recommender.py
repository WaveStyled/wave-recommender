import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback


class Recommender():

    def __init__ (self, wardrobe):
        self.dt = pd.DataFrame(columns=['outfit_id' | 'hat' | 'shirt' | 'sweater' | 'jacket' | 'bottom_layer' | 'shoes' | 'misc' | 'times_worn' | 'recent_date_worn' | 'fit_score' | 'occasion' | 'weather' | 'liked' ])

    def train(self, train):
        pass

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

    def getwd(self):
        return self.wardrobe


def enumerate_colors(colors):
  filtered_list = list(filter(lambda ele:ele is not None, colors))
  unique_colors = set(filtered_list)
  return {col: i for i, col in enumerate(unique_colors)}

def add_color_to_mapping(mapping, color):
  if color and color not in mapping:
    val = max(mapping, key=mapping.get)+1
    mapping[color] = val


#tf.debugging.set_log_device_placement(True)

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


def main():
    r = Recomennder()
    r.from_csv('./outfits.csv')
    print(w)



if __name__ == '__main__':



## IDEAS:
# the model algorithm is the same, but the occasion/weather can be treated as bias terms
# Bias can be thought of as multipliers that influence
# Can ask the user about "importance" factor, which can increase or decrease that multiplier
# As a apart of the initial screening phase, the user should be prompted to give initial preference data 
# (like favorite color or color combo or type of clothing) so the model can pretrain and give decent initial screening fits
