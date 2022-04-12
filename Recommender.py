import numpy as np
import pandas as pd
from Wardrobe import Wardrobe
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D


class Recommender():

    def __init__ (self, wardrobe):
        self.wardrobe = wardrobe
        model = Sequential()

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


if __name__ == '__main__':
    w = Wardrobe()
    w.from_csv('./good_matts_wardrobe.csv')
    r = Recommender(w)
    print(r.getwd())



## IDEAS:
# the model algorithm is the same, but the occasion/weather can be treated as bias terms
# Bias can be thought of as multipliers that influence
# Can ask the user about "importance" factor, which can increase or decrease that multiplier
# As a apart of the initial screening phase, the user should be prompted to give initial preference data 
# (like favorite color or color combo or type of clothing) so the model can pretrain and give decent initial screening fits
