####################
#
# Python Recommender Tester suite
#
# Uses pytest module as test framework
#
# INSTALL : python3 -m pip install pytest
#
# TO RUN: pytest -v recommender_tests.py
#
####################

from typing_extensions import assert_never
from Recommender import Recommender
from os.path import exists
import pytest
import numpy as np

rec = Recommender()

## tests the model can load a csv file properly
def test_recommender_load():
    rec.from_csv('./csv_files/outfits.csv')
    assert len(rec.getdf()) == 108

## tests model normalization works for the pieceids
def test_recommender_normalize():
    rec.normalize()
    dataframe = rec.getdf()
    normalized = list(dataframe[['hat', 'shirt', 'sweater', 'jacket', 'bottom_layer', 'shoes', 'misc']].max())
    assert all([0 <= i <= 1 for i in normalized])

# tests that the model can be saved and reloaded
def test_recommender_model_save():
    rec.buildModel()
    rec.save_model('wavestyled')
    with pytest.raises(OSError) as excinfo:
        testing = Recommender()
        testing.load_model('wavestyled_dummy')
    assert "No file or directory found" in str(excinfo.value)

    dummy = Recommender()
    dummy.load_model('wavestyled')
    assert dummy.getModel() is not None

# REQUIRES A WARDROBE TO BE LOADED with USERID = 123
# tests that the training data generated by the model is done properly 
def test_recommender_train_data():
    rec.fromDB('hspkeleukrvdik9ct6oes98c7bu1')
    train, labels = rec.create_train()
    
    assert len(train) == len(labels)

    for l in labels:
        assert l[0] == 1 or l[0] == 0

    for train_input in train:
        fit, colors, metadata = train_input
        occasion, weather = metadata[:2]

        for f in fit:
            assert isinstance(f, int)

        for color in colors:
            assert color in Recommender.mappings

        assert 1 <= occasion <= 6
        assert 1 <= weather <= 5