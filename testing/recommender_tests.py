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


def test_recommender_load():
    rec.from_csv('./csv_files/outfits.csv')
    assert len(rec.getdf()) == 108


def test_recommender_normalize():
    rec.normalize()
    dataframe = rec.getdf()
    assert dataframe.max() <= 1 and dataframe.max() >= 0


def test_recommender_encode_colors():
    rec.encode_colors()
    assert set(rec.mappings) == rec.mappings


def test_recommender_model_save():
    rec.save_model('wavestyled')
    with pytest.raises(FileNotFoundError) as excinfo:
        rec.load_model('wavestyled_dummy')
    assert "File Not Found" in str(excinfo.value)

    dummy = Recommender()
    dummy.load_model('wavestyled')
    assert rec.equals(dummy) == True


def test_recommender_train_data():
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
        