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

from Recommender import Recommender

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

