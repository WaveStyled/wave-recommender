####################
#
# Python Wardrobe Tester suite
#
# Uses pytest module as test framework
#
# INSTALL : python3 -m pip install pytest
#
# TO RUN: pytest -v wardrobe_tests.py
#
####################

from Wardrobe import Wardrobe as wd

wardrobe = wd()

def test_wardrobe_load():
    wardrobe.from_csv('./csv_files/good_matts_wardrobe.csv')
    assert 156 == len(wardrobe)

def test_wardrobe_add():
    toadd = (157, "TEST", "red", 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)
    wardrobe.addItem(toadd)
    assert 157 == len(wardrobe)
    assert all([test == wditem for (test, wditem)
               in zip(toadd, wardrobe.getItem(157))])

def test_wardrobe_delete():
    todelete = 2
    wardrobe.deleteItem(todelete)
    assert 156 == len(wardrobe)
    assert wardrobe.getItem(todelete) == None

def test_wardrobe_update():
    toupdate = (157, "UPDATE", "red", 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)
    old = wardrobe.getItem(157)
    wardrobe.updateItem(toupdate)
    assert all([test == wditem for (test, wditem)
               in zip(old, wardrobe.getItem(157))])


