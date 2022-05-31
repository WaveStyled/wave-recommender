####################
#
# Python User and Userbase Tester suite
#
# Uses pytest module as test framework
#
# INSTALL : python3 -m pip install pytest
#
# TO RUN: pytest -v user_tests.py
#
####################

from User import User, UserBase

user = User()
userBase = UserBase()


def test_user_load():
    user.wardrobe_init()
    assert len(user.getWD()) == 156


def test_user_wd_operations():
    toadd = (157, "TEST", "red", 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)
    user.addWDItem(toadd)

    assert len(user.getWD()) == 157
    assert all([test == wditem for (test, wditem)
               in zip(toadd, user.getWD().getItem(157))])

    toupdate = (157, "UPDATED", "red", 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)
    old = user.getWD().getItem(157)
    user.updateWDItem(toupdate)
    assert all([test == wditem for (test, wditem)
               in zip(toupdate, user.getWD().getItem(157))])

    toremove = 140
    user.removeWDItem(toremove)
    assert len(user.getWD()) == 156
    assert user.getWD().getItem(140) is None


def test_user_calibration():
    fits = user.begin_calibration(5)

    assert len(fits) <= 5
    assert all([len(fit) == 7 for fit in fits])


def test_end_calibration():
    test_ratings = [1, 0, 1, 0]
    test_attr = [("oc_typical", "we_rainy"), ("oc_formal", "we_cold"),
                 ("oc_semi_formal", "we_hot"), ("oc_workout", "we_hot")]
    test_fits = [[0, 0, 1, 23, 0, 145, 9], [1, 0, 3, 4, 5, 2, 0],
                 [0, 0, 0, 0, 1, 4, 78], [145, 9, 0, 0, 80, 90, 4]]

    user.end_calibration(ratings=test_ratings, attrs=test_attr, outfits=test_fits)
    oldrec = user.getModel()
    user.update_preferences()
    newrec = user.getModel()

    assert len(oldrec) == len(newrec) - 4
    
####

def test_userbase_load():
    userBase.add_new_user("123")
    assert len(userBase.get_userbase()) == 1

    userBase.get_user("123")
    assert len(userBase.get_userbase()) == 1

    userBase.get_user("1234")
    assert len(userBase.get_userbase()) == 2
