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

# tests that the wardrobe object loads the csv properly


def test_wardrobe_load():
    wardrobe.from_csv('./csv_files/good_matts_wardrobe.csv')
    assert 156 == len(wardrobe)


def test_wardrobe_init():
    test = wd()
    assert not len(test)


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


def test_wardrobe_delete_nonexistent():
    todelete = -1
    wardrobe.deleteItem(todelete)
    assert 157 == len(wardrobe)
    assert wardrobe.getItem(todelete) == None


def test_wardrobe_update():
    toupdate = (157, "UPDATE", "red", 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)
    old = wardrobe.getItem(157)
    wardrobe.updateItem(toupdate)
    assert all([test == wditem for (test, wditem)
               in zip(old, wardrobe.getItem(157))])


def test_wardrobe_gen():
    id = wardrobe.gen("oc_semi_formal", "we_snowy", "S")
    pieceids = wardrobe.getdf()['pieceid'].tolist()
    assert id in pieceids

    wd = wd()
    id = wardrobe.gen("blank", "test")
    assert id == -1


def test_wardrobe_gen_random():
    fit = wardrobe.gen_random("oc_workout", "we_cold")
    pieceids = wardrobe.getdf()['pieceid'].tolist()

    assert len(fit) == 7  # check that its a valid fit

    for f in fit:
        assert isinstance(f, int)
        assert f in pieceids


def test_wardrobe_get_random_fit():
    fits, meta = wardrobe.getRandomFit(5)

    assert meta[0] in wd.oc_mappings  # valid occasion and weather
    assert meta[1] in wd.we_mappings
    pieceids = wardrobe.getdf()['pieceid'].tolist()

    for fit in fits:  # valid fit checks
        assert len(fit) == 7

        for f in fit:
            assert isinstance(f, int)
            assert f in pieceids
