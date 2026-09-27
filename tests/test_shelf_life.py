from banana_ai.services.shelf_life import prototype_estimate


def test_prototype_estimate():
    assert prototype_estimate("rotten") == "0 day(s)"
    assert "days" in prototype_estimate("unripe")
