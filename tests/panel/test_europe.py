from src.panel.io import load_europe


def test_mapping_covers_every_country_in_the_source():
    assert len(load_europe()) == 210


def test_unambiguous_cases_are_classified_correctly():
    m = load_europe()
    for c in ["Italy", "France", "Germany", "Spain", "Poland", "Sweden"]:
        assert m[c] is True, c
    for c in ["United States", "China", "Brazil", "Australia", "Nigeria"]:
        assert m[c] is False, c


def test_every_country_has_a_boolean_verdict():
    assert all(isinstance(v, bool) for v in load_europe().values())
