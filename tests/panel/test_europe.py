from src.panel.io import load_europe


def test_mapping_covers_every_country_in_the_source():
    assert len(load_europe()) == 210


def test_unambiguous_cases_are_classified_correctly():
    m = load_europe()
    for c in ["Italy", "France", "Germany", "Spain", "Poland", "Sweden"]:
        assert m[c] is True, c
    for c in ["United States", "China", "Brazil", "Australia", "Nigeria"]:
        assert m[c] is False, c


def test_names_countrycode_cannot_place_are_neither_europe_nor_outside():
    # None, not False: N_Europe counts neither, but N_Outside_Europe counts
    # only False, and treating these four as False overcounts it on 44 rows.
    m = load_europe()
    for c in ["Kosovo", "Polynesia", "Micronesia", "British Indian Ocean Territory"]:
        assert m[c] is None, c


def test_every_country_has_a_verdict():
    assert all(v in (True, False, None) for v in load_europe().values())
