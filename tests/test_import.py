import rec_engine


def test_rec_engine_import_succeeds() -> None:
    assert rec_engine.__name__ == "rec_engine"
