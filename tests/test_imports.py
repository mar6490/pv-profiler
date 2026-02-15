"""Basic import tests for package scaffold."""


def test_import_package() -> None:
    import pv_profiler

    assert pv_profiler is not None


def test_import_cli() -> None:
    from pv_profiler import cli

    assert cli is not None
