import pytest



def test_import():
    # you need to import ngsolve before importing the extension
    # such that all runtime dependencies are loaded
    import ngsolve
    import ngsdiffgeo



if __name__ == "__main__":
    test_import()
    print("All tests passed!")