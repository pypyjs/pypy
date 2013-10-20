
_orig_platform = None

def pytest_runtest_setup(item):
    from rpython.translator import platform
    global _orig_platform
    _orig_platform = platform.platform
    platform.set_platform("emscripten", None)

def pytest_runtest_teardown(item, nextitem):
    from rpython.translator import platform
    global _orig_platform
    platform.platform = _orig_platform

