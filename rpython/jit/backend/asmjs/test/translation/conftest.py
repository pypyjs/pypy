def pytest_configure(config):
    from rpython.translator import platform
    platform.set_platform("emscripten", None)
