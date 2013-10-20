def pytest_runtest_setup(item):
    from rpython.rtyper.lltypesystem import ll2ctypes
    try:
        ll2ctypes.do_allocation_in_far_regions()
    except AttributeError:
        # This doesn't work so well when platform=="emscripten".
        # No harm, no foul.
        pass
