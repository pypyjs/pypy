def pytest_runtest_setup(item):
    from rpython.rtyper.lltypesystem import ll2ctypes
    ll2ctypes.do_allocation_in_far_regions()
