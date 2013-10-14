from rpython.jit.backend.llsupport.test.zrpy_gc_test import CompileFrameworkTests

# XXX TODO: have to actually get the translation settings right before
# this will run the js code; I think by default it will run platform tests?

class TestShadowStack(CompileFrameworkTests):
    gcrootfinder = "shadowstack"
