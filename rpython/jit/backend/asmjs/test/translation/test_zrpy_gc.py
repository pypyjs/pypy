import py
from rpython.jit.backend.llsupport.test.zrpy_gc_test import CompileFrameworkTests

class TestShadowStack(CompileFrameworkTests):
    compile_kwds = CompileFrameworkTests.compile_kwds.copy()
    compile_kwds.update({"platform": "emscripten", "backend": "js"})
    gcrootfinder = "shadowstack"

#    def test_compile_framework_8(self):
#        py.test.xfail("wtf?")

#    def test_compile_framework_external_exception_handling(self):
#        py.test.xfail("wtf?")

