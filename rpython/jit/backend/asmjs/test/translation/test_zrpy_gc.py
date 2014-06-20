import py
from rpython.jit.backend.llsupport.test.zrpy_gc_test import CompileFrameworkTests

class TestShadowStack(CompileFrameworkTests):
    compile_kwds = CompileFrameworkTests.compile_kwds.copy()
    compile_kwds.update({"platform": "emscripten", "backend": "js"})
    gcrootfinder = "shadowstack"
    gc = "incminimark"
