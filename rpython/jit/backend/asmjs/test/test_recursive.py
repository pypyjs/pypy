import os
import py
from rpython.jit.metainterp.test.test_recursive import RecursiveTests
from rpython.jit.backend.asmjs.test.test_basic import JitASMJSMixin

class TestRecursive(JitASMJSMixin, RecursiveTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_recursive.py

    def test_directly_call_assembler_virtualizable_force1(self):
        if 'x86_64' in os.uname():
            py.test.skip("not working on x86_64 host system")

    def test_directly_call_assembler_virtualizable_force_blackhole(self):
        if 'x86_64' in os.uname():
            py.test.skip("not working on x86_64 host system")
