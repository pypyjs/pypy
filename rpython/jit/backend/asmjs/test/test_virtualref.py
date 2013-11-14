import os
import py
from rpython.jit.metainterp.test.test_virtualref import VRefTests
from rpython.jit.backend.asmjs.test.test_basic import JitASMJSMixin

class TestVRef(JitASMJSMixin, VRefTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_virtualref.py

    def test_alloc_virtualref_and_then_alloc_structure(self):
        if 'x86_64' in os.uname():
            py.test.skip("not working on x86_64 host system")

