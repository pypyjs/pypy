from __future__ import with_statement

import os

import py
py.test.skip("emjs testing support not implemented yet")

from pypy.interpreter.gateway import unwrap_spec, interp2app


class AppTestJS(object):

    spaceconfig = {
        "usemodules": ["js"]
    }

    def test_make_string(self):
        import js
        s = js.String("hello world")
        assert str(s) == "hello world"
