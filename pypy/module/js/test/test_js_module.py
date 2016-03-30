from __future__ import with_statement

import os

import py
try:
    import PyV8
except ImportError:
    py.test.skip("PyV8 module not available")

from pypy.interpreter.gateway import unwrap_spec, interp2app


class AppTestJS(object):

    spaceconfig = {
        "usemodules": ["js"]
    }

    def test_py_to_js_conversion(self):
        import js
        assert isinstance(js.convert("hello"), js.String)
        assert isinstance(js.convert(u"\u2603"), js.String)
        assert isinstance(js.convert(42), js.Number)
        assert isinstance(js.convert(42.5), js.Number)
        assert isinstance(js.convert(True), js.Boolean)
        assert isinstance(js.convert(False), js.Boolean)
        assert isinstance(js.convert(None), js.Object)
        def func():
            pass
        assert isinstance(js.convert(func), js.Function)
        class cls():
            def meth(self):
                pass
        assert isinstance(js.convert(cls.meth), js.Function)
        assert isinstance(js.convert(cls().meth), js.Function)

    def test_py_to_js_dict_conversion(self):
        import js
        jsobj = js.convert({b'foo': 'bar'})
        assert isinstance(jsobj, js.Object)
        assert jsobj.foo == 'bar'
        jsobj = js.convert({u'foo': 'bar'})
        assert isinstance(jsobj, js.Object)
        assert jsobj.foo == 'bar'

    def test_py_to_js_list_conversion(self):
        import js
        jsobj = js.convert([1, 2, 3])
        assert jsobj[0] == 1
        assert jsobj[1] == 2
        assert jsobj[2] == 3

    def test_js_string(self):
        import js
        # Basic ASCII strings work as expected.
        s = js.String("hello world")
        assert str(s) == "hello world"
        assert len(s) == 11
        # Unicode in, unicode out.
        s = js.String(u"hello \u2603")
        assert unicode(s) == u"hello \u2603"

    def test_js_number(self):
        import js
        s = js.Number(42)
        assert s == 42
        assert str(s) == "42"
        assert s + 1 == 43
        assert s * 2 == 84
        assert s + js.Number(17) == js.Number(59)
        assert 1 + s == 43
        assert 2 * s == 84
        assert js.Number(17) + s == js.Number(59)

    def test_js_object(self):
        import js
        s = js.Object()
        assert str(s) == "[object Object]"

    def test_js_truthiness(self):
        import js
        assert not js.null
        assert not js.undefined
        assert not js.false
        assert not js.Number(0)
        assert not js.String("")
        assert js.true
        assert js.Number(12)
        assert js.String("xyz")

    def test_js_array_access(self):
        import js
        a = js.eval("[1,2,3]")
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 3
        assert a.length == len(a) == 3

    def test_js_object_iteration(self):
        import js
        obj = js.Object()
        obj["a"] = 1
        obj["b"] = 2
        obj["c"] = 3
        assert sorted(list(obj)) == ["a", "b", "c"]

    def test_js_object_prototype_semantics(self):
        import js
        proto = js.Object()
        proto["z"] = 9
        obj = js.Object()
        obj["a"] = 1
        obj["b"] = 2
        obj["c"] = 3
        obj["__proto__"] = proto
        assert sorted(list(obj)) == ["a", "b", "c"]
        assert sorted(dir(obj)) == ["a", "b", "c", "z"]
        assert obj["a"] == 1
        assert obj["b"] == 2
        assert obj["c"] == 3
        assert obj["z"] == 9

    def test_eval_operates_at_global_scope(self):
        import js
        js.eval("var x = 12")
        assert js.globals.x == 12
