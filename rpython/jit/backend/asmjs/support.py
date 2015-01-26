"""

Support functions for implementing and testing the asmjs JIT backend.

This module defines the C interfaces for the asmjs JIT supporting functions,
such as jitCompile() and jitInvoke().  It also implements support for loading
the corresponding javascript code and executing it during the tests.

"""

from __future__ import with_statement

import os
import ctypes
import struct

from rpython.rtyper.lltypesystem import lltype, rffi, ll2ctypes
from rpython.rtyper.tool import rffi_platform
from rpython.translator.tool.cbuild import ExternalCompilationInfo
from rpython.translator.platform import emscripten_platform


emscripten_include_dirs = [
    os.path.dirname(os.path.abspath(emscripten_platform.__file__)),
]


eci = ExternalCompilationInfo(
    includes=['library_jit.h'],
    include_dirs=emscripten_include_dirs,
)


def jsexternal(args_t, result_t, **kwds):
    """Decorator to define stubbed-out external javascript functions.

    This decorator can be applied to a python function to register it as
    the stubbed-out implementation of an external javascript function.
    The llinterpreter will run the python code, the compiled interpreter
    will link to the javascript function of the same name.
    """
    def do_register(func):
        def jsfunc(*args):
            ctx = load_javascript_ctx()
            with ctx:
                print "CALL JS FUNC", func.__name__, args
                refs = []
                conv_args = list(args)
                for i, arg in enumerate(args):
                    if args_t[i] == rffi.CCHARP:
                         p = ctypes.c_char_p("".join(arg))
                         refs.append(p)
                         conv_args[i] = int(str(p).split("(")[-1].split(")")[0])
                print refs
                print "CALL JS FUNC CONV", func.__name__, conv_args
                return getattr(ctx.locals, "_"+func.__name__)(*conv_args)
        kwds.setdefault('_callable', jsfunc)
        kwds.setdefault('random_effects_on_gcobjs', False)
        kwds.setdefault('compilation_info', eci)
        return rffi.llexternal(func.__name__, args_t, result_t, **kwds)
    return do_register



@jsexternal([rffi.CCHARP], rffi.INT)
def jitCompile(jssource):
    raise NotImplementedError


@jsexternal([], rffi.INT)
def jitReserve():
    raise NotImplementedError


@jsexternal([rffi.INT], rffi.INT)
def jitExists(funcid):
    raise NotImplementedError


@jsexternal([rffi.INT, rffi.CCHARP], lltype.Void)
def jitRecompile(funcid, jssource):
    raise NotImplementedError


@jsexternal([rffi.INT, rffi.INT], lltype.Void)
def jitCopy(srcId, dstId):
    raise NotImplementedError


@jsexternal([rffi.INT, rffi.INT, rffi.INT], rffi.INT,
            _nowrapper=True, random_effects_on_gcobjs=True)
def jitInvoke(funcid, frame, label):
    raise NotImplementedError


@jsexternal([rffi.INT], lltype.Void)
def jitFree(funcid):
    raise NotImplementedError


# This will on-demand load the javascript library for the implementation
# of all supporting functions.  it uses the PyV8 engine to run the javascript.

ctx = None

def load_javascript_ctx():
    global ctx
    if ctx is not None:
        return ctx

    import PyV8
    ctx = PyV8.JSContext()

    with ctx:

        # Simulate in some of the emscripten runtime environment.

        def mergeInto(dst, src):
            pass

        def debug(*args):
            print args

        ctx.locals.LibraryManager = { "library": {} }
        ctx.locals.mergeInto = mergeInto
        ctx.locals.debug = debug
        ctx.locals.Int8Array = Int8Array = makeHeapViewClass("b", ToInt8)
        ctx.locals.Int16Array = makeHeapViewClass("h", ToInt16)
        ctx.locals.Int32Array = makeHeapViewClass("i", ToInt32)
        ctx.locals.Uint8Array = makeHeapViewClass("B", ToUInt8)
        ctx.locals.Uint16Array = makeHeapViewClass("H", ToUInt16)
        ctx.locals.Uint32Array = makeHeapViewClass("I", ToUInt32)
        ctx.locals.Float32Array = makeHeapViewClass("f", float)
        ctx.locals.Float64Array = makeHeapViewClass("d", float)
        ctx.locals.buffer = buffer = NativeHeap()
        ctx.locals.HEAP8 = Int8Array(buffer)
        ctx.locals.Module = NativeModule(buffer)

        # Load the library source in this environment.

        srcfile = os.path.join(
            os.path.dirname(__file__),
            "library_jit.js",
        )
        with open(srcfile, "r") as f:
            src = f.read()
        ctx.eval(src)

        # Apply emscripten name-mangling rules.

        lib_jit = ctx.locals.LibraryJIT
        for nm in dir(lib_jit):
            if nm.startswith("$"):
                setattr(ctx.locals, nm[1:], getattr(lib_jit, nm))
            elif nm.startswith("jit"):
                setattr(ctx.locals, "_"+nm, getattr(lib_jit, nm))

    return ctx



def ToInt32(v):
    return ctypes.c_int32(int(v)).value


def ToInt16(v):
    return ctypes.c_int16(int(v)).value


def ToInt8(v):
    return ctypes.c_int8(int(v)).value


def ToUInt32(v):
    return ctypes.c_uint32(int(v)).value


def ToUInt16(v):
    return ctypes.c_uint16(int(v)).value


def ToUInt8(v):
    return ctypes.c_uint8(int(v)).value



class NativeHeap(object):
    """A python re-implementation of the asmjs heap memory object.

    This is a very basic re-implementation of the asmjs heap memory object,
    exposing the memory space of the current process as a simple array of
    bytes. Tremendously slow and unsafe! But it lets us run the tests.
    """

    @staticmethod
    def validate_addr(addr):
        # XXX TODO: how to quickly check for obviously-bad pointers?
        return ctypes.c_uint32(int(addr)).value

    def memmove(self, dst, src, size):
        dst = self.validate_addr(dst)
        src = self.validate_addr(src)
        return ctypes.memmove(dst, src, int(size))

    def __getitem__(self, addr):
        print "READMEM", addr
        return ctypes.string_at(self.validate_addr(addr), 1)

    def __setitem__(self, addr, char):
        print "WRITEMEM", addr
        ctypes.memmove(self.validate_addr(addr), char, 1)

    def __len__(self):
        print "ASKING HEAP MEMORY SIZE"
        # Length is required so that PyV8 can treat this like a JS array.
        return 2**30


def makeHeapViewClass(format, convert=lambda x: x):
    """Given native-endian struct pack format, build a heapview class.

    This is a helper function to generate python equivalents of the javascript
    typed-array classes e.g. Int8Array, Int32Array etc. Given a native-endian
    struct format string, it produces a wrapper class that will load and store
    objects of that type from an underlying array of bytes.
    """

    itemsize = struct.calcsize(format)

    class _HeapView(object):

        def __init__(self, heap):
            self.heap = heap

        def __getitem__(self, addr):
            print "GETITEM VIEW", format, addr
            data = ""
            for i in xrange(itemsize):
                data += self.heap[(itemsize * addr) + i]
            print "  DATA", repr(data)
            return struct.unpack(format, data)[0]

        def __setitem__(self, addr, value):
            print "SETITEM VIEW", addr, value, convert(value)
            data = struct.pack(format, convert(value))
            print "  DATA", repr(data)
            for i in xrange(len(data)):
                self.heap[(itemsize * addr) + i] = data[i]

        def __len__(self):
            print "ASKING HEAP VIEW MEMORY SIZE"
            return len(self.heap)

    return _HeapView


class NativeModule(object):
    """A python re-implementation of the emscripten Module object.

    This class provides a very limited subset of the functionality of
    emscripten's "Module" object, but refering to the native python runtime
    rather than a host javascript interpreter.

    In particular, it provides dynCall_XXX methods that can invoke native
    functions by pointer.
    """

    TYPEMAP = {"v": None, "i": ctypes.c_int32, "d": ctypes.c_double}

    # We need some 8-byte-aligned memory to use as temporary storage
    # when loading/storing unaligned floats.

    tempDoubleStorage = ctypes.create_string_buffer(12)
    tempDoublePtr = ctypes.addressof(tempDoubleStorage)
    if tempDoublePtr % 8 != 0:
        assert tempDoublePtr % 8 == 4
        tempDoublePtr += 4

    def __init__(self, heap):
        self._heap = heap

    def _memcpy(self, dst, src, size):
        return self._heap.memmove(dst, src, size)

    def _gettimeofday(self, ptr):
        raise NotImplementedError

    def _abort(self):
        raise RuntimeError("ABORT")

    def __getattr__(self, name):
        # On-demand generation of dynCall_XXX methods.
        # Emscripten provides one such method per type signature of function.
        # We provide a matching method on-demand, that dispatches to an
        # in-process function pointer via the magic of ctypes.
        print "GETATTR", name
        if name.startswith("dynCall_"):
            callsig = name[len("dynCall_"):]
            restype = self.TYPEMAP[callsig[0]]
            argtypes = [self.TYPEMAP[c] for c in callsig[1:]]
            argconvs = [int if c == "i" else lambda a:a for c in callsig[1:]]
            functype = ctypes.CFUNCTYPE(restype, *argtypes)

            def dynCall(funcaddr, *args):
                funcaddr = self._heap.validate_addr(funcaddr)
                args = [conv(a) for (conv, a) in zip(argconvs, args)]
                # We cheat slightly here.  The ctypes callback won't
                # raise any exceptions, but it will record them in a
                # special private area of the ll2ctypes module.
                res = functype(funcaddr)(*args)
                if ll2ctypes._callback_exc_info is not None:
                    exc_typ, exc_val, exc_tb = ll2ctypes._callback_exc_info
                    ll2ctypes._callback_exc_info = None
                    raise exc_typ, exc_val, exc_tb
                if callsig[0] != "v":
                    return float(res)

            return dynCall

        # Allow calling our own exported functions via ffi.
        if name.startswith("_jit") and name[1:] in globals():
            func = globals()[name[1:]]

            def callSupportFunc(*args):
                return func(*map(int, args))

            return callSupportFunc

        print "Unknown foreign function: %s" % (name,)
        raise AttributeError("Unknown foreign function: %s" % (name,))

