
from __future__ import with_statement

import os

from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rtyper.tool import rffi_platform
from rpython.translator.tool.cbuild import ExternalCompilationInfo
from rpython.translator.platform import emscripten_platform


# Parse the C include files for constants, type defns, etc.

emscripten_include_dirs = [
    os.path.dirname(os.path.abspath(emscripten_platform.__file__)),
]


eci = ExternalCompilationInfo(
    includes=['emjs.h'],
    include_dirs=emscripten_include_dirs,
)


CONSTANTS = [
    'EMJS_ERROR', 'EMJS_OK', 'EMJS_UNDEFINED', 'EMJS_NULL', 'EMJS_FALSE',
    'EMJS_TRUE', 'EMJS_TYPE_ERROR', 'EMJS_TYPE_UNDEFINED', 'EMJS_TYPE_BOOLEAN',
    'EMJS_TYPE_NUMBER', 'EMJS_TYPE_STRING', 'EMJS_TYPE_OBJECT',
    'EMJS_TYPE_FUNCTION',
]


class CConfig:
    _compilation_info_ = eci
    size_t = rffi_platform.SimpleType("size_t", rffi.ULONG)
    emjs_handle = rffi_platform.SimpleType("emjs_handle", rffi.LONG)
    emjs_type = rffi_platform.SimpleType("emjs_type", rffi.LONG)


for constant in CONSTANTS:
    setattr(CConfig, constant, rffi_platform.ConstantInteger(constant))


class cConfig:
    pass


for k, v in rffi_platform.configure(CConfig).items():
    setattr(cConfig, k, v)

for constant in CONSTANTS:
    locals()[constant] = getattr(cConfig, constant)
del constant, CONSTANTS

SIZE_T_TP = cConfig.size_t
EMJS_HANDLE_TP = cConfig.emjs_handle
EMJS_TYPE_TP = cConfig.emjs_type
CALLBACK_TP = rffi.CCallback([rffi.VOIDP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
CALLBACK_V_TP = rffi.CCallback([rffi.VOIDP, EMJS_HANDLE_TP], lltype.Void)


# This will on-demand load the javascript library for the implementation
# of all supporting functions.  it uses the PyV8 engine to run the javascript.

ctx = None
_strings = {}
_string_id = 1

def load_javascript_ctx():
    global ctx
    if ctx is not None:
        return ctx

    import PyV8
    ctx = PyV8.JSContext()

    with ctx:

        # Simulate in some of the emscripten runtime environment.
        # The tricky part here is handling string reads/writes.
        # For now we maintain our own int<->string mapping, but it should
        # be possible to use cast_ptr_to_int and friends aehre.

        def Pointer_stringify(ptr, *args):
            string = _strings[ptr]
            string = "".join(string)
            if args:
                string = string[:args[0]]
            while string[-1] == "\x00":
                string = string[:-1]
            return string

        def intArrayFromString(string, noNull=False):
            intArray = []
            for c in string:
                intArray.append(ord(c))
            if not noNull:
                intArray.apend(0)
            return PyV8.JSArray(intArray)

        def doSetChar(ptr, idx, char):
            string = _strings[ptr]
            string[idx] = chr(char)

        def mergeInto(dst, src):
            pass

        def debug(*args):
            print args

        class Runtime:

            def dynCall(self, callsig, fnptr, args):
                assert callsig == "iii" or callsig == "vii"
                return fnptr(*args)

    
        ctx.locals.LibraryManager = { "library": {} }
        ctx.locals.Pointer_stringify = Pointer_stringify
        ctx.locals.intArrayFromString = intArrayFromString
        ctx.locals.doSetChar = doSetChar
        ctx.locals.mergeInto = mergeInto
        ctx.locals.Runtime = Runtime()
        ctx.locals.debug = debug

        # Load the library source in this environment.

        srcfile = os.path.join(
            os.path.dirname(emscripten_platform.__file__),
            "library_emjs.js",
        )
        with open(srcfile, "r") as f:
            src = f.read()
            src = src.replace("makeSetValue('bufptr', 'i', 'chr', 'i8')",
                              "doSetChar(bufptr, i, chr)")
            ctx.eval(src)

        # Apply emscripten name-mangling rules.

        lib_emjs = ctx.locals.LibraryEMJS
        for nm in dir(lib_emjs):
            if nm.startswith("$"):
                setattr(ctx.locals, nm[1:], getattr(lib_emjs, nm))
            elif nm.startswith("emjs_"):
                setattr(ctx.locals, "_"+nm, getattr(lib_emjs, nm))

    return ctx


# Declare each function as a python wrapper to the js implementation,
# with appropriate external info for the build process.

def jsexternal(args_t, result_t, **kwds):
    """Decorator to declare javascript functions from library_emjs.

    This decorator can be applied to a stub python function to register it as
    placeholder for an external javascript function.  The llinterpreter will
    be redirected to run the loadted javascript code for that function, while
    the compiled interpreter will link to the javascript function of the same
    name at runtime.
    """
    def do_register(func):
        def jsfunc(*args):
            global _string_id
            ctx = load_javascript_ctx()
            args = list(args)
            my_strings = []
            for i, arg in enumerate(args):
                if args_t[i] == rffi.CCHARP:
                    _strings[_string_id] = arg
                    args[i] = _string_id
                    my_strings.append(_string_id)
                    _string_id += 1
            with ctx:
                return getattr(ctx.locals, "_"+func.__name__)(*args)
        kwds.setdefault('_callable', jsfunc)
        kwds.setdefault('random_effects_on_gcobjs', False)
        kwds.setdefault('_nowrapper', True)
        kwds.setdefault('compilation_info', eci)
        return rffi.llexternal(func.__name__, args_t, result_t, **kwds)
    return do_register


@jsexternal([EMJS_HANDLE_TP], lltype.Void)
def emjs_free(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_dup(h):
    raise NotImplementedError


@jsexternal([], EMJS_HANDLE_TP)
def emjs_globals():
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_prop_get(h_obj, h_prop):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_prop_set(h_obj, h_prop, h_val):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_prop_delete(h_obj, h_prop):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True)
def emjs_prop_apply(h_obj, h_prop, h_args):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, lltype.Signed], EMJS_HANDLE_TP)
def emjs_prop_get_int(h_obj, idx):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_prop_set_int(h_obj, idx, h_prop):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, lltype.Signed], EMJS_HANDLE_TP)
def emjs_prop_delete_int(h_obj, idx):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True)
def emjs_prop_apply_int(h_obj, idx, h_args):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP], EMJS_HANDLE_TP,
            _nowrapper=False)
def emjs_prop_get_str(h_obj, name):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            _nowrapper=False)
def emjs_prop_set_str(h_obj, name, h_val):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP], EMJS_HANDLE_TP,
            _nowrapper=False)
def emjs_prop_delete_str(h_obj, name):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True, _nowrapper=False)
def emjs_prop_apply_str(h_obj, name, h_args):
    raise NotImplementedError


@jsexternal([rffi.CCHARP], EMJS_HANDLE_TP,
            _nowrapper=False)
def emjs_make_str(value):
    raise NotImplementedError


@jsexternal([rffi.CCHARP, SIZE_T_TP], EMJS_HANDLE_TP,
            _nowrapper=False)
def emjs_make_strn(value):
    raise NotImplementedError


@jsexternal([lltype.Signed], EMJS_HANDLE_TP)
def emjs_make_int32(value):
    raise NotImplementedError
                          
                           
@jsexternal([rffi.DOUBLE], EMJS_HANDLE_TP)
def emjs_make_double(value):
    raise NotImplementedError


@jsexternal([lltype.Signed], EMJS_HANDLE_TP)
def emjs_make_bool(value):
    raise NotImplementedError


@jsexternal([], EMJS_HANDLE_TP)
def emjs_make_undefined():
    raise NotImplementedError


@jsexternal([], EMJS_HANDLE_TP)
def emjs_make_null():
    raise NotImplementedError


@jsexternal([], EMJS_HANDLE_TP)
def emjs_make_object():
    raise NotImplementedError


@jsexternal([lltype.Signed], EMJS_HANDLE_TP)
def emjs_make_array(size):
    raise NotImplementedError


@jsexternal([CALLBACK_TP, rffi.VOIDP], EMJS_HANDLE_TP)
def emjs_make_callback(callback, data):
    raise NotImplementedError

 
@jsexternal([], EMJS_HANDLE_TP)
def emjs_get_error():
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], lltype.Void)
def emjs_set_error(h):
    raise NotImplementedError


@jsexternal([], lltype.Void)
def emjs_clear_error():
    raise NotImplementedError


@jsexternal([rffi.CCHARP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True, _nowrapper=False)
def emjs_eval(value):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True)
def emjs_apply(h_fn, h_ctx, h_args):
    raise NotImplementedError
                      

@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP,
            random_effects_on_gcobjs=True)
def emjs_new(h_fn, h_args):
    raise NotImplementedError
                    
 
@jsexternal([EMJS_HANDLE_TP], EMJS_TYPE_TP)
def emjs_typeof(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, CALLBACK_V_TP, rffi.VOIDP], EMJS_HANDLE_TP)
def emjs_iter_all(h_obj, callback, data):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, CALLBACK_V_TP, rffi.VOIDP], EMJS_HANDLE_TP)
def emjs_iter_own(h_obj, callback, data):
    raise NotImplementedError
                         

@jsexternal([EMJS_HANDLE_TP], lltype.Signed)
def emjs_check(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_eq(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_neq(h_lhs, h_rhs):
    raise NotImplementedError
                       

@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_equiv(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_nequiv(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_gt(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_lt(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_gteq(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_lteq(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_add(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_sub(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_mul(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_div(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_mod(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_uplus(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_uminus(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_and(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_or(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_xor(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_lshift(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_rshift(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_urshift(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_op_bw_neg(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_in(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, EMJS_HANDLE_TP], lltype.Signed)
def emjs_op_instanceof(h_lhs, h_rhs):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], lltype.Signed)
def emjs_length(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_to_int32(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_to_uint32(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_to_double(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], EMJS_HANDLE_TP)
def emjs_to_string(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], lltype.Signed)
def emjs_read_int32(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP], lltype.Unsigned)
def emjs_read_uint32(h):
    raise NotImplementedError
                           

@jsexternal([EMJS_HANDLE_TP], rffi.DOUBLE)
def emjs_read_double(h):
    raise NotImplementedError


@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP], lltype.Signed)
def emjs_read_str(h, buf):
    raise NotImplementedError
                           

@jsexternal([EMJS_HANDLE_TP, rffi.CCHARP, lltype.Signed], lltype.Signed)
def emjs_read_strn(h, buf, maxlen):
    raise NotImplementedError
