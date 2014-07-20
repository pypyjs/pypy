#
# Hitlist:
#
#   * transparent conversion of primitive types; this is
#     particularly useful when defining callback functions.
#   * "Method" objects and nice handling of 'this'.
#   * split out the emjs_* functions to a separate file, add python stub
#     implementations for testing purposes.
#   * make String, Object, Function etc be proper subclasses of Value.
#   * maybe rename "Value" to "Handle"..?
#

from __future__ import with_statement

from rpython.rtyper.tool import rffi_platform
from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rtyper.tool import rffi_platform
from rpython.rlib.unroll import unrolling_iterable
from rpython.rlib.rarithmetic import intmask, is_emulated_long
from rpython.rlib.objectmodel import we_are_translated
from rpython.rlib.rmmap import alloc
from rpython.rlib.rdynload import dlopen, dlclose, dlsym, dlsym_byordinal
from rpython.rlib.rdynload import DLOpenError, DLLHANDLE
from rpython.rlib import rweakref
from rpython.rlib import jit
from rpython.rlib.objectmodel import specialize
from rpython.translator.tool.cbuild import ExternalCompilationInfo
from rpython.translator.platform import platform
from rpython.conftest import cdir
from platform import machine
import py
import os
import sys
import ctypes.util

from pypy.interpreter.error import OperationError
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.interpreter.function import Function as InterpFunction
from pypy.interpreter.function import Method as InterpMethod
from pypy.interpreter.gateway import interp2app, unwrap_spec, WrappedDefault



# First, we make the necessary declarations to import the "emjs" API.
# XXX TODO: move these into a "support" library of some kind, and add
# a pure-python dummy implementation for testing purposes.

eci = ExternalCompilationInfo(
    includes = ['emjs.h'],
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


def external(name, args, result, **kwds):
    return rffi.llexternal(name, args, result, compilation_info=eci, **kwds)


emjs_free = external('emjs_free',
                     [EMJS_HANDLE_TP],
                     lltype.Void)

emjs_dup = external('emjs_dup',
                    [EMJS_HANDLE_TP],
                    EMJS_HANDLE_TP)

emjs_globals = external('emjs_globals',
                        [],
                        EMJS_HANDLE_TP)

emjs_prop_get = external('emjs_prop_get',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_prop_set = external('emjs_prop_set',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_prop_delete = external('emjs_prop_delete',
                            [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                            EMJS_HANDLE_TP)

emjs_prop_apply = external('emjs_prop_apply',
                           [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                           EMJS_HANDLE_TP)

emjs_prop_get_int = external('emjs_prop_get_int',
                             [EMJS_HANDLE_TP, lltype.Signed],
                             EMJS_HANDLE_TP)

emjs_prop_set_int = external('emjs_prop_set_int',
                             [EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_prop_delete_int = external('emjs_prop_delete_int',
                                [EMJS_HANDLE_TP, lltype.Signed],
                                EMJS_HANDLE_TP)

emjs_prop_apply_int = external('emjs_prop_apply_int',
                               [EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP],
                               EMJS_HANDLE_TP)

emjs_prop_get_str = external('emjs_prop_get_str',
                             [EMJS_HANDLE_TP, rffi.CCHARP],
                             EMJS_HANDLE_TP)

emjs_prop_set_str = external('emjs_prop_set_str',
                             [EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_prop_delete_str = external('emjs_prop_delete_str',
                                [EMJS_HANDLE_TP, rffi.CCHARP],
                                EMJS_HANDLE_TP)

emjs_prop_apply_str = external('emjs_prop_apply_str',
                               [EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP],
                               EMJS_HANDLE_TP)

emjs_make_str = external('emjs_make_str',
                         [rffi.CCHARP],
                         EMJS_HANDLE_TP)

emjs_make_strn = external('emjs_make_strn',
                          [rffi.CCHARP, SIZE_T_TP],
                          EMJS_HANDLE_TP)

emjs_make_int32 = external('emjs_make_int32',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_double = external('emjs_make_double',
                           [rffi.DOUBLE],
                           EMJS_HANDLE_TP)

emjs_make_bool = external('emjs_make_bool',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_undefined = external('emjs_make_undefined',
                               [],
                               EMJS_HANDLE_TP)

emjs_make_null = external('emjs_make_null',
                          [],
                          EMJS_HANDLE_TP)

emjs_make_object = external('emjs_make_object',
                            [],
                            EMJS_HANDLE_TP)

emjs_make_array = external('emjs_make_array',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_callback = external('emjs_make_callback',
                              [CALLBACK_TP, rffi.VOIDP],
                              EMJS_HANDLE_TP)

emjs_get_error = external('emjs_get_error',
                          [],
                          EMJS_HANDLE_TP)

emjs_set_error = external('emjs_set_error',
                          [EMJS_HANDLE_TP],
                          lltype.Void)

emjs_clear_error = external('emjs_clear_error',
                            [],
                            lltype.Void)

emjs_eval = external('emjs_eval',
                     [rffi.CCHARP],
                     EMJS_HANDLE_TP)

emjs_apply = external('emjs_apply',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      EMJS_HANDLE_TP)

emjs_new = external('emjs_new',
                    [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                    EMJS_HANDLE_TP)

emjs_typeof = external('emjs_typeof',
                       [EMJS_HANDLE_TP],
                       EMJS_TYPE_TP)

emjs_iter_all = external('emjs_iter_all',
                         [EMJS_HANDLE_TP, CALLBACK_TP, rffi.VOIDP],
                         EMJS_HANDLE_TP)

emjs_iter_own = external('emjs_iter_own',
                         [EMJS_HANDLE_TP, CALLBACK_TP, rffi.VOIDP],
                         EMJS_HANDLE_TP)

emjs_check = external('emjs_check',
                      [EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_eq = external('emjs_op_eq',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_neq = external('emjs_op_neq',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       lltype.Signed)

emjs_op_equiv = external('emjs_op_equiv',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         lltype.Signed)

emjs_op_nequiv = external('emjs_op_nequiv',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          lltype.Signed)

emjs_op_gt = external('emjs_op_gt',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_lt = external('emjs_op_lt',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_gteq = external('emjs_op_gteq',
                        [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                        lltype.Signed)

emjs_op_lteq = external('emjs_op_lteq',
                        [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                        lltype.Signed)

emjs_op_add = external('emjs_op_add',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_sub = external('emjs_op_sub',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_mul = external('emjs_op_mul',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_div = external('emjs_op_div',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_mod = external('emjs_op_mod',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_uplus = external('emjs_op_uplus',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_op_uminus = external('emjs_op_uminus',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_and = external('emjs_op_bw_and',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_or = external('emjs_op_bw_or',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_op_bw_xor = external('emjs_op_bw_xor',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_lshift = external('emjs_op_bw_lshift',
                             [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_op_bw_rshift = external('emjs_op_bw_rshift',
                             [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_op_bw_urshift = external('emjs_op_bw_urshift',
                              [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                              EMJS_HANDLE_TP)

emjs_op_bw_neg = external('emjs_op_bw_neg',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_in = external('emjs_op_in',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_instanceof = external('emjs_op_instanceof',
                              [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                              lltype.Signed)

emjs_length = external('emjs_length',
                       [EMJS_HANDLE_TP],
                       lltype.Signed)

emjs_to_int32 = external('emjs_to_int32',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_to_uint32 = external('emjs_to_uint32',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_to_double = external('emjs_to_double',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_to_string = external('emjs_to_string',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_read_int32 = external('emjs_read_int32',
                           [EMJS_HANDLE_TP],
                           lltype.Signed)

emjs_read_uint32 = external('emjs_read_uint32',
                            [EMJS_HANDLE_TP],
                            lltype.Unsigned)

emjs_read_double = external('emjs_read_double',
                            [EMJS_HANDLE_TP],
                            rffi.DOUBLE)

emjs_read_str = external('emjs_read_str',
                         [EMJS_HANDLE_TP, rffi.CCHARP],
                         lltype.Signed)

emjs_read_strn = external('emjs_read_strn',
                         [EMJS_HANDLE_TP, rffi.CCHARP, SIZE_T_TP],
                         lltype.Signed)


# Now, we build a higher-level API on top of the emjs C API.
# We use objects with finalizers to automatically manage the freeeing of
# handles, and a bunch of magic methods to map the appropriate python-level
# operators into corresponding js-level operators.


class W_Value(W_Root):

    _immutable_fields_ = ['handle']

    def __init__(self, handle, callback=None):
        self.handle = handle
        self.callback = callback

    def __del__(self):
        emjs_free(self.handle)

    def descr__repr__(self, space):
        return space.wrap("<js.Value handle=%d>" % (self.handle,))

    # Expose a whole host of operators via app-level magic methods.

    def descr__float__(self, space):
        res = emjs_read_double(self.handle)
        return space.wrap(res)

    def descr__int__(self, space):
        res = emjs_read_int32(self.handle)
        return space.wrap(res)

    def descr__str__(self, space):
        h_str = emjs_to_string(self.handle)
        _check_error(space, h_str)
        bufsize = emjs_length(h_str)
        _check_error(space, bufsize)
        with rffi.scoped_alloc_buffer(bufsize) as buf:
            n = emjs_read_strn(h_str, buf.raw, buf.size)
            return space.wrap(buf.str(n))

    def descr__eq__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_equiv(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__ne__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_nequiv(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__lt__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_lt(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__le__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_lteq(self.handle, h_other)
        return space.newbool(bool(res))
        
    def descr__gt__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_gt(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__ge__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = emjs_op_gteq(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__bool__(self, space):
        res = emjs_check(self.handle)
        return space.newbool(bool(res))

    # The semntics of __getitem__ and __getattr__ are equivalent for
    # javascript objects.  We use __getattr__ to provide python-friendly
    # conveniences:
    #
    #    * raise AttributeError for undefined attributes
    #    * bind 'this' when loading callables as attributes
    #
    # This seems to give a good balance between js and python semantics.

    @unwrap_spec(name=str)
    def descr__getitem__(self, space, name):
        h_res = emjs_prop_get_str(self.handle, name)
        return _wrap_handle(space, h_res)

    @unwrap_spec(name=str)
    def descr__setitem__(self, space, name, w_value):
        with _unwrap_handle(space, w_value) as h_value:
            res = emjs_prop_set_str(self.handle, name, h_value)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__delitem__(self, space, name):
        res = emjs_prop_delete_str(self.handle, name)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__getattr__(self, space, name):
        h_res = emjs_prop_get_str(self.handle, name)
        if h_res == EMJS_UNDEFINED:
            raise OperationError(space.w_AttributeError, space.wrap(name))
        _check_error(space, h_res)
        if emjs_typeof(h_res) == EMJS_TYPE_FUNCTION:
            return space.wrap(W_Method(h_res, self))
        else:
            return space.wrap(W_Value(h_res))

    @unwrap_spec(name=str)
    def descr__setattr__(self, space, name, w_value):
        with _unwrap_handle(space, w_value) as h_value:
            res = emjs_prop_set_str(self.handle, name, h_value)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__delattr__(self, space, name):
        res = emjs_prop_delete_str(self.handle, name)
        _check_error(space, res)

    def descr__add__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_add(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__sub__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_sub(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__mul__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_mul(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__div__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_div(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__truediv__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_div(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__mod__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_mod(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__lshift__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_bw_lshift(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__rshift__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_bw_rshift(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__and__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_bw_and(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__or__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_bw_or(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__xor__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = emjs_op_bw_xor(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__neg__(self, space):
        h_res = emjs_op_uminus(self.handle)
        return _wrap_handle(space, h_res)

    def descr__pos__(self, space):
        h_res = emjs_op_uplus(self.handle)
        return _wrap_handle(space, h_res)

    def descr__invert__(self, space):
        h_res = emjs_op_bw_neg(self.handle)
        return _wrap_handle(space, h_res)

    def descr__contains__(self, space, w_item):
        with _unwrap_handle(space, w_item) as h_item:
            res = emjs_op_in(h_item, self.handle)
        _check_error(space, res)
        return space.newbool(bool(res))

    def descr__len__(self, space):
        res = emjs_length(self.handle)
        _check_error(space, res)
        return space.wrap(res)


class Cache:
    def __init__(self, space):
        self.w_error = space.new_exception_class("js.Error")


def _raise_error(space):
    err_h = emjs_get_error()
    if err_h != EMJS_UNDEFINED:
        emjs_clear_error()
        w_error = space.fromcache(Cache).w_error
        raise OperationError(w_error, _wrap_handle(space, err_h))


def _wrap_handle(space, handle, callback=None):
    if handle == EMJS_ERROR:
        _raise_error(space)
    return space.wrap(W_Value(handle, callback))


class _unwrap_handle(object):
    """Context-manager for unwrapping an app-level object to a js-level handle.

    This class can manage both existing W_Value instances, and the transient
    conversion of immutable app-level datatypes such as ints and strings.  It
    needs to be a context-manager to allow proper lifetime management of
    such transient handles.
    """

    def __init__(self, space, w_value):
        self.space = space
        self.w_value = w_value
        self.h_transient = EMJS_ERROR
        self.cb_transient = None

    def __enter__(self):
        space = self.space
        w_value = self.w_value
        # Optimistically assume that it's a proper W_Value instance.
        try:
            return space.interp_w(W_Value, w_value).handle
        except OperationError, e:
            if not e.match(space, space.w_TypeError):
                raise
            # Try to convert it to a transient W_Value instance.
            h_value = self._convert_transiently()
            if h_value == EMJS_ERROR:
                raise
            self.h_transient = h_value
            return h_value

    def __exit__(self, exc_typ, exc_val, exc_tb):
        if self.h_transient != EMJS_ERROR:
            emjs_free(self.h_transient)

    def _convert_transiently(self):
        space = self.space
        w_value = self.w_value
        if space.isinstance_w(w_value, space.w_int):
            return emjs_make_int32(space.int_w(w_value))
        if space.isinstance_w(w_value, space.w_long):
            return emjs_make_int32(space.int_w(w_value))
        if space.isinstance_w(w_value, space.w_float):
            return emjs_make_double(space.float_w(w_value))
        if space.isinstance_w(w_value, space.w_str):
            value = space.str_w(w_value)
            return emjs_make_strn(value, len(value))
        # XXX TODO: auto-convert functions to callbacks.
        #if space.isinstance_w(w_value, InterpFunction)):
        #    h_cb, cb = _make_callback(space, w_value, ())
        #    self.cb_transient = cb 
        #    return h_cb
        #if space.isinstance_w(w_value, space.wrap(InterpMethod)):
        #    h_cb, cb = _make_callback(space, w_value, ())
        #    self.cb_transient = cb 
        #    return h_cb
        return EMJS_ERROR


def _check_error(space, result):
    if result == EMJS_ERROR:
        _raise_error(space)


def globals(space):
    """Get a reference to the global scope 'this' object."""
    h = emjs_globals()
    return _wrap_handle(space, h)


@unwrap_spec(data=str)
def eval(space, data):
    """Evaluate javascript code at global scope."""
    h = emjs_eval(data)
    return _wrap_handle(space, h)


def new(space, w_fn, args_w):
    with _unwrap_handle(space, w_fn) as h_fn:
        h_args = emjs_make_array(len(args_w))
        _check_error(space, h_args)
        for i in xrange(len(args_w)):
            with _unwrap_handle(space, args_w[i]) as h_arg:
                res = emjs_prop_set_int(h_args, i, h_arg)
            _check_error(space, res)
        h_res = emjs_new(h_fn, h_args)
    return _wrap_handle(space, h_res)


def instanceof(space, w_lhs, w_rhs):
    with _unwrap_handle(space, w_lhs) as h_lhs:
        with  _unwrap_handle(space, w_rhs) as h_rhs:
            res = emjs_op_instanceof(h_lhs, h_rhs)
    _check_error(space, res)
    return space.newbool(bool(res))


def urshift(space, w_lhs, w_rhs):
    with _unwrap_handle(space, w_lhs) as h_lhs:
        with  _unwrap_handle(space, w_rhs) as h_rhs:
            h_res = emjs_op_bw_urshift(h_lhs, h_rhs)
    return _wrap_handle(space, h_res)


def uint32(space, w_value):
    with _unwrap_handle(space, w_value) as h_value:
        res = emjs_read_uint32(h_value)
    return space.wrap(res)


def Undefined(space):
    return _wrap_handle(space, EMJS_UNDEFINED)


def Boolean(space, w_value):
    if space.is_true(w_value):
        return _wrap_handle(space, EMJS_TRUE)
    else:
        return _wrap_handle(space, EMJS_FALSE)


def Number(space, w_value):
    value = space.float_w(w_value)
    h = emjs_make_double(value)
    return _wrap_handle(space, h)


@unwrap_spec(value=str)
def String(space, value):
    h = emjs_make_strn(value, len(value))
    return _wrap_handle(space, h)


def Object(space):
    # XXX TODO: kwargs for initial properties?
    h = emjs_make_object()
    return _wrap_handle(space, h)


@unwrap_spec(size=int)
def Array(space, w_items=None, size=0):
    h_res = emjs_make_array(size)
    _check_error(space, h_res)
    if space.is_true(w_items):
        w_iterator = space.iter(w_items)
        idx = 0
        # XXX TODO: baseobjspace uses a jit merge point in this loop...?
        while True:
            try:
                w_item = space.next(w_iterator)
            except OperationError, e:
                if not e.match(space, space.w_StopIteration):
                    raise
                break
            else:
                with _unwrap_handle(space, w_item) as h_item:
                    res = emjs_prop_set_int(h_res, idx, h_item)
                idx += 1
    return _wrap_handle(space, h_res)


def Function(space, w_callback, __args__):
    h_res, callback = _make_callback(space, w_callback, __args__)
    return _wrap_handle(space, h_res, callback)


class W_Method(W_Value):

    def __init__(self, handle, w_ctx):
        W_Value.__init__(self, handle)
        self.w_ctx = w_ctx


def _make_callback(space, w_callback, args):
    callback = Callback(space, w_callback, args)
    dataptr = rffi.cast(rffi.VOIDP, callback.id)
    ll_dispatch_callback = rffi.llhelper(CALLBACK_TP, dispatch_callback)
    return emjs_make_callback(ll_dispatch_callback, dataptr), callback


class Callback(object):

    def __init__(self, space, w_callback, __args__):
        self.id = global_callback_map.add(self)
        self.space = space
        self.w_callback = w_callback
        self.__args__ = __args__


class GlobalCallbackMap(object):
    """Some global storage to map numbers to wrapped callback data."""

    def __init__(self):
        self.next_id = 1
        self.callbacks = rweakref.RWeakValueDictionary(int, Callback)

    def add(self, w_value):
        id = self.next_id
        self.next_id += 1
        self.callbacks.set(id, w_value)
        return id

    def remove(self, id):
        self.callbacks.set(id, None)

    def get(self, id):
        return self.callbacks.get(id)


global_callback_map = GlobalCallbackMap()


def dispatch_callback(dataptr, h_args):
    id = rffi.cast(lltype.Signed, dataptr)
    callback = global_callback_map.get(id)
    if callback is None:
        raise RuntimeError("invalid callback id; was it garbage-collected?")
    space = callback.space
    # Unpack h_args as an extension of the default args from the callback.
    # XXX TODO: what to do about args.this?
    # XXX TODO: keyword arguments, and general cleanup of the below mess...
    args_w, kw_w = callback.__args__.unpack()
    if kw_w:
        raise RuntimeError("callback function kw args not implemented yet")
    h_args_len = emjs_length(h_args)
    all_args_w = [None] * (len(args_w) + h_args_len)
    for i, w_arg in enumerate(args_w):
        all_args_w[i] = w_arg
    for i in xrange(emjs_length(h_args)):
        w_arg = _wrap_handle(space, emjs_prop_get_int(h_args, i))
        all_args_w[i + len(args_w)] = w_arg
    try:
        w_res = space.call(callback.w_callback, space.newtuple(all_args_w))
    except OperationError, pyerr:
        # XXX TODO: allow the callback to raise js.Error and reflect it
        # properly through to the javascript side.
        # XXX TODO: tunnel the exception through JS and back to Python?
        # We could do this by e.g. throwing a special object...
        w_jserr = String(space, pyerr.errorstr(space))
        with _unwrap_handle(space, w_jserr) as h_jserr:
            emjs_set_error(h_jserr)
        return EMJS_ERROR
    else:
        # Note that the js-side callback stub frees the result handle,
        # so we have to dup it here to avoid breaking the w_res object.
        with _unwrap_handle(space, w_res) as h_res:
            return emjs_dup(h_res)


undefined = W_Value(EMJS_UNDEFINED)
null = W_Value(EMJS_NULL)
false = W_Value(EMJS_FALSE)
true = W_Value(EMJS_TRUE)


def W_Value_descr__call__(space, w_self, args_w):
    # XXX TODO: maybe we can pass 'this' as a keyword argument?
    self = space.interp_w(W_Value, w_self)
    h_args = emjs_make_array(len(args_w))
    _check_error(space, h_args)
    for i in xrange(len(args_w)):
        with _unwrap_handle(space, args_w[i]) as h_arg:
            emjs_prop_set_int(h_args, i, h_arg)
    h_res = emjs_apply(self.handle, EMJS_UNDEFINED, h_args)
    return _wrap_handle(space, h_res)


def W_Method_descr__call__(space, w_self, args_w):
    self = space.interp_w(W_Method, w_self)
    ctx = space.interp_w(W_Value, self.w_ctx)
    h_args = emjs_make_array(len(args_w))
    _check_error(space, h_args)
    for i in xrange(len(args_w)):
        with _unwrap_handle(space, args_w[i]) as h_arg:
            emjs_prop_set_int(h_args, i, h_arg)
    h_res = emjs_apply(self.handle, ctx.handle, h_args)
    return _wrap_handle(space, h_res)


W_Value.typedef = TypeDef(
    "Value",
    __doc__ = "Handle to an abstract JS value.",
    __call__ = interp2app(W_Value_descr__call__),
    __repr__ = interp2app(W_Value.descr__repr__),
    __str__ = interp2app(W_Value.descr__str__),
    __bool__ = interp2app(W_Value.descr__bool__),
    __float__ = interp2app(W_Value.descr__float__),
    __int__ = interp2app(W_Value.descr__int__),
    __eq__ = interp2app(W_Value.descr__eq__),
    __ne__ = interp2app(W_Value.descr__ne__),
    __lt__ = interp2app(W_Value.descr__lt__),
    __le__ = interp2app(W_Value.descr__le__),
    __gt__ = interp2app(W_Value.descr__gt__),
    __ge__ = interp2app(W_Value.descr__ge__),
    __getitem__ = interp2app(W_Value.descr__getitem__),
    __setitem__ = interp2app(W_Value.descr__setitem__),
    __delitem__ = interp2app(W_Value.descr__delitem__),
    __getattr__ = interp2app(W_Value.descr__getattr__),
    __setattr__ = interp2app(W_Value.descr__setattr__),
    __delattr__ = interp2app(W_Value.descr__delattr__),
    __add__ = interp2app(W_Value.descr__add__),
    __sub__ = interp2app(W_Value.descr__sub__),
    __mul__ = interp2app(W_Value.descr__mul__),
    __div__ = interp2app(W_Value.descr__div__),
    __truediv__ = interp2app(W_Value.descr__truediv__),
    __mod__ = interp2app(W_Value.descr__mod__),
    __lshift__ = interp2app(W_Value.descr__lshift__),
    __rshift__ = interp2app(W_Value.descr__rshift__),
    __and__ = interp2app(W_Value.descr__and__),
    __or__ = interp2app(W_Value.descr__or__),
    __xor__ = interp2app(W_Value.descr__xor__),
    __neg__ = interp2app(W_Value.descr__neg__),
    __pos__ = interp2app(W_Value.descr__pos__),
    __invert__ = interp2app(W_Value.descr__invert__),
    __contains__ = interp2app(W_Value.descr__contains__),
    # XXX TODO: move __len__ onto appropriate subclasses only
    __len__ = interp2app(W_Value.descr__len__),
    # TODO: __new__, for subtypes where it makes sense
    # TODO: __hash__, but only for immutable types
    # TODO: __dir__ and/or __iter__ for object types
    # TODO: swapped variants of the arithmetic operators?
)


W_Method.typedef = TypeDef(
    "Method",
    (W_Value.typedef,),
    __doc__ = "Handle to a JS function with bound context.",
    __call__ = interp2app(W_Method_descr__call__),
    # XXX TODO: expose the underlying function as a property?
    # I think there is a "GetSetProperty" typedef thing that can do this.
)

