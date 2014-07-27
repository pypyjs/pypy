
from __future__ import with_statement

from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rlib import rweakref

import pypy.interpreter.function
from pypy.interpreter.error import OperationError
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.interpreter.gateway import interp2app, unwrap_spec

from pypy.module.js import support


class W_Value(W_Root):
    """Base class for js value handles.

    This is the base class for wrappe javascript value handles.  It cannot
    be instantiated directly from application code, but provides much of
    the base functionality that's common to all value types.
    """

    _immutable_fields_ = ['handle']

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        support.emjs_free(self.handle)

    def descr__repr__(self, space):
        return space.wrap("<js.Value handle=%d>" % (self.handle,))

    # Expose a whole host of operators via app-level magic methods.

    def descr__float__(self, space):
        res = support.emjs_read_double(self.handle)
        return space.wrap(res)

    def descr__int__(self, space):
        res = support.emjs_read_int32(self.handle)
        return space.wrap(res)

    def descr__str__(self, space):
        h_str = support.emjs_to_string(self.handle)
        _check_error(space, h_str)
        bufsize = support.emjs_length(h_str)
        _check_error(space, bufsize)
        with rffi.scoped_alloc_buffer(bufsize) as buf:
            n = support.emjs_read_strn(h_str, buf.raw, buf.size)
            return space.wrap(buf.str(n))

    # We expose === as default equality operator, for that I hope
    # are fairly obvious reasons....

    def descr__eq__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_equiv(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__ne__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_nequiv(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__lt__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_lt(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__le__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_lteq(self.handle, h_other)
        return space.newbool(bool(res))
        
    def descr__gt__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_gt(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__ge__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            res = support.emjs_op_gteq(self.handle, h_other)
        return space.newbool(bool(res))

    def descr__bool__(self, space):
        res = support.emjs_check(self.handle)
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
        h_res = support.emjs_prop_get_str(self.handle, name)
        return _wrap_handle(space, h_res)

    @unwrap_spec(name=str)
    def descr__setitem__(self, space, name, w_value):
        with _unwrap_handle(space, w_value) as h_value:
            res = support.emjs_prop_set_str(self.handle, name, h_value)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__delitem__(self, space, name):
        res = support.emjs_prop_delete_str(self.handle, name)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__getattr__(self, space, name):
        h_res = support.emjs_prop_get_str(self.handle, name)
        if h_res == support.EMJS_UNDEFINED:
            raise OperationError(space.w_AttributeError, space.wrap(name))
        w_res = _wrap_handle(space, h_res)
        # Turn functions into methods during attribute lookup.
        # XXX TODO: find a non-exception-driven check for this.
        try:
            func = space.interp_w(W_Function, w_res)
        except OperationError, e:
            if not e.match(space, space.w_TypeError):
                raise
        else:
            w_res = bind(space, w_res, space.wrap(self))
        return w_res

    @unwrap_spec(name=str)
    def descr__setattr__(self, space, name, w_value):
        with _unwrap_handle(space, w_value) as h_value:
            res = support.emjs_prop_set_str(self.handle, name, h_value)
        _check_error(space, res)

    @unwrap_spec(name=str)
    def descr__delattr__(self, space, name):
        res = support.emjs_prop_delete_str(self.handle, name)
        _check_error(space, res)

    def descr__add__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_add(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__sub__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_sub(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__mul__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_mul(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__div__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_div(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__truediv__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_div(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__mod__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_mod(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__lshift__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_bw_lshift(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__rshift__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_bw_rshift(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__and__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_bw_and(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__or__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_bw_or(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__xor__(self, space, w_other):
        with _unwrap_handle(space, w_other) as h_other:
            h_res = support.emjs_op_bw_xor(self.handle, h_other)
        return _wrap_handle(space, h_res)

    def descr__neg__(self, space):
        h_res = support.emjs_op_uminus(self.handle)
        return _wrap_handle(space, h_res)

    def descr__pos__(self, space):
        h_res = support.emjs_op_uplus(self.handle)
        return _wrap_handle(space, h_res)

    def descr__invert__(self, space):
        h_res = support.emjs_op_bw_neg(self.handle)
        return _wrap_handle(space, h_res)

    def descr__contains__(self, space, w_item):
        with _unwrap_handle(space, w_item) as h_item:
            res = support.emjs_op_in(h_item, self.handle)
        _check_error(space, res)
        return space.newbool(bool(res))


W_Value.typedef = TypeDef(
    "Value",
    __doc__ = "Handle to an abstract JS value.",
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
    # TODO: __hash__, but only for immutable types
    # TODO: __dir__ and/or __iter__ for object types
)


class W_Undefined(W_Value):
    """W_Value subclass for the singleton 'undefined'.

    This class provides no additional methods, and exists mainly to allow
    python code to typecheck for undefined.
    """

    def descr__repr__(self, space):
        return space.wrap("<js.Undefined>")


def W_Undefined_descr__new__(space, w_subtype):
    # The constructor always returns the singleton instance.
    return space.wrap(undefined)


W_Undefined.typedef = TypeDef(
    "Undefined",
    (W_Value.typedef,),
    __doc__ = "Handle to the JS singleton 'undefined'.",
    __new__ = interp2app(W_Undefined_descr__new__),
    __repr__ = interp2app(W_Undefined.descr__repr__),
)


class W_Boolean(W_Value):
    """W_Value subclass for singleton booleans 'true' and 'false'.

    This class provides no additional methods, and exists mainly to allow
    python code to typecheck for booleans.
    """

    def descr__repr__(self, space):
        if self.handle == support.EMJS_TRUE:
            return space.wrap("<js.Boolean true>")
        else:
            return space.wrap("<js.Boolean false>")


def W_Boolean_descr__new__(space, w_subtype, w_value):
    if space.is_true(w_value):
        return true
    else:
        return false


W_Boolean.typedef = TypeDef(
    "Boolean",
    (W_Value.typedef,),
    __doc__ = "Handle to a JS boolean value.",
    __new__ = interp2app(W_Boolean_descr__new__),
    __repr__ = interp2app(W_Boolean.descr__repr__),
)


class W_Number(W_Value):
    """W_Value subclass for javascript numeric values.

    This class provides no additional methods, and exists mainly to allow
    python code to typecheck for numbers.
    """

    def descr__repr__(self, space):
        value = support.emjs_read_double(self.handle)
        return space.wrap("<js.Number %f>" % (value,))


def W_Number_descr__new__(space, w_subtype, w_value):
    w_self = space.allocate_instance(W_Number, w_subtype)
    value = space.float_w(w_value)
    h_value = support.emjs_make_double(value)
    W_Number.__init__(space.interp_w(W_Number, w_self), h_value)
    return w_self


W_Number.typedef = TypeDef(
    "Number",
    (W_Value.typedef,),
    __doc__ = "Handle to a JS numeric value.",
    __new__ = interp2app(W_Number_descr__new__),
    __repr__ = interp2app(W_Number.descr__repr__),
)


class W_String(W_Value):
    """W_Value subclass for javascript string values.

    This class provides an additional method __len__ for easy access to
    the length of the string.  It's also useful for typechecking.
    """

    def descr__repr__(self, space):
        bufsize = support.emjs_length(self.handle)
        _check_error(space, bufsize)
        truncated = False
        if bufsize > 17:
            bufsize = 17
            truncated = True
        with rffi.scoped_alloc_buffer(bufsize) as buf:
            n = support.emjs_read_strn(self.handle, buf.raw, buf.size)
            strval = buf.str(n)
        if truncated:
            strval = strval + "..."
        return space.wrap("<js.String '%s'>" % (strval,))

    def descr__len__(self, space):
        res = support.emjs_length(self.handle)
        _check_error(space, res)
        return space.wrap(res)

    # XXX TODO: fun things like slicing?


def W_String_descr__new__(space, w_subtype, w_value):
    w_self = space.allocate_instance(W_String, w_subtype)
    value = space.str_w(w_value)
    h_value = support.emjs_make_strn(value, len(value))
    W_String.__init__(space.interp_w(W_String, w_self), h_value)
    return w_self


W_String.typedef = TypeDef(
    "String",
    (W_Value.typedef,),
    __doc__ = "Handle to a JS string value.",
    __new__ = interp2app(W_String_descr__new__),
    __repr__ = interp2app(W_String.descr__repr__),
    __len__ = interp2app(W_String.descr__len__),
)


class W_Object(W_Value):
    """W_Value subclass for javascript object values.

    This class provides no additional methods, and exists mainly to allow
    python code to typecheck for numbers.
    """

    def descr__repr__(self, space):
        return space.wrap("<js.Object handle=%d>" % (self.handle,))


def W_Object_descr__new__(space, w_subtype):
    w_self = space.allocate_instance(W_Object, w_subtype)
    # XXX TODO: kwargs for initial properties?
    h_value = support.emjs_make_object()
    W_Object.__init__(space.interp_w(W_Object, w_self), h_value)
    return w_self


W_Object.typedef = TypeDef(
    "Object",
    (W_Value.typedef,),
    __doc__ = "Handle to a JS object value.",
    __new__ = interp2app(W_Object_descr__new__),
    __repr__ = interp2app(W_Object.descr__repr__),
)


class W_Array(W_Object):
    """W_Object subclass for javascript array values.

    This class provides an additional method __len__ for easy access to
    the length of the array.  It's also useful for typechecking.
    """

    def descr__repr__(self, space):
        return space.wrap("<js.Array handle=%d>" % (self.handle,))

    def descr__len__(self, space):
        res = support.emjs_length(self.handle)
        _check_error(space, res)
        return space.wrap(res)


def W_Array_descr__new__(space, w_subtype, w_items, w_size):
    # XXX TODO: default arguments, somehow...
    size = space.int_w(w_size)
    h_self = support.emjs_make_array(size)
    _check_error(space, h_self)
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
                    res = support.emjs_prop_set_int(h_self, idx, h_item)
                    _check_error(space, res)
                idx += 1
    # XXX TODO: we should free the array if the above raises an error.
    w_self = space.allocate_instance(W_Array, w_subtype)
    W_Array.__init__(space.interp_w(W_Array, w_self), h_self)
    return w_self


W_Array.typedef = TypeDef(
    "Array",
    (W_Object.typedef,),
    __doc__ = "Handle to a JS array value.",
    __new__ = interp2app(W_Array_descr__new__),
    __repr__ = interp2app(W_Array.descr__repr__),
    __len__ = interp2app(W_Array.descr__len__),
)


class W_Function(W_Object):
    """W_Object subclass for javascript function values.

    This class provides an additional method __call__ for invoking the
    function.

    It can also be instantiated from python code to convert a python function
    into a javascript callback.  For callbacks that need to access the invoking
    'this' context, see the Method subclass.

    Function objects are always bound to a particular value of 'this',
    which is undefined by default but is set implicity when they are
    retreived as attributes on another Value instance.  To bind to a
    specific value for 'this', use the js.bind() function.
    """

    _immutable_fields_ = ['handle', 'context', 'callback']

    def __init__(self, handle, w_context=None, callback=None):
        self.w_context = w_context
        self.callback = callback
        W_Object.__init__(self, handle)

    def descr__repr__(self, space):
        return space.wrap("<js.Function handle=%d>" % (self.handle,))

    def descr__call__(self, space, args_w):
        h_args = support.emjs_make_array(len(args_w))
        _check_error(space, h_args)
        for i in xrange(len(args_w)):
            with _unwrap_handle(space, args_w[i]) as h_arg:
                support.emjs_prop_set_int(h_args, i, h_arg)
        w_ctx = self.w_context
        if w_ctx is None:
            w_ctx = undefined
        with _unwrap_handle(space, w_ctx) as h_ctx:
            h_res = support.emjs_apply(self.handle, h_ctx, h_args)
        return _wrap_handle(space, h_res)


def W_Function_descr__new__(space, w_subtype, w_callback, __args__):
    h_self, cb = _make_callback(space, w_callback, False, __args__)
    # XXX TODO: we should free the callback if the below raises an error.
    w_self = space.allocate_instance(W_Function, w_subtype)
    W_Function.__init__(space.interp_w(W_Function, w_self), h_self, None, cb)
    return w_self


W_Function.typedef = TypeDef(
    "Function",
    (W_Object.typedef,),
    __doc__ = "Handle to a JS function value.",
    __new__ = interp2app(W_Function_descr__new__),
    __repr__ = interp2app(W_Function.descr__repr__),
    __call__ = interp2app(W_Function.descr__call__),
)


class W_Method(W_Function):
    """W_Function subclass for python method callbacks.

    When instantiated from python code, this subclass will arrange for the
    python callback to be invoked with 'this' as its first argument, mirroring
    the way the instance is passed to standard python methods.
    """
    pass


def W_Method_descr__new__(space, w_subtype, w_callback, __args__):
    h_self, cb = _make_callback(space, w_callback, True, __args__)
    # XXX TODO: we should free the callback if the below raises an error.
    w_self = space.allocate_instance(W_Method, w_subtype)
    W_Method.__init__(space.interp_w(W_Method, w_self), h_self, None, cb)
    return w_self


W_Method.typedef = TypeDef(
    "Method",
    (W_Function.typedef,),
    __doc__ = "Handle to a JS function that invokes a python method.",
    __new__ = interp2app(W_Method_descr__new__),
)


class State:
    """State-holding class for additional app-level definitions.

    This class holds the mutable global state for the module, some of which
    can only be properly initialized at runtime.
    """

    def __init__(self, space):
        # Create a new Exception class for js-level errors.
        self.w_jserror = space.new_exception_class("js.Error")
        # Stubs for handles to some commonly-used js values.
        self.h_globals = 0
        self.h_array = 0
        self.h_pyerror = 0
        # A permanent object for accessing global scope.
        # This gets assigned the proper handle at module initialization.
        self.w_globals = space.wrap(W_Object(support.EMJS_ERROR))

    def startup(self, space):
        # Hold a permanent handle to the globals object.
        self.h_globals = support.emjs_globals()
        _check_error(space, self.h_globals)
        # And a permanent object while we're at it.
        self.w_globals.handle = self.h_globals
        # Hold a permanent handle to the Array type constructor.
        self.h_array = support.emjs_prop_get_str(self.h_globals, "Array")
        _check_error(space, self.h_array)
        # Create a custom Error type for throwing errors from python callbacks.
        self.h_pyerror = support.emjs_eval("""
          (function() {
            function PyError(message) {
              this.name = PyError;
              this.message = message || 'Python Error';
            }
            PyError.prototype = new Error();
            PyError.prototype.constructor = PyError;
            return PyError;
          })()
        """)
        _check_error(space, self.h_pyerror)


def getstate(space):
    """Get the (possibly cached) module state object."""
    return space.fromcache(State)
    

def _raise_error(space):
    """Helper function to raise an error if the js error flag is set."""
    h_err = support.emjs_get_error()
    if h_err != support.EMJS_UNDEFINED:
        support.emjs_clear_error()
        # if emjs_op_instanceof(err, h_pyerror):
        #    try:
        #        unpickle the error and raise it
        #    except:
        #        raise a RuntimeError with debugging info
        # this is how we pickle:
        #w_builtins = space.getbuiltinmodule('__builtin__')
        #w_picklemodule = space.call_method(
        #    w_builtins, '__import__', space.wrap("pickle"))
        #w_unpickled = space.call_method(
        #    w_picklemodule, "loads", w_received)
        w_jserror = getstate(space).w_jserror
        raise OperationError(w_jserror, _wrap_handle(space, h_err))


def _check_error(space, result):
    if result == support.EMJS_ERROR:
        _raise_error(space)


def _wrap_handle(space, handle):
    """Helper function to wrap a handle into an appropriate app-level object.

    This function does some basic type dispatching to return a wrapped instance
    of the appropriate W_Value subclass for the given handle.
    """
    if handle == support.EMJS_ERROR:
        _raise_error(space)
    if handle == support.EMJS_UNDEFINED:
        return space.wrap(undefined)
    if handle == support.EMJS_NULL:
        return space.wrap(null)
    if handle == support.EMJS_FALSE:
        return space.wrap(false)
    if handle == support.EMJS_TRUE:
        return space.wrap(true)
    typ = support.emjs_typeof(handle)
    if typ == support.EMJS_TYPE_ERROR:
        _raise_error(space)
    if typ == support.EMJS_TYPE_NUMBER:
        return space.wrap(W_Number(handle))
    if typ == support.EMJS_TYPE_STRING:
        return space.wrap(W_String(handle))
    if typ == support.EMJS_TYPE_FUNCTION:
        return space.wrap(W_Function(handle))
    assert typ == support.EMJS_TYPE_OBJECT
    h_array = getstate(space).h_array
    if support.emjs_op_instanceof(handle, h_array):
        return space.wrap(W_Array(handle))
    return space.wrap(W_Object(handle))


class _unwrap_handle(object):
    """Context-manager for unwrapping an app-level object to a js-level handle.

    This class can manage both existing W_Value instances, and do transient
    conversion of immutable app-level datatypes such as ints and strings.  It
    needs to be a context-manager to allow proper lifetime management of
    such transient handles, keeping them alive until the end of the block.
    """

    def __init__(self, space, w_value):
        self.space = space
        self.w_value = w_value
        self.w_transient = None

    def __enter__(self):
        space = self.space
        w_value = self.w_value
        try:
            # Optimistically assume that it's a proper W_Value instance.
            return space.interp_w(W_Value, w_value).handle
        except OperationError, e:
            # If not, try to convert it to a transient W_Value instance.
            if not e.match(space, space.w_TypeError):
                raise
            self.w_transient = _convert(space, w_value)
            return self.w_transient.handle

    def __exit__(self, exc_typ, exc_val, exc_tb):
        # A transient handle will be cleaned up by normal garbage collection.
        # We just needed to keep it alive until the end of the block.
        pass


def _convert(space, w_value):
    """Convert a wrapped value into a wrapped W_Value."""
    if space.is_w(w_value, space.w_None):
        return undefined
    if space.isinstance_w(w_value, space.w_int):
        return W_Number(support.emjs_make_int32(space.int_w(w_value)))
    if space.isinstance_w(w_value, space.w_long):
        return W_Number(support.emjs_make_int32(space.int_w(w_value)))
    if space.isinstance_w(w_value, space.w_float):
        return W_Number(support.emjs_make_double(space.float_w(w_value)))
    if space.isinstance_w(w_value, space.w_str):
        value = space.str_w(w_value)
        return W_String(support.emjs_make_strn(value, len(value)))
    # XXX TODO: is this typecheck safe and accurate?
    if isinstance(w_value, pypy.interpreter.function.Function):
        args = pypy.interpreter.function.Arguments(space, [])
        h_cb, cb = _make_callback(space, w_value, False, args)
        return W_Function(h_cb, None, cb)
    if isinstance(w_value, pypy.interpreter.function.Method):
        args = pypy.interpreter.function.Arguments(space, [])
        h_cb, cb = _make_callback(space, w_value, False, args)
        return W_Function(h_cb, None, cb)
    errmsg = "could not convert py type to js type"
    raise OperationError(space.w_TypeError, space.wrap(errmsg))


def convert(space, w_value):
    """Convert a python value into a matching Value instance.

    This function can convert immutable primitive python objects (e.g. ints
    and strings) into matching javascript Value objects.  It raises TypeError
    for objects that cannot be converted.
    """
    try:
        # XXX TODO: how to check this without using try-except?
        return space.wrap(space.interp_w(W_Value, w_value))
    except OperationError, e:
        if not e.match(space, space.w_TypeError):
            raise
        return space.wrap(_convert(space, w_value))


def globals(space):
    """Get a reference to the global scope 'this' object."""
    h = support.emjs_globals()
    return _wrap_handle(space, h)


@unwrap_spec(data=str)
def eval(space, data):
    """Evaluate javascript code at global scope."""
    h = support.emjs_eval(data)
    return _wrap_handle(space, h)


def bind(space, w_fn, w_ctx):
    """Create a copy of a function, bound to the given context."""
    func = space.interp_w(W_Function, w_fn)
    h_bound = support.emjs_dup(func.handle)
    _check_error(space, h_bound)
    bound = W_Function(h_bound, w_ctx, func.callback)
    return space.wrap(bound)


def new(space, w_fn, args_w):
    with _unwrap_handle(space, w_fn) as h_fn:
        h_args = support.emjs_make_array(len(args_w))
        _check_error(space, h_args)
        for i in xrange(len(args_w)):
            with _unwrap_handle(space, args_w[i]) as h_arg:
                res = support.emjs_prop_set_int(h_args, i, h_arg)
            _check_error(space, res)
        h_res = support.emjs_new(h_fn, h_args)
    return _wrap_handle(space, h_res)


def equal(space, w_lhs, w_rhs):
    """Invoking the javascript "==" operator betwee two values.

    The default equality operator for js.Value objects is mapped to
    javascript's strict "===" operator, for what are hopefully obvious
    reasons. This helper function exists for those rare cases when you
    actually *want* the wacky behaviour of the non-strict variant.
    """
    with _unwrap_handle(space, w_lhs) as h_lhs:
        with  _unwrap_handle(space, w_rhs) as h_rhs:
            res = support.emjs_op_eq(h_lhs, h_rhs)
    _check_error(space, res)
    return space.newbool(bool(res))


def instanceof(space, w_lhs, w_rhs):
    with _unwrap_handle(space, w_lhs) as h_lhs:
        with  _unwrap_handle(space, w_rhs) as h_rhs:
            res = support.emjs_op_instanceof(h_lhs, h_rhs)
    _check_error(space, res)
    return space.newbool(bool(res))


def urshift(space, w_lhs, w_rhs):
    with _unwrap_handle(space, w_lhs) as h_lhs:
        with  _unwrap_handle(space, w_rhs) as h_rhs:
            h_res = support.emjs_op_bw_urshift(h_lhs, h_rhs)
    return _wrap_handle(space, h_res)


def uint32(space, w_value):
    with _unwrap_handle(space, w_value) as h_value:
        res = support.emjs_read_uint32(h_value)
    return space.wrap(res)


def _make_callback(space, w_callback, wants_this, args):
    callback = Callback(space, w_callback, wants_this, args)
    dataptr = rffi.cast(rffi.VOIDP, callback.id)
    ll_dispatch_callback = rffi.llhelper(support.CALLBACK_TP, dispatch_callback)
    return support.emjs_make_callback(ll_dispatch_callback, dataptr), callback


class Callback(object):

    def __init__(self, space, w_callback, wants_this, __args__):
        self.id = global_callback_map.add(self)
        self.space = space
        self.w_callback = w_callback
        self.wants_this = wants_this
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
    # If the callback wants the value of 'this', pass it as first argument.
    args_w, kw_w = callback.__args__.unpack()
    if kw_w:
        raise RuntimeError("callback function kw args not implemented yet")
    h_args_len = support.emjs_length(h_args)
    all_args_len = len(args_w) + h_args_len
    if callback.wants_this:
        all_args_len += 1
    all_args_w = [None] * all_args_len
    i = 0
    if callback.wants_this:
        h_this = support.emjs_prop_get_str(h_args, "this")
        w_this = _wrap_handle(space, h_this)
        all_args_w[0] = w_this
        i += 1
    for w_arg in args_w:
        all_args_w[i] = w_arg
        i += 1
    for j in xrange(h_args_len):
        w_arg = _wrap_handle(space, support.emjs_prop_get_int(h_args, j))
        all_args_w[i] = w_arg
        i += 1
    # Do the call, propagating return value or error as appropriate.
    try:
        w_res = space.call(callback.w_callback, space.newtuple(all_args_w))
    except OperationError, pyerr:
        # XXX TODO: tunnel the exception through JS and back to Python?
        # XXX TODO: allow the callback to raise js.Error and reflect it
        # properly through to the javascript side.
        # w_jserror = getstate(space).w_jserror
        # if isinstance(pyerr, w_jserror):
        #    with _unwrap_handle(pyerror.args[0]) as h_err:
        #        support.emjs_set_error(h_err)
        # else:
        #    h_msg = make_error_string()
        #    h_data = make_error_pickle()
        #    h_err = new(h_pyerror, [h_msg, h_data])
        #    support.emjs_set_error(h_err)
        #    support.emjs_free(h_err)
        errmsg = pyerr.errorstr(space)
        h_errmsg = support.emjs_make_strn(errmsg, len(errmsg))
        support.emjs_set_error(h_errmsg)
        support.emjs_free(h_errmsg)
        return support.EMJS_ERROR
    else:
        # Note that the js-side callback stub frees the result handle,
        # so we have to dup it here to avoid breaking the w_res object.
        # XXX TODO: avoid the dup for transiently-unwrapped values.
        with _unwrap_handle(space, w_res) as h_res:
            return support.emjs_dup(h_res)


# Some useful singleton instances.

undefined = W_Undefined(support.EMJS_UNDEFINED)
null = W_Object(support.EMJS_NULL)
false = W_Boolean(support.EMJS_FALSE)
true = W_Boolean(support.EMJS_TRUE)
