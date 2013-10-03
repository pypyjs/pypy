
from rpython.rtyper.lltypesystem import lltype, llmemory, rffi
from rpython.rtyper.extfunc import register_external

from rpython.jit.backend.asmjs import pyasmjs


def jsexternal(args, result):
    """Decorator to define stubbed-out external javascript functions.

    This decorator can be applied to a python function to register it as
    the stubbed-out implementation of an external javascript function.
    The llinterpreter will run the python code, the compiled interpreter
    will link to the javascript function of the same name.
    """
    def do_register(func):
        register_external(func, args, result, llfakeimpl=func,
                          llimpl=rffi.llexternal(func.__name__, args, result))
        return func
    return do_register


_jitCompiledFunctions = {0: None}
_jitTriggeredGuards = {}


@jsexternal([rffi.CCHARP], rffi.INT)
def jitCompile(jssource):
    funcid = jitReserve()
    jitRecompile(funcid, jssource)
    return funcid


@jsexternal([], rffi.INT)
def jitReserve():
    funcid = len(_jitCompiledFunctions)
    _jitCompiledFunctions[funcid] = None
    return funcid


@jsexternal([rffi.INT, rffi.CCHARP], lltype.Void)
def jitRecompile(funcid, jssource):
    _jitCompiledFunctions[funcid] = pyasmjs.load_asmjs(jssource)


@jsexternal([rffi.INT, rffi.INT], lltype.Void)
def jitReplace(oldfuncid, newfuncid):
    _jitCompiledFunctions[oldid] = _jitCompiledFunctions[newid]


@jsexternal([rffi.INT, llmemory.GCREF, rffi.INT], rffi.INT)
def jitInvoke(funcid, llframe, target):
    func = _jitCompiledFunctions.get(funcid, None)
    if func is None:
        return 0
    return func(llframe, target)


@jsexternal([rffi.INT], lltype.Void)
def jitFree(funcid):
    _jitCompiledFunctions[funcid] = None
    while len(_jitCompiledFunctions) == funcid + 1:
        del _jitCompiledFunctions[funcid]
        funcid -= 1


@jsexternal([rffi.INT], lltype.Void)
def jitTriggerGuard(guardid):
    _jitTriggeredGuards[guardid] = True


@jsexternal([rffi.INT], rffi.INT)
def jitGuardWasTriggered(guardid):
    return _jitTriggeredGuards.get(guardid, False)
