
from rpython.rlib.unroll import unrolling_iterable
from rpython.rtyper.lltypesystem import lltype, llmemory
from rpython.rtyper.llinterp import LLInterpreter
from rpython.rtyper.annlowlevel import llhelper, cast_instance_to_gcref
from rpython.jit.backend.llsupport.llmodel import AbstractLLCPU
from rpython.jit.backend.llsupport import jitframe
from rpython.jit.metainterp import history

from rpython.jit.backend.asmjs import support
from rpython.jit.backend.asmjs.assembler import AssemblerASMJS


def execute_trampoline(func_id, ll_frame):
    frame = lltype.cast_opaque_ptr(jitframe.JITFRAMEPTR, ll_frame)
    ll_frame_ptr = CPU_ASMJS.cast_ptr_to_int(ll_frame)
    next_call = support.jitInvoke(func_id, ll_frame_ptr, 0)
    frame = frame.resolve()
    ll_frame = lltype.cast_opaque_ptr(llmemory.GCREF, frame)
    while next_call != 0:
        # High 24 bits give the function id.
        # Low 8 bits give the target label within that function.
        func_id = next_call >> 8
        func_goto = next_call & 0xFF
        ll_frame_ptr = CPU_ASMJS.cast_ptr_to_int(ll_frame)
        next_call = support.jitInvoke(func_id, ll_frame_ptr, func_goto)
        frame = frame.resolve()
        ll_frame = lltype.cast_opaque_ptr(llmemory.GCREF, frame)
    return ll_frame

ARGS = [lltype.Signed, llmemory.GCREF]
EXEFUNCPTR = lltype.Ptr(lltype.FuncType(ARGS, llmemory.GCREF))


class CPU_ASMJS(AbstractLLCPU):
    """'CPU' for emitting and running asmjs javascript.

    This inherits from AbstractLLCPU because it interfaces with the rest of
    the interpreter in the same way, via low-level descrs.  But the details
    of compiling and running a trace are very very different.  Rather than
    mapping executable memory and jumping into it, we build up javascript
    source code and call external functions to compile and invoke it.
    """

    IS_64_BIT = False
    from rpython.jit.backend.asmjs.arch import JITFRAME_FIXED_SIZE
    supports_floats = True
    supports_singlefloats = False
    supports_longlong = False
    with_threads = False

    def set_debug(self, flag):
        return self.assembler.set_debug(flag)

    def get_failargs_limit(self):
        if self.opts is not None:
            return self.opts.failargs_limit
        else:
            return 1000

    def setup(self):
        self.assembler = AssemblerASMJS(self)

    def setup_once(self):
        self.assembler.setup_once()

    def finish_once(self):
        self.assembler.finish_once()

    def make_execute_token(self, *ARGS):
        """Build and return a function for executing the given JIT token.

        Each chunk of compiled code is represented by an integer "function id".
        We need to look up the id, build the necessary frame, and then call the
        helper function "jitInvoke" to execute the compiled function.

        The compiled code will return either 0, indicating that it's done and
        we can find the results in the frame, or a positive integer indicating
        another compiled function that should be invoked.  We loop over this,
        executing each function in turn until they're done.  It's a simple
        simulation of GOTOs between the compiled chunks of code.

        This little trampoline is necessary because the main interpreter is
        running in asmjs mode, where it is forbidden to define new functions
        at runtime.  It has to call into helper code hosted outside of the
        asmjs fast-path.
        """
        #  This is mostly copied from llsupport/llmodel.py, but with changes
        #  to invoke the external javascript helper trampoline thingy.
        lst = [(i, history.getkind(ARG)[0]) for i, ARG in enumerate(ARGS)]
        kinds = unrolling_iterable(lst)

        def execute_token(executable_token, *args):
            clt = executable_token.compiled_loop_token
            func_id = clt._compiled_function_id
            frame_info = clt.frame_info
            frame = self.gc_ll_descr.malloc_jitframe(frame_info)
            ll_frame = lltype.cast_opaque_ptr(llmemory.GCREF, frame)
            locs = clt._ll_initial_locs
            if not self.translate_support_code:
                prev_interpreter = LLInterpreter.current_interpreter
                LLInterpreter.current_interpreter = self.debug_ll_interpreter
            try:
                for i, kind in kinds:
                    arg = args[i]
                    num = locs[i]
                    if kind == history.INT:
                        self.set_int_value(ll_frame, num, arg)
                    elif kind == history.FLOAT:
                        self.set_float_value(ll_frame, num, arg)
                    else:
                        assert kind == history.REF
                        self.set_ref_value(ll_frame, num, arg)
                ll_frame = execute_trampoline(func_id, ll_frame)
            finally:
                if not self.translate_support_code:
                    LLInterpreter.current_interpreter = prev_interpreter
            return ll_frame

        return execute_token

    def get_execute_trampoline_adr(self):
        exeptr = llhelper(EXEFUNCPTR, execute_trampoline)
        return self.cast_ptr_to_int(exeptr)

    def compile_loop(self, inputargs, operations, looptoken,
                     log=True, name=''):
        return self.assembler.assemble_loop(name, inputargs, operations,
                                            looptoken, log=log)

    def compile_bridge(self, faildescr, inputargs, operations,
                       original_loop_token, log=True):
        clt = original_loop_token.compiled_loop_token
        clt.compiling_a_bridge()
        return self.assembler.assemble_bridge(faildescr, inputargs, operations,
                                              original_loop_token, log=log)

    def free_loop_and_bridges(self, compiled_loop_token):
        AbstractLLCPU.free_loop_and_bridges(self, compiled_loop_token)
        self.assembler.free_loop_and_bridges(compiled_loop_token)

    def cast_ptr_to_int(x):
        adr = llmemory.cast_ptr_to_adr(x)
        return CPU_ASMJS.cast_adr_to_int(adr)
    cast_ptr_to_int._annspecialcase_ = 'specialize:arglltype(0)'
    cast_ptr_to_int = staticmethod(cast_ptr_to_int)

    def redirect_call_assembler(self, oldlooptoken, newlooptoken):
        old_func_id = oldlooptoken.compiled_loop_token._compiled_function_id
        new_func_id = newlooptoken.compiled_loop_token._compiled_function_id
        support.jitReplace(old_func_id, new_func_id)

    def invalidate_loop(self, looptoken):
        self.assembler.invalidate_loop(looptoken)

    def _decode_pos(self, deadframe, index):
        descr = self.get_latest_descr(deadframe)
        if descr.final_descr:
            assert index == 0
            return 0
        return descr._asmjs_faillocs[index]
