
import os
import sys

from rpython.rlib import rgc
from rpython.rlib.rarithmetic import r_uint
from rpython.jit.backend.llsupport import symbolic, jitframe, rewrite
from rpython.jit.backend.llsupport.regalloc import compute_vars_longevity
from rpython.jit.backend.llsupport.descr import (unpack_fielddescr,
                                                 unpack_arraydescr,
                                                 unpack_interiorfielddescr,
                                                 ArrayDescr, CallDescr,
                                                 FieldDescr)
from rpython.jit.codewriter import longlong
from rpython.rtyper.lltypesystem import lltype, rffi, rstr
from rpython.rtyper.annlowlevel import cast_instance_to_gcref, llhelper
from rpython.jit.backend.model import CompiledLoopToken
from rpython.jit.metainterp.resoperation import rop
from rpython.jit.metainterp.history import (AbstractFailDescr, ConstInt,
                                            Box, TargetToken, INT,
                                            REF, FLOAT, BoxInt,
                                            JitCellToken)

from rpython.jit.backend.asmjs import support
from rpython.jit.backend.asmjs import jsvalue as js
from rpython.jit.backend.asmjs.arch import WORD, SANITYCHECK
from rpython.jit.backend.asmjs.jsbuilder import ASMJSBuilder


GILFUNCPTR = lltype.Ptr(lltype.FuncType([], lltype.Void))


@rgc.no_collect
def release_gil_shadowstack():
    before = rffi.aroundstate.before
    if before:
        before()


@rgc.no_collect
def reacquire_gil_shadowstack():
    after = rffi.aroundstate.after
    if after:
        after()


INVALIDATION_COUNTER = lltype.Struct(
    "INVALIDATIONCOUNTER",
    ("counter", lltype.Signed)
)


class CompiledLoopTokenASMJS(CompiledLoopToken):
    """CompiledLoopToken with extra fields for asmjs backend."""

    def __init__(self, cpu, number):
        CompiledLoopToken.__init__(self, cpu, number)
        self.all_gcrefs = []
        self.compiled_function_id = 0
        self.compiled_guard_funcs = []
        self.compiled_gcmaps = []
        self.invalidation = lltype.malloc(INVALIDATION_COUNTER, flavor="raw")
        self.invalidation.counter = 0

    def __del__(self):
        CompiledLoopToken.__del__(self)
        lltype.free(self.invalidation, flavor="raw")


class AssemblerASMJS(object):
    """Class for assembling a Trace into a compiled ASMJS function."""

    def __init__(self, cpu):
        self.cpu = cpu

    def set_debug(self, v):
        return False

    def setup_once(self):
        self.execute_trampoline_addr = self.cpu.get_execute_trampoline_adr()
        release_gil_func = llhelper(GILFUNCPTR,
                                    release_gil_shadowstack)
        self.release_gil_addr = self.cpu.cast_ptr_to_int(release_gil_func)
        reacquire_gil_func = llhelper(GILFUNCPTR,
                                      reacquire_gil_shadowstack)
        self.reacquire_gil_addr = self.cpu.cast_ptr_to_int(reacquire_gil_func)
        gc_ll_descr = self.cpu.gc_ll_descr
        gc_ll_descr.initialize()
        if gc_ll_descr.get_malloc_slowpath_addr is not None:
            nm = "malloc_nursery"
            self.gc_malloc_nursery_addr = gc_ll_descr.get_malloc_fn_addr(nm)
            nm = "malloc_array"
            self.gc_malloc_array_addr = gc_ll_descr.get_malloc_fn_addr(nm)
        if hasattr(gc_ll_descr, "malloc_str"):
            nm = "malloc_str"
            self.gc_malloc_str_addr = gc_ll_descr.get_malloc_fn_addr(nm)
        else:
            self.gc_malloc_str_addr = None
        if hasattr(gc_ll_descr, "malloc_unicode"):
            nm = "malloc_unicode"
            self.gc_malloc_unicode_addr = gc_ll_descr.get_malloc_fn_addr(nm)
        else:
            self.gc_malloc_unicode_addr = None

    def finish_once(self):
        pass

    def setup(self, looptoken):
        self.current_clt = looptoken.compiled_loop_token
        self.bldr = ASMJSBuilder(self.cpu)
        self.pending_target_tokens = []
        self.has_unimplemented_ops = False
        self.longevity = {}
        self.box_to_jsval = {}
        self.required_frame_depth = 0
        self.spilled_frame_locations = {}
        self.spilled_frame_values = {}
        self.spilled_frame_offset = 0

    def teardown(self):
        if self.has_unimplemented_ops:
            raise RuntimeError("TRACE CONTAINS UNIMPLEMENTED OPERATIONS")
        self.current_clt = None
        self.bldr = None
        self.pending_target_tokens = None
        self.has_unimplemented_ops = False
        self.longevity = None
        self.box_to_jsval = None
        self.required_frame_depth = 0
        self.spilled_frame_locations = None
        self.spilled_frame_values = None
        self.spilled_frame_offset = 0

    def assemble_loop(self, loopname, inputargs, operations, looptoken, log):
        """Assemble and compile a new loop function from the given trace.

        This method takes the recorded trace, generates asmjs source code from
        it, then invokes an external helper to compile that source into a new
        function object.  It attached a "function id" to the CompiledLoopToken
        which can be used to invoke the new function.
        """
        clt = CompiledLoopTokenASMJS(self.cpu, looptoken.number)
        looptoken.compiled_loop_token = clt
        self.setup(looptoken)
        operations = self._prepare_loop(inputargs, operations, looptoken)
        self._assemble(inputargs, operations)
        clt.frame_info = self._allocate_frame_info(self.required_frame_depth)
        clt.compiled_function_id = self._compile(self.bldr)
        self._finalize_pending_target_tokens(clt.compiled_function_id)
        self.teardown()

    def assemble_bridge(self, faildescr, inputargs, operations,
                        original_loop_token, log):
        """Assemble, compile and link a new bridge from the given trace.

        This method takes the recorded trace, generates asmjs source code
        from it, then invokes and external helper to compile it over the top
        of the existing function id for the bridged guard.
        """
        assert isinstance(faildescr, AbstractFailDescr)
        self.setup(original_loop_token)
        operations = self._prepare_bridge(inputargs, operations,
                                          original_loop_token)
        self._assemble(inputargs, operations)
        self._update_frame_info(self.current_clt.frame_info,
                                self.required_frame_depth)
        self._recompile(faildescr._asmjs_funcid, self.bldr)
        self._finalize_pending_target_tokens(faildescr._asmjs_funcid)
        self.teardown()

    def free_loop_and_bridges(self, compiled_loop_token):
        for gcmap in compiled_loop_token.compiled_gcmaps:
            lltype.free(gcmap, flavor="raw")
        for funcid in compiled_loop_token.compiled_guard_funcs:
            support.jitFree(funcid)
        self._free_frame_info(compiled_loop_token.frame_info)
        support.jitFree(compiled_loop_token.compiled_function_id)

    def invalidate_loop(self, looptoken):
        looptoken.compiled_loop_token.invalidation.counter += 1

    def _allocate_frame_info(self, required_depth):
        baseofs = self.cpu.get_baseofs_of_frame_field()
        frame_info = lltype.malloc(jitframe.JITFRAMEINFO, flavor="raw")
        frame_info_ptr = rffi.cast(jitframe.JITFRAMEINFOPTR, frame_info)
        frame_info_ptr.clear()
        frame_info_ptr.update_frame_depth(baseofs, required_depth)
        return frame_info_ptr

    def _update_frame_info(self, frame_info_ptr, required_depth):
        baseofs = self.cpu.get_baseofs_of_frame_field()
        frame_info_ptr.update_frame_depth(baseofs, required_depth)
        return frame_info_ptr

    def _free_frame_info(self, frame_info_ptr):
        lltype.free(frame_info_ptr, flavor="raw")

    def _ensure_frame_depth(self, required_depth):
        if self.required_frame_depth < required_depth:
            self.required_frame_depth = required_depth

    def _prepare_loop(self, inputargs, operations, looptoken):
        operations = self._prepare(inputargs, operations, looptoken)
        locations = self._get_frame_locations(inputargs)
        # Store the list of locs on the token, so that the calling
        # code knows where to write them to and the GCRewriter can find refs.
        looptoken.compiled_loop_token._ll_initial_locs = locations
        return operations

    def _prepare_bridge(self, inputargs, operations, looptoken):
        operations = self._prepare(inputargs, operations, looptoken)
        return operations

    def _prepare(self, inputargs, operations, looptoken):
        cpu = self.cpu
        allgcrefs = looptoken.compiled_loop_token.all_gcrefs
        return cpu.gc_ll_descr.rewrite_assembler(cpu, operations, allgcrefs)

    def _find_loop_entry_token(self, operations):
        """Find the looptoken targetted by the final JUMP, if any.

        This method finds and marks the target of the backwards jump at the
        end of the loop.  It costs us an iteration over the ops list, but it
        allows the js to more efficiently generate its looping structure
        by knowing this information up-front.
        """
        local_targets = []
        for i in range(len(operations) - 1):
            op = operations[i]
            assert op.getopnum() != rop.JUMP
            if op.getopnum() == rop.LABEL:
                descr = op.getdescr()
                assert isinstance(descr, TargetToken)
                local_targets.append(descr)
        final_op = operations[-1]
        if final_op.getopnum() == rop.JUMP:
            descr = final_op.getdescr()
            assert isinstance(descr, TargetToken)
            if descr in local_targets:
                self.bldr.set_loop_entry_token(descr)

    def _assemble(self, inputargs, operations):
        """Generate the body of the asmjs function for the given trace."""
        self.longevity, _ = compute_vars_longevity(inputargs, operations)
        self._find_loop_entry_token(operations)
        # Load input arguments from frame if we are entering from the top.
        with self._genop_if_entered_at(0):
            self._genop_load_input_args(inputargs)
        # Check that the frame is big enough, and re-allocate it if not.
        # XXX TODO: this is silly to do at the entry to every function,
        # but greatly simplifies things for now.
        cur_depth = js.HeapData(js.Int32, js.JitFrameSizeAddr())
        req_depth = self.bldr.allocate_intvar()
        with self.bldr.emit_if_block(js.LessThan(cur_depth, req_depth)):
            newframe = js.DynCallFunc("iii",
                                      js.ConstInt(self.cpu.realloc_frame),
                                      [js.jitFrame, cur_depth])
            self.bldr.emit_assignment(js.jitFrame, newframe)
        # Walk the list of operations, emitting code for each.
        # We expend some modest effort to generate "nice" javascript code,
        # by e.g. folding constant expressions and eliminating temp variables.
        i = 0
        while i < len(operations):
            op = operations[i]
            # Can we omit the operation completely?
            if op.has_no_side_effect() and op.result not in self.longevity:
                self.bldr.emit_comment("OMMITTED USELESS JIT OP: %s" % (op,))
            # Do we need to emit it in conjunction with a guard?
            elif self._op_needs_guard(op):
                i += 1
                if SANITYCHECK:
                    assert i < len(operations)
                guardop = operations[i]
                if SANITYCHECK:
                    assert guardop.is_guard()
                self.bldr.emit_comment("BEGIN JIT OP: %s" % (op,))
                self.bldr.emit_comment("  WITH GUARD OP: %s" % (guardop,))
                genop_withguard_list[op.getopnum()](self, op, guardop)
                self.bldr.emit_comment("DONE JIT OP WITH GUARD")
            # Do we need to write complex code for it?
            elif not self._op_is_simple_expr(op):
                self.bldr.emit_comment("BEGIN JIT OP: %s" % (op,))
                genop_list[op.getopnum()](self, op)
                self.bldr.emit_comment("DONE JIT OP")
            # It's just a simple expression.
            # Maybe we can fold it into the next op?
            else:
                self.bldr.emit_comment("BEGIN JIT EXPR OP: %s" % (op,))
                expr = genop_expr_list[op.getopnum()](self, op)
                if self._is_final_use(op.result, i + 1):
                    if SANITYCHECK:
                        assert op.result not in self.box_to_jsval
                    self.box_to_jsval[op.result] = expr
                    self.bldr.emit_comment("FOLDED JIT EXPR OP")
                else:
                    boxvar = self._get_jsval(op.result)
                    self.bldr.emit_assignment(boxvar, expr)
                    self.bldr.emit_comment("DONE JIT EXPR OP")
            # Free vars for boxes that are no longer needed.
            for j in range(op.numargs()):
                self._maybe_free_boxvar(op.getarg(j), i)
            self._maybe_free_boxvar(op.result, i)
            i += 1
        # Back-patch the required frame depth into the above check.
        req_depth_value = self.required_frame_depth
        self.bldr.set_initial_value_intvar(req_depth, req_depth_value)

    def _get_jsval(self, jitval):
        if isinstance(jitval, Box):
            try:
                return self.box_to_jsval[jitval]
            except KeyError:
                if jitval.type == FLOAT:
                    jsval = self.bldr.allocate_doublevar()
                else:
                    jsval = self.bldr.allocate_intvar()
                self.box_to_jsval[jitval] = jsval
                return jsval
        return jitval

    def _is_final_use(self, box, i):
        if box is None or not isinstance(box, Box):
            return False
        assert isinstance(box, Box)
        if box not in self.longevity:
            return False
        return self.longevity[box][1] == i

    def _maybe_free_boxvar(self, box, i):
        if self._is_final_use(box, i):
            boxvar = self.box_to_jsval.pop(box, None)
            if boxvar is not None:
                if isinstance(boxvar, js.IntVar):
                    self.bldr.free_intvar(boxvar)
                elif isinstance(boxvar, js.DoubleVar):
                    self.bldr.free_doublevar(boxvar)

    @staticmethod
    def _op_needs_guard(op):
        """Check if the given op must be implemented together with a guard.

        Some operations only make sense when followed immediately by a specific
        guard op, and we can only emit correct and efficient code when we
        consider the operation and its guard as a single unit.
        """
        # Possibly-overflowing ops must have an overflow-checking guard.
        if op.is_ovf():
            return True
        # Calls that may force must be followed by a force-checking guard.
        opnum = op.getopnum()
        if opnum == rop.CALL_MAY_FORCE:
            return True
        elif opnum == rop.CALL_ASSEMBLER:
            return True
        elif opnum == rop.CALL_RELEASE_GIL:
            return True
        # All other operations can be implemented stand-alone.
        return False

    @staticmethod
    def _op_is_simple_expr(op):
        """Check if the given op can be implemented as a simple expression.

        This check identifies operations that can be implemented as a simple
        expression, and are therefore easy to fold into other ops if their
        result is only used a single time.
        """
        if not op.is_always_pure():
            return False
        if op.getopnum() == rop.INT_FORCE_GE_ZERO:
            return False
        if op.getopnum() == rop.FLOAT_ABS:
            return False
        if op.result.type == FLOAT:
            if op.getopnum() == rop.GETFIELD_GC_PURE:
                return False
            if op.getopnum() == rop.GETFIELD_RAW_PURE:
                return False
            if op.getopnum() == rop.GETARRAYITEM_GC_PURE:
                return False
            if op.getopnum() == rop.GETARRAYITEM_RAW_PURE:
                return False
        return True

    def _genop_realize_box(self, box):
        boxexpr = self._get_jsval(box)
        if isinstance(boxexpr, js.Variable):
            return boxexpr
        del self.box_to_jsval[box]
        boxvar = self._get_jsval(box)
        self.bldr.emit_assignment(boxvar, boxexpr)
        return boxvar

    def _genop_if_entered_at(self, target):
        test = js.Equal(js.IntVar("goto"), js.ConstInt(target))
        return self.bldr.emit_if_block(test)

    def _genop_load_input_args(self, inputargs):
        locations = self._get_frame_locations(inputargs)
        assert len(inputargs) == len(locations)
        for i in xrange(len(inputargs)):
            box = inputargs[i]
            if not box:
                continue
            offset = locations[i]
            addr = js.JitFrameSlotAddr(offset)
            typ = js.HeapType.from_box(box)
            self.bldr.emit_load(self._get_jsval(box), addr, typ)
        return locations

    def _genop_write_output_args(self, inputargs):
        locations = self._get_frame_locations(inputargs)
        assert len(inputargs) == len(locations)
        with ctx_spill_to_frame(self) as f:
            for i in xrange(len(inputargs)):
                box = inputargs[i]
                offset = locations[i]
                curval = self.spilled_frame_values.get(offset, None)
                if curval is not None:
                    if SANITYCHECK:
                        assert curval == box
                else:
                    f.genop_spill_to_frame(box, offset)
            self._genop_store_gcmap()
        return locations

    def _get_frame_locations(self, arguments):
        """Allocate locations in the frame for all the given arguments."""
        locations = [-1] * len(arguments)
        offset = 0
        for i in xrange(len(arguments)):
            box = arguments[i]
            typ = js.HeapType.from_box(box)
            alignment = offset % typ.size
            if alignment:
                offset += alignment
            locations[i] = offset
            offset += typ.size
        self._ensure_frame_depth(offset)
        return locations

    def _compile(self, js):
        """Compile the generated source into an invokable asmjs function.

        This method passes the generated source to an external javascript
        helper, which compiles it into a function and returns an opaque
        integer "function id" that can be used to invoke the code.
        """
        jssrc = js.finish()
        os.write(2, "-=-=-=-= COMPILING ASMJS =-=-=-=-\n")
        os.write(2, jssrc)
        os.write(2, "\n-=-=-=-=-=-=-=-=-\n")
        assert not self.has_unimplemented_ops
        funcid = support.jitCompile(jssrc)
        return funcid

    def _recompile(self, function_id, js):
        """Compile generated source as replacement for an existing function.

        This method passes the generated source to an external javascript
        helper, which compiles it into a function and uses that to replace
        any existing function associated with the given function id.
        """
        jssrc = js.finish()
        os.write(2, "-=-=-=-= RECOMPILING ASMJS =-=-=-=-\n")
        os.write(2, jssrc)
        os.write(2, "\n-=-=-=-=-=-=-=-=-\n")
        assert not self.has_unimplemented_ops
        support.jitRecompile(function_id, jssrc)

    def _finalize_pending_target_tokens(self, function_id):
        """Attach the given function id to any pending target tokens.

        The target tokens have to be annotated with their compiled function
        id before other code can generate jump to them.
        """
        for token in self.pending_target_tokens:
            token._asmjs_funcid = function_id
            # Let the argboxes be GC'd, we can't generate any
            # more local jumps after compilation.
            token._asmjs_argboxes = None

    #
    #  Code-Generating dispatch methods.
    #  There's a method here for every resop we support.
    #  They are built into dispatch tables by code at the end of this file.
    #

    def genop_label(self, op):
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        # Attach the argument boxes to the descr, so that
        # jumps to this label know where to put their arguments.
        inputargs = op.getarglist()
        argboxes = [None] * len(inputargs)
        for i in range(len(inputargs)):
            argbox = inputargs[i]
            assert isinstance(argbox, Box)
            argboxes[i] = argbox
            # Make this box immortal.
            self.longevity[argbox] = (self.longevity[argbox][0], 2**30)
        descr._asmjs_argboxes = argboxes
        descr._asmjs_label = self.bldr.emit_token_label(descr)
        descr._asmjs_funcid = 0  # placeholder; this gets set after compilation
        self.pending_target_tokens.append(descr)
        # Load input arguments from frame if we are entering at this label.
        with self._genop_if_entered_at(descr._asmjs_label):
            self._genop_load_input_args(inputargs)

    def genop_expr_strgetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        assert itemsize == 1
        itemoffset = js.Plus(js.ConstInt(basesize), offset)
        return js.HeapData(js.Int8, js.Plus(base, itemoffset))

    def genop_strsetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        assert itemsize == 1
        itemoffset = js.Plus(js.ConstInt(basesize), offset)
        self.bldr.emit_store(value, js.Plus(base, itemoffset), js.Int8)

    def genop_expr_strlen(self, op):
        base = self._get_jsval(op.getarg(0))
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        return js.HeapData(js.Int32, js.Plus(base, js.ConstInt(len_offset)))

    def genop_copystrcontent(self, op):
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        self._genop_copy_array(op, arraytoken)

    def genop_expr_unicodegetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        arraytoken = symbolic.get_array_token(rstr.UNICODE,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        typ = js.HeapType.from_size(itemsize)
        itemoffset = js.Plus(js.ConstInt(basesize),
                             js.IMul(offset, js.ConstInt(itemsize)))
        return js.HeapData(typ, js.Plus(base, itemoffset))

    def genop_unicodesetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        arraytoken = symbolic.get_array_token(rstr.UNICODE,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        typ = js.HeapType.from_size(itemsize)
        itemoffset = js.Plus(js.ConstInt(basesize),
                             js.IMul(offset, js.ConstInt(itemsize)))
        self.bldr.emit_store(value, js.Plus(base, itemoffset), typ)

    def genop_expr_unicodelen(self, op):
        base = self._get_jsval(op.getarg(0))
        arraytoken = symbolic.get_array_token(rstr.UNICODE,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        return js.HeapData(js.Int32, js.Plus(base, js.ConstInt(len_offset)))

    def genop_copyunicodecontent(self, op):
        arraytoken = symbolic.get_array_token(rstr.UNICODE,
                                              self.cpu.translate_support_code)
        self._genop_copy_array(op, arraytoken)

    def _genop_copy_array(self, op, arraytoken):
        srcbase = self._get_jsval(op.getarg(0))
        dstbase = self._get_jsval(op.getarg(1))
        srcoffset = self._get_jsval(op.getarg(2))
        dstoffset = self._get_jsval(op.getarg(3))
        lengthbox = self._get_jsval(op.getarg(4))
        assert srcbase != dstbase
        basesize = js.ConstInt(arraytoken[0])
        itemsize = js.ConstInt(arraytoken[1])
        # Calculate offset into source array.
        srcaddr = js.Plus(srcbase,
                          js.Plus(basesize, js.IMul(srcoffset, itemsize)))
        # Calculate offset into destination array.
        dstaddr = js.Plus(dstbase,
                          js.Plus(basesize, js.IMul(dstoffset, itemsize)))
        # Memcpy required number of bytes.
        nbytes = js.IMul(lengthbox, itemsize)
        self.bldr.emit_expr(js.CallFunc("memcpy", [dstaddr, srcaddr, nbytes]))

    def genop_getfield_gc(self, op):
        base = self._get_jsval(op.getarg(0))
        fielddescr = op.getdescr()
        assert isinstance(fielddescr, FieldDescr)
        offset, fieldsize, signed = unpack_fielddescr(op.getdescr())
        addr = js.Plus(base, js.ConstInt(offset))
        typ = js.HeapType.from_size_and_sign(fieldsize, signed)
        self.bldr.emit_load(self._get_jsval(op.result), addr, typ)

    genop_getfield_raw = genop_getfield_gc
    genop_getfield_gc_pure = genop_getfield_gc
    genop_getfield_raw_pure = genop_getfield_gc

    def genop_expr_getfield_gc_pure(self, op):
        base = self._get_jsval(op.getarg(0))
        fielddescr = op.getdescr()
        assert isinstance(fielddescr, FieldDescr)
        offset, fieldsize, signed = unpack_fielddescr(op.getdescr())
        addr = js.Plus(base, js.ConstInt(offset))
        typ = js.HeapType.from_size_and_sign(fieldsize, signed)
        return js.HeapData(typ, addr)

    genop_expr_getfield_raw_pure = genop_expr_getfield_gc_pure

    def genop_setfield_gc(self, op):
        base = self._get_jsval(op.getarg(0))
        value = self._get_jsval(op.getarg(1))
        offset, fieldsize, signed = unpack_fielddescr(op.getdescr())
        addr = js.Plus(base, js.ConstInt(offset))
        typ = js.HeapType.from_size_and_sign(fieldsize, signed)
        self.bldr.emit_store(value, addr, typ)

    genop_setfield_raw = genop_setfield_gc

    def genop_getinteriorfield_gc(self, op):
        t = unpack_interiorfielddescr(op.getdescr())
        offset, itemsize, fieldsize, signed = t
        base = self._get_jsval(op.getarg(0))
        which = self._get_jsval(op.getarg(1))
        addr = js.Plus(base, js.Plus(js.ConstInt(offset),
                                     js.IMul(which, js.ConstInt(itemsize))))
        typ = js.HeapType.from_size_and_sign(fieldsize, signed)
        self.bldr.emit_load(self._get_jsval(op.result), addr, typ)

    def genop_setinteriorfield_gc(self, op):
        t = unpack_interiorfielddescr(op.getdescr())
        offset, itemsize, fieldsize, signed = t
        base = self._get_jsval(op.getarg(0))
        which = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        addr = js.Plus(base, js.Plus(js.ConstInt(offset),
                                     js.IMul(which, js.ConstInt(itemsize))))
        typ = js.HeapType.from_size_and_sign(fieldsize, signed)
        self.bldr.emit_store(value, addr, typ)

    genop_setinteriorfield_raw = genop_setinteriorfield_gc

    def genop_expr_arraylen_gc(self, op):
        descr = op.getdescr()
        assert isinstance(descr, ArrayDescr)
        len_offset = descr.lendescr.offset
        base = self._get_jsval(op.getarg(0))
        addr = js.Plus(base, js.ConstInt(len_offset))
        return js.HeapData(js.Int32, addr)

    def genop_getarrayitem_gc(self, op):
        itemsize, offset, signed = unpack_arraydescr(op.getdescr())
        base = self._get_jsval(op.getarg(0))
        which = self._get_jsval(op.getarg(1))
        addr = js.Plus(base, js.Plus(js.ConstInt(offset),
                                     js.IMul(which, js.ConstInt(itemsize))))
        typ = js.HeapType.from_size_and_sign(itemsize, signed)
        self.bldr.emit_load(self._get_jsval(op.result), addr, typ)

    genop_getarrayitem_gc_pure = genop_getarrayitem_gc
    genop_getarrayitem_raw = genop_getarrayitem_gc
    genop_getarrayitem_raw_pure = genop_getarrayitem_gc
    genop_raw_load = genop_getarrayitem_gc

    def genop_expr_getarrayitem_gc_pure(self, op):
        itemsize, offset, signed = unpack_arraydescr(op.getdescr())
        base = self._get_jsval(op.getarg(0))
        which = self._get_jsval(op.getarg(1))
        addr = js.Plus(base, js.Plus(js.ConstInt(offset),
                                     js.IMul(which, js.ConstInt(itemsize))))
        typ = js.HeapType.from_size_and_sign(itemsize, signed)
        return js.HeapData(typ, addr)

    genop_expr_getarrayitem_raw_pure = genop_expr_getarrayitem_gc_pure

    def genop_setarrayitem_gc(self, op):
        itemsize, offset, signed = unpack_arraydescr(op.getdescr())
        base = self._get_jsval(op.getarg(0))
        where = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        itemoffset = js.Plus(js.ConstInt(offset),
                             js.IMul(where, js.ConstInt(itemsize)))
        addr = js.Plus(base, itemoffset)
        typ = js.HeapType.from_size_and_sign(itemsize, signed)
        self.bldr.emit_store(value, addr, typ)

    genop_setarrayitem_raw = genop_setarrayitem_gc
    genop_raw_store = genop_setarrayitem_gc

    def _genop_expr_int_unaryop(operator):
        def genop_expr_int_unaryop(self, op):
            return operator(self._get_jsval(op.getarg(0)))
        return genop_expr_int_unaryop

    genop_expr_int_is_true = _genop_expr_int_unaryop(js.SignedCast)
    genop_expr_int_is_zero = _genop_expr_int_unaryop(js.UNot)
    genop_expr_int_neg = _genop_expr_int_unaryop(js.UNeg)
    genop_expr_int_invert = _genop_expr_int_unaryop(js.UNot)

    def _genop_expr_int_binop(binop):
        def genop_expr_int_binop(self, op):
            return js.IntCast(binop(self._get_jsval(op.getarg(0)),
                                    self._get_jsval(op.getarg(1))))
        return genop_expr_int_binop

    genop_expr_int_lt = _genop_expr_int_binop(js.LessThan)
    genop_expr_int_le = _genop_expr_int_binop(js.LessThanEq)
    genop_expr_int_eq = _genop_expr_int_binop(js.Equal)
    genop_expr_int_ne = _genop_expr_int_binop(js.NotEqual)
    genop_expr_int_gt = _genop_expr_int_binop(js.GreaterThan)
    genop_expr_int_ge = _genop_expr_int_binop(js.GreaterThanEq)
    genop_expr_int_add = _genop_expr_int_binop(js.Plus)
    genop_expr_int_sub = _genop_expr_int_binop(js.Minus)
    genop_expr_int_mul = _genop_expr_int_binop(js.IMul)
    genop_expr_int_floordiv = _genop_expr_int_binop(js.Div)
    genop_expr_int_mod = _genop_expr_int_binop(js.Mod)
    genop_expr_int_and = _genop_expr_int_binop(js.And)
    genop_expr_int_or = _genop_expr_int_binop(js.Or)
    genop_expr_int_xor = _genop_expr_int_binop(js.Xor)
    genop_expr_int_lshift = _genop_expr_int_binop(js.LShift)
    genop_expr_int_rshift = _genop_expr_int_binop(js.RShift)
    genop_expr_uint_rshift = _genop_expr_int_binop(js.URShift)

    def _genop_expr_uint_binop(binop):
        def genop_expr_uint_binop(self, op):
            lhs = js.UnsignedCast(self._get_jsval(op.getarg(0)))
            rhs = js.UnsignedCast(self._get_jsval(op.getarg(1)))
            return binop(lhs, rhs)
        return genop_expr_uint_binop

    genop_expr_uint_lt = _genop_expr_uint_binop(js.LessThan)
    genop_expr_uint_le = _genop_expr_uint_binop(js.LessThanEq)
    genop_expr_uint_gt = _genop_expr_uint_binop(js.GreaterThan)
    genop_expr_uint_ge = _genop_expr_uint_binop(js.GreaterThanEq)

    def genop_expr_uint_floordiv(self, op):
        lhs = js.UnsignedCast(self._get_jsval(op.getarg(0)))
        rhs = js.UnsignedCast(self._get_jsval(op.getarg(1)))
        return js.UnsignedCast(js.Div(lhs, rhs))

    def _genop_withguard_int_binop_ovf(dblop, intop):
        def genop_withguard_int_binop_ovf(self, op, guardop):
            assert guardop.is_guard_overflow()
            lhs = self._get_jsval(op.getarg(0))
            rhs = self._get_jsval(op.getarg(1))
            resvar = self._get_jsval(op.result)
            # Temporary box to hold intermediate double result.
            dblresvar = self.bldr.allocate_doublevar()
            # XXX TODO: better way to detect overflow in asmjs?
            # We cast the operands to doubles, do the operation as a double,
            # then see if it remains unchanged when we round-trip through int.
            # The tricky case is multiplication, which we have to do twice
            # because double-mul is not the same as int-mul.
            dblres = dblop(js.DoubleCast(lhs), js.DoubleCast(rhs))
            self.bldr.emit_assignment(dblresvar, dblres)
            if dblop == intop:
                intres = js.SignedCast(dblresvar)
            else:
                intres = intop(lhs, rhs)
            self.bldr.emit_assignment(resvar, intres)
            # If the res makes it back to dbl unchanged, there was no overflow.
            self._prepare_guard_descr(guardop)
            if guardop.getopnum() == rop.GUARD_NO_OVERFLOW:
                test = js.NotEqual(dblresvar, js.DoubleCast(resvar))
            else:
                assert guardop.getopnum() == rop.GUARD_OVERFLOW
                test = js.Equal(dblresvar, js.DoubleCast(resvar))
            self._genop_guard_failure(test, guardop)
            # Discard temporary box.
            self.bldr.free_doublevar(dblresvar)
        return genop_withguard_int_binop_ovf

    genop_withguard_int_add_ovf = _genop_withguard_int_binop_ovf(js.Plus,
                                                                 js.Plus)
    genop_withguard_int_sub_ovf = _genop_withguard_int_binop_ovf(js.Minus,
                                                                 js.Minus)
    genop_withguard_int_mul_ovf = _genop_withguard_int_binop_ovf(js.Mul,
                                                                 js.IMul)

    def genop_int_force_ge_zero(self, op):
        argbox = op.getarg(0)
        arg = self._get_jsval(argbox)
        if isinstance(argbox, Box):
            if not isinstance(arg, js.Variable):
                arg = self._genop_realize_box(argbox)
        resvar = self._get_jsval(op.result)
        with self.bldr.emit_if_block(js.LessThan(arg, js.zero)):
            self.bldr.emit_assignment(resvar, js.zero)
        with self.bldr.emit_else_block():
            self.bldr.emit_assignment(resvar, arg)

    genop_expr_ptr_eq = genop_expr_int_eq
    genop_expr_ptr_ne = genop_expr_int_ne
    genop_expr_instance_ptr_eq = genop_expr_ptr_eq
    genop_expr_instance_ptr_ne = genop_expr_ptr_ne

    def genop_expr_same_as(self, op):
        return self._get_jsval(op.getarg(0))

    genop_expr_cast_ptr_to_int = genop_expr_same_as
    genop_expr_cast_int_to_ptr = genop_expr_same_as

    def _genop_expr_float_unaryop(operator):
        def genop_expr_float_unaryop(self, op):
            return operator(self._get_jsval(op.getarg(0)))
        return genop_expr_float_unaryop

    genop_expr_float_neg = _genop_expr_float_unaryop(js.UMinus)

    def genop_float_abs(self, op):
        argbox = op.getarg(0)
        arg = self._get_jsval(argbox)
        if isinstance(argbox, Box):
            if not isinstance(arg, js.Variable):
                arg = self._genop_realize_box(argbox)
        resvar = self._get_jsval(op.result)
        zero = js.ConstFloat(longlong.getfloatstorage(0.0))
        with self.bldr.emit_if_block(js.LessThan(arg, zero)):
            self.bldr.emit_assignment(resvar, js.UMinus(arg))
        with self.bldr.emit_else_block():
            self.bldr.emit_assignment(resvar, arg)

    def _genop_expr_float_binop(binop):
        def genop_expr_float_binop(self, op):
            return js.DoubleCast(binop(self._get_jsval(op.getarg(0)),
                                       self._get_jsval(op.getarg(1))))
        return genop_expr_float_binop

    genop_expr_float_add = _genop_expr_float_binop(js.Plus)
    genop_expr_float_sub = _genop_expr_float_binop(js.Minus)
    genop_expr_float_mul = _genop_expr_float_binop(js.Mul)
    genop_expr_float_truediv = _genop_expr_float_binop(js.Div)

    genop_expr_float_lt = _genop_expr_int_binop(js.LessThan)
    genop_expr_float_le = _genop_expr_int_binop(js.LessThanEq)
    genop_expr_float_eq = _genop_expr_int_binop(js.Equal)
    genop_expr_float_ne = _genop_expr_int_binop(js.NotEqual)
    genop_expr_float_gt = _genop_expr_int_binop(js.GreaterThan)
    genop_expr_float_ge = _genop_expr_int_binop(js.GreaterThanEq)

    def genop_expr_convert_float_bytes_to_longlong(self, op):
        return self._get_jsval(op.getarg(0))

    def genop_expr_convert_longlong_bytes_to_float(self, op):
        return self._get_jsval(op.getarg(0))

    def genop_expr_cast_float_to_int(self, op):
        return js.SignedCast(self._get_jsval(op.getarg(0)))

    def genop_expr_cast_int_to_float(self, op):
        return js.DoubleCast(self._get_jsval(op.getarg(0)))

    def genop_read_timestamp(self, op):
        # Simulate processor time using gettimeofday().
        # XXX TODO: Probably this is all sorts of technically incorrect.
        # It needs to write into the heap, so we use the frame as scratch.
        self._ensure_frame_depth(2*WORD)
        addr = js.JitFrameSlotAddr(0)
        self.bldr.emit_expr(js.CallFunc("gettimeofday", [addr]))
        secs = js.HeapData(js.Int32, addr)
        micros = js.HeapData(js.Int32, js.JitFrameSlotAddr(WORD))
        millis = js.Div(micros, js.ConstInt(1000))
        millis = js.Plus(millis, js.IMul(secs, js.ConstInt(1000)))
        self.bldr.emit_assignment(self._get_jsval(op.result), millis)

    #
    # Calls and Jumps and Exits, Oh My!
    #

    def genop_call(self, op):
        descr = op.getdescr()
        assert isinstance(descr, CallDescr)
        assert op.numargs() == len(descr.arg_classes) + 1
        addr = self._get_jsval(op.getarg(0))
        args = []
        i = 1
        while i < op.numargs():
            args.append(self._get_jsval(op.getarg(i)))
            i += 1
        self._genop_call(op, descr, addr, args)

    def genop_call_malloc_gc(self, op):
        descr = op.getdescr()
        assert isinstance(descr, CallDescr)
        assert op.numargs() == len(descr.arg_classes) + 1
        addr = self._get_jsval(op.getarg(0))
        args = []
        i = 1
        while i < op.numargs():
            args.append(self._get_jsval(op.getarg(i)))
            i += 1
        self._genop_call(op, descr, addr, args)
        resvar = self._get_jsval(op.result)
        with self.bldr.emit_if_block(js.Equal(resvar, js.zero)):
            self._genop_check_and_propagate_exception()

    def genop_cond_call(self, op):
        descr = op.getdescr()
        assert isinstance(descr, CallDescr)
        assert op.numargs() == len(descr.arg_classes) + 2
        cond = self._get_jsval(op.getarg(0))
        addr = self._get_jsval(op.getarg(1))
        args = []
        i = 2
        while i < op.numargs():
            args.append(self._get_jsval(op.getarg(i)))
            i += 1
        with self.bldr.emit_if_block(cond):
            self._genop_call(op, descr, addr, args)

    def genop_withguard_call_may_force(self, op, guardop):
        descr = op.getdescr()
        assert isinstance(descr, CallDescr)
        assert op.numargs() == len(descr.arg_classes) + 1
        addr = self._get_jsval(op.getarg(0))
        args = []
        i = 1
        while i < op.numargs():
            args.append(self._get_jsval(op.getarg(i)))
            i += 1
        with ctx_guard_not_forced(self, guardop):
            self._genop_call(op, descr, addr, args)

    def genop_withguard_call_release_gil(self, op, guardop):
        descr = op.getdescr()
        assert isinstance(descr, CallDescr)
        assert op.numargs() == len(descr.arg_classes) + 1
        addr = self._get_jsval(op.getarg(0))
        args = []
        i = 1
        while i < op.numargs():
            args.append(self._get_jsval(op.getarg(i)))
            i += 1
        with ctx_guard_not_forced(self, guardop):
            release = js.ConstInt(self.release_gil_addr)
            self.bldr.emit_expr(js.DynCallFunc("v", release, []))
            self._genop_call(op, descr, addr, args)
            reacquire = js.ConstInt(self.reacquire_gil_addr)
            self.bldr.emit_expr(js.DynCallFunc("v", reacquire, []))

    def genop_withguard_call_assembler(self, op, guardop):
        descr = op.getdescr()
        assert isinstance(descr, JitCellToken)
        frame = self._get_jsval(op.getarg(0))
        if op.numargs() == 2:
            virtref = self._get_jsval(op.getarg(1))
        else:
            virtref = js.zero
        jd = descr.outermost_jitdriver_sd
        assert jd is not None
        with ctx_guard_not_forced(self, guardop):
            # The GC-rewrite pass has allocated a frame and populated it.
            # Use the execute-trampoline helper to execute it to completion.
            # This may produce a new frame object, capture it in a temp box.
            exeaddr = self.execute_trampoline_addr
            funcid = descr.compiled_loop_token.compiled_function_id
            args = [js.ConstInt(funcid), frame]
            resvar = self.bldr.allocate_intvar()
            with ctx_allow_gc(self):
                call = js.DynCallFunc("iii", js.ConstInt(exeaddr), args)
                self.bldr.emit_assignment(resvar, call)
            # Load the descr resulting from that call.
            offset = self.cpu.get_ofs_of_frame_field('jf_descr')
            resdescr = js.HeapData(js.Int32,
                                   js.Plus(resvar, js.ConstInt(offset)))
            # Check if it's equal to done-with-this-frame.
            # The particular brand of DWTF depends on the result type.
            if op.result is None:
                dwtf = self.cpu.done_with_this_frame_descr_void
            else:
                kind = op.result.type
                if kind == INT:
                    dwtf = self.cpu.done_with_this_frame_descr_int
                elif kind == REF:
                    dwtf = self.cpu.done_with_this_frame_descr_ref
                elif kind == FLOAT:
                    dwtf = self.cpu.done_with_this_frame_descr_float
                else:
                    raise AssertionError(kind)
            gcref = cast_instance_to_gcref(dwtf)
            rgc._make_sure_does_not_move(gcref)
            dwtf = js.ConstInt(rffi.cast(lltype.Signed, gcref))
            with self.bldr.emit_if_block(js.Equal(resdescr, dwtf)):
                # If so, then we're on the happy fast path.
                # Reset the vable token  (whatever the hell that means...)
                # and return the result from the frame.
                if jd.index_of_virtualizable >= 0:
                    fielddescr = jd.vable_token_descr
                    assert isinstance(fielddescr, FieldDescr)
                    fieldaddr = js.Plus(virtref,
                                        js.ConstInt(fielddescr.offset))
                    self.bldr.emit_store(js.zero, fieldaddr, js.Int32)
                if op.result is not None:
                    kind = op.result.type
                    descr = self.cpu.getarraydescr_for_frame(kind)
                    offset = self.cpu.unpack_arraydescr(descr)
                    addr = js.Plus(resvar, js.ConstInt(offset))
                    typ = js.HeapType.from_kind(kind)
                    self.bldr.emit_load(self._get_jsval(op.result), addr, typ)
            # If not, then we need to invoke a helper function.
            with self.bldr.emit_else_block():
                if op.result is None:
                    callsig = "vii"
                elif op.result.type == FLOAT:
                    callsig = "fii"
                else:
                    callsig = "iii"
                args = [resvar, virtref]
                helpaddr = self.cpu.cast_adr_to_int(jd.assembler_helper_adr)
                helpaddr = js.ConstInt(helpaddr)
                with ctx_allow_gc(self, exclude=[op.result]):
                    call = js.DynCallFunc(callsig, helpaddr, args)
                    if op.result is None:
                        self.bldr.emit_expr(call)
                    else:
                        opresvar = self._get_jsval(op.result)
                        self.bldr.emit_assignment(opresvar, call)
            # Cleanup.
            self.bldr.free_intvar(resvar)

    def _genop_call(self, op, descr, addr, args):
        assert isinstance(descr, CallDescr)
        assert len(descr.arg_classes) == len(args)
        # Map CallDescr type tags into dynCall type tags.
        sigmap = {"i": "i", "r": "i", "f": "f", "v": "v"}
        callsig = sigmap[descr.result_type]
        i = 0
        while i < len(args):
            callsig += sigmap[descr.arg_classes[i]]
            i += 1
        with ctx_allow_gc(self, exclude=[op.result]):
            call = js.DynCallFunc(callsig, addr, args)
            if op.result is None:
                self.bldr.emit_expr(call)
            else:
                resvar = self._get_jsval(op.result)
                self.bldr.emit_assignment(resvar, call)
                # If the result is a less-than-full-sized integer,
                # mask off just the bits we expect.
                # XXX TODO: is this only required for the ctypes testing setup?
                if descr.result_type == "i" and descr.result_size < WORD:
                    mask = js.ConstInt((2 << (8 * descr.result_size - 1)) - 1)
                    self.bldr.emit_assignment(resvar, js.And(resvar, mask))

    def genop_force_token(self, op):
        self.bldr.emit_assignment(self._get_jsval(op.result), js.jitFrame)

    def genop_jump(self, op):
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        # For jumps to local loop, we can send arguments via boxes.
        if descr == self.bldr.loop_entry_token:
            argboxes = descr._asmjs_argboxes
            assert len(argboxes) == op.numargs()
            # We're going to use a bunch of temporary variables to swap the
            # the new values of the variables into the old, so that we don't
            # mess with the state of any boxes that we might need for a future
            # assignment.  This is the only place where a box can change value.
            tempvars = [None] * op.numargs()
            for i in range(op.numargs()):
                box = op.getarg(i)
                boxvar = self._get_jsval(box)
                if argboxes[i] != box:
                    if box.type == FLOAT:
                        tempvars[i] = self.bldr.allocate_doublevar()
                    else:
                        tempvars[i] = self.bldr.allocate_intvar()
                    self.bldr.emit_assignment(tempvars[i], boxvar)
            # Now copy each tempvar into the target variable for the argbox.
            for i in range(op.numargs()):
                box = op.getarg(i)
                if argboxes[i] != box:
                    boxvar = self._get_jsval(argboxes[i])
                    self.bldr.emit_assignment(boxvar, tempvars[i])
        # For jumps to a different function, we spill to the frame.
        # Each label has logic to load from frame when it's the jump target.
        else:
            arguments = op.getarglist()
            self._genop_write_output_args(arguments)
        self.bldr.emit_jump(descr)

    def genop_finish(self, op):
        descr = op.getdescr()
        fail_descr = cast_instance_to_gcref(descr)
        rgc._make_sure_does_not_move(fail_descr)
        # If there's a return value, write it into the frame.
        if op.numargs() == 1:
            self._genop_write_output_args(op.getarglist())
        # Write the descr into the frame slot.
        addr = js.JitFrameDescrAddr()
        self.bldr.emit_store(js.ConstPtr(fail_descr), addr, js.Int32)
        self.bldr.emit_exit()

    #
    # Guard-related things.
    #

    def genop_guard_true(self, op):
        self._prepare_guard_descr(op)
        test = js.UNot(self._get_jsval(op.getarg(0)))
        self._genop_guard_failure(test, op)

    def genop_guard_isnull(self, op):
        self._prepare_guard_descr(op)
        test = js.NotEqual(self._get_jsval(op.getarg(0)), js.zero)
        self._genop_guard_failure(test, op)

    def genop_guard_nonnull(self, op):
        self._prepare_guard_descr(op)
        test = js.UNot(self._get_jsval(op.getarg(0)))
        self._genop_guard_failure(test, op)

    def genop_guard_false(self, op):
        self._prepare_guard_descr(op)
        test = self._get_jsval(op.getarg(0))
        self._genop_guard_failure(test, op)

    def genop_guard_value(self, op):
        self._prepare_guard_descr(op)
        test = js.NotEqual(self._get_jsval(op.getarg(0)),
                           self._get_jsval(op.getarg(1)))
        self._genop_guard_failure(test, op)

    def genop_guard_class(self, op):
        self._prepare_guard_descr(op)
        objptr = self._get_jsval(op.getarg(0))
        clsptr = self._get_jsval(op.getarg(1))
        test = self._genop_expr_not_has_class(objptr, clsptr)
        self._genop_guard_failure(test, op)

    def _genop_expr_not_has_class(self, objptr, clsptr):
        # If compiled without type pointers, we have to read the "typeid"
        # from the first half-word of the object and compare it to the
        # expected typeid for the class.
        offset = self.cpu.vtable_offset
        if offset is not None:
            objcls = js.HeapData(js.Int32, js.Plus(objptr,
                                                   js.ConstInt(offset)))
            test = js.NotEqual(objcls, clsptr)
        else:
            typeid = js.And(js.HeapData(js.Int32, objptr), js.ConstInt(0xFFFF))
            test = js.NotEqual(typeid, js.ClassPtrTypeID(clsptr))
        return test

    def genop_guard_nonnull_class(self, op):
        self._prepare_guard_descr(op)
        # This is essentially short-circuiting logical or: the guard must
        # fail if the ref is null or it does not have the appropriate class.
        # Since asmjs doesn't have short-circuiting logical operators, we
        # simulate it with a temporary variable.
        objptr = self._get_jsval(op.getarg(0))
        clsptr = self._get_jsval(op.getarg(1))
        testvar = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(testvar, objptr)
        with self.bldr.emit_if_block(testvar):
            not_has_class = self._genop_expr_not_has_class(objptr, clsptr)
            self.bldr.emit_assignment(testvar, not_has_class)
        with self.bldr.emit_else_block():
            self.bldr.emit_assignment(testvar, js.UNot(testvar))
        self._genop_guard_failure(testvar, op)
        self.bldr.free_intvar(testvar)

    def genop_guard_exception(self, op):
        self._prepare_guard_descr(op)
        pos_exctyp = js.ConstInt(self.cpu.pos_exception())
        pos_excval = js.ConstInt(self.cpu.pos_exc_value())
        exctyp = js.HeapData(js.Int32, pos_exctyp)
        excval = js.HeapData(js.Int32, pos_excval)
        test = js.NotEqual(exctyp, self._get_jsval(op.getarg(0)))
        self._genop_guard_failure(test, op)
        if op.result is not None:
            self.bldr.emit_assignment(self._get_jsval(op.result), excval)
        self.bldr.emit_store(js.zero, pos_exctyp, js.Int32)
        self.bldr.emit_store(js.zero, pos_excval, js.Int32)

    def genop_guard_no_exception(self, op):
        self._prepare_guard_descr(op)
        pos_exctyp = js.ConstInt(self.cpu.pos_exception())
        exctyp = js.HeapData(js.Int32, pos_exctyp)
        test = js.NotEqual(exctyp, js.zero)
        self._genop_guard_failure(test, op)

    def genop_guard_not_invalidated(self, op):
        self._prepare_guard_descr(op)
        clt = self.current_clt
        translate_support_code = self.cpu.translate_support_code
        offset, size = symbolic.get_field_token(INVALIDATION_COUNTER,
                                                "counter",
                                                translate_support_code)
        assert size == js.Int32.size
        invalidation = rffi.cast(lltype.Signed, clt.invalidation)
        cur_val = js.HeapData(js.Int32, js.Plus(js.ConstInt(invalidation),
                                                js.ConstInt(offset)))
        orig_val = js.ConstInt(clt.invalidation.counter)
        test = js.NotEqual(cur_val, orig_val)
        self._genop_guard_failure(test, op)

    def _prepare_guard_descr(self, op):
        descr = op.getdescr()
        assert isinstance(descr, AbstractFailDescr)
        # Allocate a new function for this guard.
        # The initial implementation just fails back to the runner,
        # but it might get re-compiled into a bridge that does other
        # interesting things.
        guardjs = ASMJSBuilder(self.cpu)
        # Indenting generation of guard js, for visual clarity.
        if True:
            # If there might be an exception, capture it to the frame.
            if self._guard_might_have_exception(op):
                pos_exctyp = js.ConstInt(self.cpu.pos_exception())
                pos_excval = js.ConstInt(self.cpu.pos_exc_value())
                exctyp = js.HeapData(js.Int32, pos_exctyp)
                excval = js.HeapData(js.Int32, pos_excval)
                with guardjs.emit_if_block(exctyp):
                    addr = js.JitFrameGuardExcAddr()
                    guardjs.emit_store(excval, addr, js.Int32)
                    guardjs.emit_store(js.zero, pos_exctyp, js.Int32)
                    guardjs.emit_store(js.zero, pos_excval, js.Int32)
            # Write the fail_descr into the frame.
            fail_descr = cast_instance_to_gcref(descr)
            addr = js.JitFrameDescrAddr()
            guardjs.emit_store(js.ConstPtr(fail_descr), addr, js.Int32)
            # That's it!
            funcid = self._compile(guardjs)
        # Store it on the guard, so we can re-compile it in future.
        descr._asmjs_funcid = funcid
        descr._asmjs_clt = self.current_clt
        self.current_clt.compiled_guard_funcs.append(descr._asmjs_funcid)
        return descr

    def _genop_guard_failure(self, test, op):
        descr = op.getdescr()
        assert isinstance(descr, AbstractFailDescr)
        assert descr._asmjs_funcid
        with self.bldr.emit_if_block(test):
            # Write the failargs into the frame.
            # This is the only mechanism we have available for arg passing.
            locations = self._genop_write_output_args(op.getfailargs())
            descr._asmjs_faillocs = locations
            # Execute the separately-compiled function implementing this guard.
            self.bldr.emit_goto(descr._asmjs_funcid)

    def _guard_might_have_exception(self, op):
        opnum = op.getopnum()
        if opnum == rop.GUARD_EXCEPTION:
            return True
        if opnum == rop.GUARD_NO_EXCEPTION:
            return True
        if opnum == rop.GUARD_NOT_FORCED:
            return True
        return False

    def _genop_check_and_propagate_exception(self):
        if not self.cpu.propagate_exception_descr:
            return
        pos_exctyp = js.ConstInt(self.cpu.pos_exception())
        pos_excval = js.ConstInt(self.cpu.pos_exc_value())
        exctyp = js.HeapData(js.Int32, pos_exctyp)
        excval = js.HeapData(js.Int32, pos_excval)
        with self.bldr.emit_if_block(exctyp):
            # Store the exception on the frame, and clear it.
            self.bldr.emit_store(excval, js.JitFrameGuardExcAddr(), js.Int32)
            self.bldr.emit_store(js.zero, pos_exctyp, js.Int32)
            self.bldr.emit_store(js.zero, pos_excval, js.Int32)
            # Store the special propagate-exception descr on the frame.
            descr = cast_instance_to_gcref(self.cpu.propagate_exception_descr)
            addr = js.JitFrameDescrAddr()
            self.bldr.emit_store(js.ConstPtr(descr), addr, js.Int32)
            # Bail back to the invoking code to deal with it.
            self.bldr.emit_exit()

    #
    #  GC-related things.
    #

    def genop_call_malloc_nursery(self, op):
        sizebox = op.getarg(0)
        # Sanity-check for correct alignment.
        assert isinstance(sizebox, ConstInt)
        size = sizebox.getint()
        assert size & (WORD-1) == 0
        self._genop_malloc_nursery(op, sizebox)

    def genop_call_malloc_nursery_varsize_frame(self, op):
        sizebox = op.getarg(0)
        self._genop_malloc_nursery(op, sizebox)

    def _genop_malloc_nursery(self, op, sizebox):
        gc_ll_descr = self.cpu.gc_ll_descr
        sizevar = self._get_jsval(sizebox)
        if isinstance(sizebox, Box):
            if not isinstance(sizevar, js.Variable):
                sizevar = self._genop_realize_box(sizebox)
        # This is essentially an in-lining of MiniMark.malloc_fixedsize_clear()
        nfree_addr = js.ConstInt(gc_ll_descr.get_nursery_free_addr())
        ntop_addr = js.ConstInt(gc_ll_descr.get_nursery_top_addr())
        nfree = js.HeapData(js.Int32, nfree_addr)
        ntop = js.HeapData(js.Int32, ntop_addr)
        # Optimistically, we can just use the space at nursery_free.
        resvar = self._get_jsval(op.result)
        self.bldr.emit_assignment(resvar, nfree)
        new_nfree = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(new_nfree, js.Plus(resvar, sizevar))
        # But we have to check whether we overflowed nursery_top.
        with self.bldr.emit_if_block(js.LessThanEq(new_nfree, ntop)):
            # If we didn't, we're all good, just increment nursery_free.
            self.bldr.emit_store(new_nfree, nfree_addr, js.Int32)
        with self.bldr.emit_else_block():
            # If we did, we have to call into the GC for a collection.
            mallocfn = js.ConstInt(self.gc_malloc_nursery_addr)
            with ctx_allow_gc(self, exclude=[op.result]):
                call = js.DynCallFunc("ii", mallocfn, [sizevar])
                self.bldr.emit_assignment(resvar, call)
            self._genop_check_and_propagate_exception()
        # Cleanup temp vars.
        self.bldr.free_intvar(new_nfree)

    def genop_call_malloc_nursery_varsize(self, op):
        gc_ll_descr = self.cpu.gc_ll_descr
        arraydescr = op.getdescr()
        assert isinstance(arraydescr, ArrayDescr)
        if hasattr(gc_ll_descr, 'minimal_size_in_nursery'):
            assert arraydescr.basesize >= gc_ll_descr.minimal_size_in_nursery
        kind = op.getarg(0).getint()
        itemsize = op.getarg(1).getint()
        lengthbox = op.getarg(2)
        assert isinstance(lengthbox, BoxInt)
        lengthvar = self._get_jsval(lengthbox)
        # Figure out the total size to be allocated.
        # It's gcheader + basesize + length*itemsize, rounded up to wordsize.
        if hasattr(gc_ll_descr, 'gcheaderbuilder'):
            size_of_header = gc_ll_descr.gcheaderbuilder.size_gc_header
        else:
            size_of_header = WORD
        constsize = size_of_header + arraydescr.basesize
        if itemsize % WORD == 0:
            num_items = lengthvar
        else:
            assert itemsize < WORD
            items_per_word = js.ConstInt(WORD / itemsize)
            padding = js.Mod(lengthvar, items_per_word)
            padding = js.Minus(items_per_word, padding)
            num_items = js.Plus(lengthvar, padding)
        calc_totalsize = js.IMul(num_items, js.ConstInt(itemsize))
        calc_totalsize = js.Plus(js.ConstInt(constsize), calc_totalsize)
        totalsize = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(totalsize, calc_totalsize)
        # This is essentially an in-lining of MiniMark.malloc_fixedsize_clear()
        nfree_addr = js.ConstInt(gc_ll_descr.get_nursery_free_addr())
        ntop_addr = js.ConstInt(gc_ll_descr.get_nursery_top_addr())
        nfree = js.HeapData(js.Int32, nfree_addr)
        ntop = js.HeapData(js.Int32, ntop_addr)
        maxsize = js.ConstInt(gc_ll_descr.max_size_of_young_obj - WORD * 2)
        # Optimistically, we can just use the space at nursery_free.
        resvar = self._get_jsval(op.result)
        self.bldr.emit_assignment(resvar, nfree)
        new_nfree = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(new_nfree, js.Plus(resvar, totalsize))
        # But we have to check whether we overflowed nursery_top,
        # or created an object too large for the nursery.
        check = js.And(js.LessThanEq(new_nfree, ntop),
                       js.LessThan(totalsize, maxsize))
        with self.bldr.emit_if_block(check):
            # If we fit in the nursery, we're all good!
            # Increment nursery_free and set type flags on the object.
            self.bldr.emit_store(new_nfree, nfree_addr, js.Int32)
            self.bldr.emit_store(js.ConstInt(arraydescr.tid), resvar, js.Int32)
        with self.bldr.emit_else_block():
            # If it didn't fit in the nursery, we have to call out to malloc.
            if kind == rewrite.FLAG_ARRAY:
                args = [js.ConstInt(WORD),
                        js.ConstInt(arraydescr.tid),
                        lengthvar]
                callsig = "iiii"
                mallocfn = self.gc_malloc_array_addr
            else:
                args = [lengthvar]
                callsig = "ii"
                if kind == rewrite.FLAG_STR:
                    mallocfn = self.gc_malloc_str_addr
                else:
                    assert kind == rewrite.FLAG_UNICODE
                    mallocfn = self.gc_malloc_unicode_addr
            mallocfn = js.ConstInt(mallocfn)
            with ctx_allow_gc(self, exclude=[op.result]):
                call = js.DynCallFunc(callsig, mallocfn, args)
                self.bldr.emit_assignment(resvar, call)
            self._genop_check_and_propagate_exception()
        # That's it!  Cleanup temp variables.
        self.bldr.free_intvar(new_nfree)
        self.bldr.free_intvar(totalsize)

    def genop_cond_call_gc_wb(self, op):
        assert op.result is None
        self._genop_write_barrier(op.getarglist())

    def genop_cond_call_gc_wb_array(self, op):
        assert op.result is None
        self._genop_write_barrier(op.getarglist(), array=True)

    def _genop_write_barrier(self, arguments, array=False):
        # Decode and grab the necessary function pointer.
        # If it's zero, the GC doesn't need a write barrier here.
        wbdescr = self.cpu.gc_ll_descr.write_barrier_descr
        if not wbdescr:
            return
        card_marking = False
        if array and wbdescr.jit_wb_cards_set != 0:
            assert (wbdescr.jit_wb_cards_set_byteofs ==
                    wbdescr.jit_wb_if_flag_byteofs)
            card_marking = True
        if not array:
            wbfunc = wbdescr.get_write_barrier_fn(self.cpu)
        else:
            wbfunc = wbdescr.get_write_barrier_from_array_fn(self.cpu)
        if wbfunc == 0:
            return
        # Here we are inlining a bunch of code from the write-barrier,
        # in a similar way to how it gets inlined in the non-jitted code.
        # The structure of the generated code looks like this for plain
        # objects:
        #
        #    if (obj has JIT_WB_IF_FLAG) {
        #      dynCall(write_barrier, obj)
        #    }
        #
        # And like this for arrays with potential card-marking:
        #
        #    if (obj has JIT_WB_IF_FLAG|JIT_WB_CARDS_SET) {
        #      if (not obj has JIT_WB_CARDS_SET) {
        #        dynCall(write_barrier, obj)
        #      }
        #      if (obj has JIT_WB_CARDS_SET) {
        #        do the card marking
        #      }
        #    }
        #
        # XXX TODO: would this be neater if split into separate functions?
        #
        obj = self._get_jsval(arguments[0])
        flagaddr = js.Plus(obj, js.ConstInt(wbdescr.jit_wb_if_flag_byteofs))
        flagbyte = js.HeapData(js.Int8, flagaddr)
        flagbytevar = self.bldr.allocate_intvar()
        chk_flag = js.ConstInt(wbdescr.jit_wb_if_flag_singlebyte)
        chk_card = js.zero
        flag_has_cards = js.zero
        if card_marking:
            chk_card = js.ConstInt(wbdescr.jit_wb_cards_set_singlebyte)
            flag_has_cards = js.And(flagbytevar, chk_card)
        flag_needs_wb = js.And(flagbytevar, js.Or(chk_flag, chk_card))
        # Check if we actually need to establish a writebarrier.
        self.bldr.emit_assignment(flagbytevar, flagbyte)
        with self.bldr.emit_if_block(flag_needs_wb):
            call = js.DynCallFunc("vi", js.ConstInt(wbfunc), [obj])
            if not card_marking:
                self.bldr.emit_expr(call)
            else:
                with self.bldr.emit_if_block(js.UNot(flag_has_cards)):
                    # This might change the GC flags on the object.
                    self.bldr.emit_expr(call)
                    self.bldr.emit_assignment(flagbytevar, flagbyte)
                # Check if we need to set a card-marking flag.
                with self.bldr.emit_if_block(flag_has_cards):
                    # This is how we decode the array index into a card
                    # bit to set.  Logic cargo-culted from x86 backend.
                    which = self._get_jsval(arguments[1])
                    card_shift = js.ConstInt(wbdescr.jit_wb_card_page_shift)
                    byte_index = js.RShift(which, card_shift)
                    byte_ofs = js.UNeg(js.RShift(byte_index, js.ConstInt(3)))
                    byte_mask = js.LShift(js.ConstInt(1),
                                          js.And(byte_index, js.ConstInt(7)))
                    byte_addr = js.Plus(obj, byte_ofs)
                    old_byte_data = js.HeapData(js.Int8, byte_addr)
                    new_byte_data = js.Or(old_byte_data, byte_mask)
                    self.bldr.emit_store(new_byte_data, byte_addr, js.Int8)
        self.bldr.free_intvar(flagbytevar)

    def _genop_store_gcmap(self):
        # If there's nothing spilled, no gcmap is needed.
        if self.spilled_frame_offset == 0:
            self.bldr.emit_store(js.zero, js.JitFrameGCMapAddr(), js.Int32)
            return
        # Make a new gcmap sized to match current size of frame.
        # Remember, our offsets are in bytes but the gcmap indexes whole words.
        frame_size = self.spilled_frame_offset // WORD
        gcmap_size = (frame_size // WORD // 8) + 1
        rawgcmap = lltype.malloc(jitframe.GCMAP, gcmap_size, flavor="raw")
        gcmap = rffi.cast(lltype.Ptr(jitframe.GCMAP), rawgcmap)
        for i in xrange(gcmap_size):
            gcmap[i] = r_uint(0)
        # Set a bit for every REF that has been spilled.
        num_refs = 0
        for pos, box in self.spilled_frame_values.iteritems():
            if box and box.type == REF:
                pos = pos // WORD
                gcmap[pos // WORD // 8] |= r_uint(1) << (pos % (WORD * 8))
                num_refs += 1
        # Do we actually have any refs?
        # Generate code to store it to the frame, and keep it alive
        # by attaching it to the clt.
        if not num_refs:
            lltype.free(gcmap, flavor="raw")
            self.bldr.emit_store(js.zero, js.JitFrameGCMapAddr(), js.Int32)
        else:
            self.current_clt.compiled_gcmaps.append(gcmap)
            gcmapref = js.ConstInt(self.cpu.cast_ptr_to_int(gcmap))
            self.bldr.emit_store(gcmapref, js.JitFrameGCMapAddr(), js.Int32)
            # We might have just stored some young pointers into the frame.
            # Emit a write barrier just in case.
            # XXX TODO: x86 backend only does that when reloading after a gc.
            self._genop_write_barrier([js.jitFrame])

    def genop_debug_merge_point(self, op):
        pass

    def genop_jit_debug(self, op):
        pass

    def genop_keepalive(self, op):
        pass

    def not_implemented_op_withguard(self, op, guardop):
        self.not_implemented_op(op)
        self.not_implemented_op(guardop)

    def not_implemented_op_expr(self, op):
        self.not_implemented_op(op)
        return js.zero

    def not_implemented_op(self, op):
        """Insert code for an as-yet-unimplemented operation.

        This inserts a comment into the generated code to mark where
        the unimplemented op would have gone, and sets a flag to let us
        know that the trace is incomplete.  It's useful for debugging
        but would never be triggered in the final version.
        """
        raise RuntimeError("Unimplemented", op)
        self._print_op(op)
        self.has_unimplemented_ops = True
        self.bldr.emit_comment("NOT IMPLEMENTED: %s" % (op,))
        for i in range(op.numargs()):
            arg = op.getarg(i)
            if isinstance(arg, Box):
                self._get_jsval(arg)
            self.bldr.emit_comment("    ARG: %s" % (arg,))
        if op.result is not None and isinstance(op.result, Box):
            self._get_jsval(op.result)
        self.bldr.emit_comment("    RESULT: %s" % (op.result,))

    def _print_op(self, op):
        print "OPERATION:", op
        for i in range(op.numargs()):
            print "  ARG:", op.getarg(i)
        print "  RES:", op.result


class ctx_spill_to_frame(object):

    def __init__(self, assembler):
        self.assembler = assembler
        self.orig_spilled_frame_offset = 0

    def _get_jsval(self, jitval):
        return self.assembler._get_jsval(jitval)

    def __enter__(self):
        # Remember the current offset, so we can pop all items
        # after this when we exit the context.
        self.orig_spilled_frame_offset = self.assembler.spilled_frame_offset
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        # Pop any items that were pushed in this context.
        orig_offset = self.orig_spilled_frame_offset
        self.assembler._ensure_frame_depth(self.assembler.spilled_frame_offset)
        for pos, box in self.assembler.spilled_frame_values.items():
            if pos >= orig_offset:
                del self.assembler.spilled_frame_values[pos]
                self.assembler.spilled_frame_locations[box].remove(pos)
                if not self.assembler.spilled_frame_locations[box]:
                    del self.assembler.spilled_frame_locations[box]
        # Reset frame metadata to point to restored offset.
        self.assembler.spilled_frame_offset = orig_offset
        self.assembler._genop_store_gcmap()

    def is_spilled(self, box):
        try:
            self.assembler.spilled_frame_locations[box]
        except KeyError:
            return False
        else:
            return True

    def genop_spill_to_frame(self, box, offset=-1):
        typ = js.HeapType.from_box(box)
        # Allocate it a position at the next available offset.
        # Align each value to a multiple of its size.
        if offset == -1:
            offset = self.assembler.spilled_frame_offset
            alignment = offset % typ.size
            if alignment:
                offset += alignment
        if offset >= self.assembler.spilled_frame_offset:
            self.assembler.spilled_frame_offset = offset + typ.size
        # Generate code to write the value into the frame.
        addr = js.JitFrameSlotAddr(offset)
        if isinstance(box, Box):
            boxexpr = self.assembler._genop_realize_box(box)
        else:
            boxexpr = self._get_jsval(box)
        self.assembler.bldr.emit_store(boxexpr, addr, typ)
        # Record where we spilled it.
        if box not in self.assembler.spilled_frame_locations:
            self.assembler.spilled_frame_locations[box] = []
        self.assembler.spilled_frame_locations[box].append(offset)
        self.assembler.spilled_frame_values[offset] = box
        return offset

class ctx_guard_not_forced(ctx_spill_to_frame):

    def __init__(self, assembler, guardop):
        ctx_spill_to_frame.__init__(self, assembler)
        self.guardop = guardop

    def __enter__(self):
        ctx_spill_to_frame.__enter__(self)
        bldr = self.assembler.bldr
        # Store the force-descr where forcing code can find it.
        descr = self.assembler._prepare_guard_descr(self.guardop)
        faildescr = js.ConstPtr(cast_instance_to_gcref(descr))
        bldr.emit_store(faildescr, js.JitFrameForceDescrAddr(), js.Int32)
        # Write the potential failargs into the frame.
        # We have to spill them here because the forcing logic might
        # need to read them out to populate the virtualizable.
        # They must be spilled at their final output location.
        assert self.orig_spilled_frame_offset == 0
        failargs = self.guardop.getfailargs()
        locations = self.assembler._get_frame_locations(failargs)
        assert len(failargs) == len(locations)
        for i in xrange(len(failargs)):
            failarg = failargs[i]
            # Careful, some boxes may not have a value yet.
            if failarg and failarg not in self.assembler.box_to_jsval:
                continue
            self.genop_spill_to_frame(failarg, locations[i])
        self.assembler._genop_store_gcmap()
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        # Emit the guard check, testing for whether jf_descr has been set.
        descr = js.HeapData(js.Int32, js.JitFrameDescrAddr())
        test = js.NotEqual(descr, js.zero)
        self.assembler._genop_guard_failure(test, self.guardop)
        # It's now safe to pop from the frame as usual.
        ctx_spill_to_frame.__exit__(self, exc_typ, exc_val, exc_tb)


class ctx_allow_gc(ctx_spill_to_frame):

    def __init__(self, assembler, exclude=None):
        ctx_spill_to_frame.__init__(self, assembler)
        self.exclude = exclude

    def __enter__(self):
        ctx_spill_to_frame.__enter__(self)
        bldr = self.assembler.bldr
        # Spill any active REF boxes into the frame.
        for box in self.assembler.box_to_jsval:
            if not box or box.type != REF:
                continue
            if self.exclude is not None and box in self.exclude:
                continue
            if not self.is_spilled(box):
                self.genop_spill_to_frame(box)
        self.assembler._genop_store_gcmap()
        # Push the jitframe itself onto the gc shadowstack.
        # We do the following:
        #   * get a pointer to the pointer to the root-stack top.
        #   * deref it to get the root-stack top, and write the frame there.
        #   * in-place increment root-stack top via its pointer.
        gcrootmap = self.assembler.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rstaddr = js.ConstInt(gcrootmap.get_root_stack_top_addr())
            rst = js.HeapData(js.Int32, rstaddr)
            bldr.emit_store(js.jitFrame, rst, js.Int32)
            newrst = js.Plus(rst, js.ConstInt(WORD))
            bldr.emit_store(newrst, rstaddr, js.Int32)
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        bldr = self.assembler.bldr
        # Pop the jitframe from the root-stack.
        # This is an in-place decrement of root-stack top.
        gcrootmap = self.assembler.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rstaddr = js.ConstInt(gcrootmap.get_root_stack_top_addr())
            rst = js.HeapData(js.Int32, rstaddr)
            newrst = js.Minus(rst, js.ConstInt(WORD))
            bldr.emit_store(newrst, rstaddr, js.Int32)
            # For moving GCs, the address of the jitframe may have changed.
            # Read the possibly-updated address out of root-stack top.
            # NB: this instruction re-evaluates the HeapData expression in rst.
            bldr.emit_assignment(js.jitFrame, js.HeapData(js.Int32, rst))
        # Similarly, read potential new addresss of any spilled boxes.
        # XXX TODO: don't double-load boxes that appear multiple times.
        for pos, box in self.assembler.spilled_frame_values.iteritems():
            if not box or box.type != REF:
                continue
            if self.exclude is not None and box in self.exclude:
                continue
            addr = js.JitFrameSlotAddr(pos)
            bldr.emit_load(self._get_jsval(box), addr, js.Int32)
        # It's now safe to pop from the frame as usual.
        ctx_spill_to_frame.__exit__(self, exc_typ, exc_val, exc_tb)
        # XXX TODO x86 backend seems to put a wb here.  Why?
        self.assembler._genop_write_barrier([js.jitFrame])


# Build a dispatch table mapping opnums to the method that emits code for them.
# There are two different method signatures, depending on whether we can work
# with a single opcode or require an opcode/guard pair.

genop_list = [AssemblerASMJS.not_implemented_op] * rop._LAST
genop_expr_list = [AssemblerASMJS.not_implemented_op_expr] * rop._LAST
genop_withguard_list = [AssemblerASMJS.not_implemented_op_withguard] * rop._LAST

for name, value in AssemblerASMJS.__dict__.iteritems():
    if name.startswith('genop_withguard_'):
        opname = name[len('genop_withguard_'):]
        num = getattr(rop, opname.upper())
        genop_withguard_list[num] = value
    elif name.startswith('genop_expr_'):
        opname = name[len('genop_expr_'):]
        num = getattr(rop, opname.upper())
        genop_expr_list[num] = value
    elif name.startswith('genop_'):
        opname = name[len('genop_'):]
        num = getattr(rop, opname.upper())
        genop_list[num] = value
