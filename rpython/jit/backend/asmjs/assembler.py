
import os
import sys

from rpython.rlib import rgc
from rpython.rlib.rarithmetic import r_uint, intmask
from rpython.rlib.objectmodel import we_are_translated
from rpython.jit.backend.llsupport import symbolic, jitframe, rewrite
from rpython.jit.backend.llsupport.regalloc import compute_vars_longevity
from rpython.jit.backend.llsupport.descr import (unpack_fielddescr,
                                                 unpack_arraydescr,
                                                 unpack_interiorfielddescr,
                                                 ArrayDescr, CallDescr,
                                                 FieldDescr)
from rpython.jit.codewriter import longlong
from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.rtyper.lltypesystem import lltype, rffi, rstr
from rpython.rtyper.annlowlevel import cast_instance_to_gcref, llhelper
from rpython.jit.backend.model import CompiledLoopToken
from rpython.jit.metainterp.compile import DoneWithThisFrameDescrVoid
from rpython.jit.metainterp.resoperation import rop, ResOperation
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


# XXX TODO: this is workaround for my inability to get concrete descrs
# of high-leveltypes.  I attach a low-level type to then and use descrs
# off that.  It's pretty dumb.  How to do this better?

INVALIDATION_COUNTER = lltype.Struct(
    "INVALIDATIONCOUNTER",
    ("counter", lltype.Signed)
)


GUARD_LABEL = lltype.GcStruct(
    "COMPILEDGUARDLABEL",
    ("label", lltype.Signed)
)


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
            d = gc_ll_descr
            self.gc_malloc_nursery_addr = d.get_malloc_slowpath_addr()
            self.gc_malloc_array_addr = d.get_malloc_slowpath_array_addr()
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

    def teardown(self):
        self.current_clt = None
        self.bldr = None

    def assemble_loop(self, loopname, inputargs, operations, looptoken, log):
        """Assemble and compile a new loop function from the given trace."""
        clt = CompiledLoopTokenASMJS(self.cpu, looptoken.number)
        rgc._make_sure_does_not_move(clt)
        looptoken.compiled_loop_token = clt
        self.setup(looptoken)
        clt.compiled_funcid = support.jitReserve()
        label = self.add_code_to_loop(clt, inputargs, operations)
        assert label == 0
        self.reassemble(clt)
        clt._ll_initial_locs = clt.compiled_blocks[0].inputlocs
        self.teardown()

    def assemble_bridge(self, faildescr, inputargs, operations,
                        original_loop_token, log):
        """Assemble, compile and link a new bridge from the given trace."""
        assert isinstance(faildescr, AbstractFailDescr)
        self.setup(original_loop_token)
        clt = self.current_clt
        assert faildescr._asmjs_funcid == clt.compiled_funcid
        label = self.add_code_to_loop(clt, inputargs, operations, faildescr)
        faildescr._asmjs_glbl.label = label
        self.reassemble(clt)
        self.teardown()

    def redirect_call_assembler(self, oldlooptoken, newlooptoken):
        oldCLT = oldlooptoken.compiled_loop_token
        newCLT = newlooptoken.compiled_loop_token
        oldCLT.redirected_to = newCLT.compiled_funcid
        if newCLT.redirected_funcids is None:
            newCLT.redirected_funcids = []
        newCLT.redirected_funcids.append(oldCLT.compiled_funcid)
        support.jitCopy(newCLT.compiled_funcid, oldCLT.compiled_funcid)

    def free_loop_and_bridges(self, compiled_loop_token):
        # All freeing is taken care of in the CLT destructor.
        pass

    def invalidate_loop(self, looptoken):
        looptoken.compiled_loop_token.invalidation.counter += 1

    def add_code_to_loop(self, clt, inputargs, operations, intoken=None):
        # Re-write to use lower level GC operations, and record
        # any inlined GC refs to the CLT.
        gcrefs = clt.inlined_gcrefs
        gcdescr = self.cpu.gc_ll_descr
        operations = gcdescr.rewrite_assembler(self.cpu, operations, gcrefs)
        # Split the new operations up into labelled blocks.
        start_op = 0
        first_new_label = len(clt.compiled_blocks)
        for i in xrange(len(operations)):
            op = operations[i]
            # Guard descrs need to be told the current funcid.
            # They initially get label = 0, which is changed when bridged.
            if op.is_guard():
                faildescr = op.getdescr()
                assert isinstance(faildescr, AbstractFailDescr)
                faildescr._asmjs_clt = clt
                faildescr._asmjs_funcid = clt.compiled_funcid
                faildescr._asmjs_glbl = lltype.malloc(GUARD_LABEL, flavor="gc")
                rgc._make_sure_does_not_move(faildescr._asmjs_glbl)
                faildescr._asmjs_glbl.label = 0
            # Label descrs start a new block.
            # They need to be told the current funcid.
            elif op.getopnum() == rop.LABEL:
                labeldescr = op.getdescr()
                assert isinstance(labeldescr, TargetToken)
                # Make the preceding operations into a block.
                # NB: if the first op is a label, this makes an empty block.
                # That's OK for now; it might do some arg shuffling etc.
                new_block = CompiledBlockASMJS(
                  self, clt, len(clt.compiled_blocks), intoken, inputargs,
                  operations[start_op:i], labeldescr, op.getarglist(),
                )
                clt.compiled_blocks.append(new_block)
                # Tell the label about its eventual location in the clt.
                labeldescr._asmjs_clt = clt
                labeldescr._asmjs_funcid = clt.compiled_funcid
                labeldescr._asmjs_label = len(clt.compiled_blocks)
                # Start a new block from this label.
                start_op = i
                intoken = labeldescr
                inputargs = op.getarglist()
        # Make the final block.
        if start_op < len(operations):
            new_block = CompiledBlockASMJS(
              self, clt, len(clt.compiled_blocks), intoken, inputargs,
              operations[start_op:], None, [],
            )
            clt.compiled_blocks.append(new_block)
        # Return the label of the entry to the new block.
        if SANITYCHECK:
            assert len(clt.compiled_blocks) > 0
        return first_new_label

    def reassemble(self, clt):
        """Re-compile the asmjs function for the given look token.

        The code is a simple switch-in-a-loop dispatching between the blocks.
        It looks like this:

          function jitted(frame) {

            // Initialization code for the loop.
            label = <load label from frame>

            // Load the input args, for the initial target label.
            switch(label) {
              case 0:
                <load input args for block 0>
              ...
              case N:
                <load input args for block N>
              default:
                return frame
            }

            // Dispatch between blocks until one exits.
            while (1) {
              switch(label) {
                case 0:
                  <code for for block 0>
                  label = X
                  continue
                ...
                case N:
                  <code for block N>
                  return frame
                default:
                  return frame
              }
            }
          }

        The switch-in-a-loop construct is known to be inefficient.  Ideally we
        would use something like emscripten's relooper algorithm to generate
        higher-level control flow constructs here.  But it should't be too
        bad when used with AOT asmjs compilation, and it's a solid start.
        """
        # A variable to hold the current jump target label.
        # We initialize it with the low 8 bits of jf_force_descr,
        # which the trampoline uses to pass in the target label.
        # We clear jf_force_descr once it's loaded.
        # XXX TODO: maybe just pass it in as a second arg to the function?
        self.label_var = self.bldr.allocate_intvar()
        self.bldr.emit_comment("LOAD INITIAL LABEL")
        self._genop_get_frame_next_call(self.label_var, js.frame)
        masked_label = js.And(self.label_var, js.ConstInt(0xFF))
        self.bldr.emit_assignment(self.label_var, masked_label)
        self._genop_set_frame_next_call(js.frame, js.zero, js.zero)

        # A suite of variables to hold the input args for the loop.
        # Every time we go to jump to a block, we populate a subset of these.
        # They are never freed, which simplifies the generated code.
        self.input_vars_int = []
        self.input_vars_double = []
        for block in clt.compiled_blocks:
            block.setup()
            while block.num_int_args > len(self.input_vars_int):
                self.input_vars_int.append(self.bldr.allocate_intvar())
            while block.num_double_args > len(self.input_vars_double):
                self.input_vars_double.append(self.bldr.allocate_doublevar())

        # A variable to hold the final required frame depth.
        # The initial value of this var is set once all code is generated.
        self.req_depth_var = self.bldr.allocate_intvar()

        # We check the depth of the frame at entry to the function.
        # If it's too small then we rellocate it via a helper.
        cur_depth = js.HeapData(js.Int32, js.FrameSizeAddr())
        frame_too_small = js.LessThan(cur_depth, self.req_depth_var)
        self.bldr.emit_comment("CHECK FRAME DEPTH")
        with self.bldr.emit_if_block(frame_too_small):
            # We must store a gcmap to prevent input args from being gc'd.
            # The layout of input args depends on the target label.
            gcmapaddr = js.FrameGCMapAddr()
            with self.bldr.emit_switch_block(self.label_var):
                for block in clt.compiled_blocks:
                    with self.bldr.emit_case_block(js.ConstInt(block.label)):
                        gcmap = block.initial_gcmap
                        gcmapadr = self.cpu.cast_ptr_to_int(gcmap)
                        gcmapref = js.ConstInt(gcmapadr)
                        addr = js.FrameGCMapAddr()
                        self.bldr.emit_store(gcmapref, gcmapaddr, js.Int32)
            # Now we can call the helper function.
            # There might be an exception active, which we want to preserve.
            reallocfn = js.ConstInt(self.cpu.realloc_frame)
            args = [js.frame, self.req_depth_var]
            with ctx_preserve_exception(self):
                newframe = js.DynCallFunc("iii", reallocfn, args)
                self.bldr.emit_assignment(js.frame, newframe)
        self.bldr.free_intvar(self.req_depth_var)

        # Load input args for the block being entered.
        self.bldr.emit_comment("LOAD INPUT ARGS")
        with self.bldr.emit_switch_block(self.label_var):
            for block in clt.compiled_blocks:
                with self.bldr.emit_case_block(js.ConstInt(block.label)):
                    block._genop_load_input_args()

        # Generate the dispatch loop, with the body of
        # each block inside switch statement.
        self.bldr.emit_comment("MAIN DISPATCH LOOP")
        with self.bldr.emit_while_block(js.true):
            with self.bldr.emit_switch_block(self.label_var):
                for block in clt.compiled_blocks:
                    with self.bldr.emit_case_block(js.ConstInt(block.label)):
                        block._genop_block_body()
                    block.teardown()

        # Insert the final required frame depth into the variable.
        req_depth = clt.frame_info.jfi_frame_depth
        self.bldr.set_initial_value_intvar(self.req_depth_var, req_depth)

        # Compile the replacement source code for our function.
        jssrc = self.bldr.finish()
        os.write(2, jssrc)
        support.jitRecompile(clt.compiled_funcid, jssrc)
        if clt.redirected_funcids is not None:
            for dstid in clt.redirected_funcids:
                support.jitCopy(clt.compiled_funcid, dstid)

    def _genop_set_frame_next_call(self, framevar, funcid, label):
        # High 24 bits give the function id.
        # Low 8 bits give the target label within that function.
        if SANITYCHECK:
            if isinstance(funcid, ConstInt):
                assert funcid.getint() < 2**24
            if isinstance(label, ConstInt):
                assert label.getint() < 0xFF
        next_call = js.Or(js.LShift(funcid, js.ConstInt(8)), label)
        addr = js.FrameForceDescrAddr(framevar)
        self.bldr.emit_store(next_call, addr, js.UInt32)

    def _genop_get_frame_next_call(self, callvar, framevar):
        addr = js.FrameForceDescrAddr(framevar)
        self.bldr.emit_load(callvar, addr, js.UInt32)


class CompiledLoopTokenASMJS(CompiledLoopToken):
    """CompiledLoopToken with extra fields for asmjs backend."""

    def __init__(self, cpu, number):
        CompiledLoopToken.__init__(self, cpu, number)
        self.compiled_funcid = 0
        self.compiled_blocks = []
        self.redirected_funcids = None
        self.redirected_to = 0
        self.invalidation = lltype.malloc(INVALIDATION_COUNTER, flavor="raw")
        self.invalidation.counter = 0
        self.inlined_gcrefs = []
        frame_info = lltype.malloc(jitframe.JITFRAMEINFO, flavor="raw")
        self.frame_info = rffi.cast(jitframe.JITFRAMEINFOPTR, frame_info)
        self.frame_info.clear()
        baseofs = self.cpu.get_baseofs_of_frame_field()
        self.frame_info.update_frame_depth(baseofs, 0)

    def ensure_frame_depth(self, required_offset):
        if SANITYCHECK:
            assert required_offset >= 0
            assert required_offset % WORD == 0
        required_depth = intmask(r_uint(required_offset // WORD))
        baseofs = self.cpu.get_baseofs_of_frame_field()
        self.frame_info.update_frame_depth(baseofs, required_depth)

    def __del__(self):
        CompiledLoopToken.__del__(self)
        lltype.free(self.invalidation, flavor="raw")
        lltype.free(self.frame_info, flavor="raw")
        support.jitFree(self.compiled_funcid)


class CompiledBlockASMJS(object):

    def __init__(self, assembler, clt, label, intoken, inputargs, operations,
                 outtoken, outputargs):
        self.assembler = assembler
        self.cpu = assembler.cpu
        self.clt = clt
        self.label = label
        self.operations = operations
        self.intoken = intoken
        self.inputargs = inputargs
        self.outtoken = outtoken
        self.outputargs = outputargs

        # Remember value of invalidation counter when this loop was created.
        # If it goes above this value, then GUARD_NOT_INVALIDATED fails.
        self.initial_invalidation_counter = clt.invalidation.counter

        # Calculate the locations at which our input args will appear
        # on the frame.  In the process, count how many variables of
        # each time we will need when loading them.  Also track which
        # are refs so that we can build a gcmap.
        self.inputlocs = [-1] * len(inputargs)
        self.num_int_args = 0
        self.num_double_args = 0
        reflocs = []
        offset = 0
        for i in xrange(len(inputargs)):
            box = inputargs[i]
            typ = js.HeapType.from_box(box)
            alignment = offset % typ.size
            if alignment:
                offset += typ.size - alignment
            self.inputlocs[i] = offset
            if box:
                if box.type == FLOAT:
                    self.num_double_args += 1
                elif box.type == REF:
                    self.num_int_args += 1
                    reflocs.append(offset)
                else:
                    self.num_int_args += 1
            offset += typ.size
        self.clt.ensure_frame_depth(offset)

        # Calculate a gcmap corresponding to the initial layout of the frame.
        # This will be needed if we ever need to enlarge the frame.
        self.initial_gcmap = gcmap = self._allocate_gcmap(offset)
        for pos in reflocs:
            pos = r_uint(pos // WORD)
            gcmap[pos // WORD // 8] |= r_uint(1) << (pos % (WORD * 8))

        # The first time we generate code for this block, we will create
        # gcmaps and store them in the following list.  During subsequent
        # generations we'll re-use each map from the list in turn.  This
        # works because re-compiling the code can't change ref layout.
        self.compiled_gcmaps = []
        self.next_compiled_gcmap = 0
        self.has_been_compiled = False

        # Ensure the block ends with an explicit jump.
        # This simplifies calculation of box longevity below.
        if operations and operations[-1].getopnum()  == rop.JUMP:
            self.outtoken = operations[-1].getdescr()
            self.outputargs = operations[-1].getarglist()
        else:
            jump = ResOperation(rop.JUMP, outputargs, None, descr=outtoken)
            operations.append(jump)

        # Calculate the longevity of all internal boxes.
        # This will let us re-use js variables for multiple boxes.
        self.longevity, _ = compute_vars_longevity(inputargs, operations)

        # Remove the final jump.
        # It doesn't hold its descr after the initial compile, so we
        # have to stash all its info on the block anyway.
        operations.pop()

    def __del__(self):
        lltype.free(self.initial_gcmap, flavor="raw")
        for gcmap in self.compiled_gcmaps:
            lltype.free(gcmap, flavor="raw")

    def _allocate_gcmap(self, offset):
        frame_size = r_uint(offset // WORD)
        gcmap_size = (frame_size // WORD // 8) + 1
        rawgcmap = lltype.malloc(jitframe.GCMAP, gcmap_size, flavor="raw")
        gcmap = rffi.cast(lltype.Ptr(jitframe.GCMAP), rawgcmap)
        for i in xrange(gcmap_size):
            gcmap[i] = r_uint(0)
        return gcmap

    def setup(self):
        self.bldr = self.assembler.bldr
        self.spilled_frame_locations = {}
        self.spilled_frame_values = {}
        self.spilled_frame_offset = 0
        self.box_to_jsval = {}
        self.next_compiled_gcmap = 0

    def teardown(self):
        self.bldr = None
        self.spilled_frame_locations = None
        self.spilled_frame_values = None
        self.spilled_frame_offset = 0
        self.box_to_jsval = None
        self.next_compiled_gcmap = 0
        if not self.has_been_compiled:
            self.has_been_compiled = True

    def _genop_load_input_args(self):
        self.bldr.emit_comment("LOAD INPUT ARGS FOR %d" % (self.label,))
        num_int_args = 0
        num_double_args = 0
        for i in xrange(len(self.inputargs)):
            box = self.inputargs[i]
            if not box:
                continue
            if box.type == FLOAT:
                jsval = self.assembler.input_vars_double[num_double_args]
                num_double_args += 1
            else:
                jsval = self.assembler.input_vars_int[num_int_args]
                num_int_args += 1
            pos = self.inputlocs[i]
            typ = js.HeapType.from_box(box)
            self.bldr.emit_load(jsval, js.FrameSlotAddr(pos), typ)
            # XXX TODO: this is a hack to trick tests into passing.
            # Necessary because we have no JITFRAME_FIXED_SIZE.
            # But I don't think it should be needed in real life?
            self.bldr.emit_store(js.zero, js.FrameSlotAddr(pos), typ)
            self.box_to_jsval[box] = jsval
        if SANITYCHECK:
            assert num_int_args == self.num_int_args
            assert num_double_args == self.num_double_args

    def _genop_block_body(self):
        #self.bldr.emit_debug("ENTER BLOCK %d" % (self.label,))
        #self.bldr.emit_debug("  ARGS:", [self._get_jsval(a) for a in self.inputargs])
        # Walk the list of operations, emitting code for each.
        # We expend some modest effort to generate "nice" javascript code,
        # by e.g. folding constant expressions and eliminating temp variables.
        self.pos = 0
        while self.pos < len(self.operations):
            op = self.operations[self.pos]
            step = 1
            # Is it one of the special test-only opcodes?
            if not we_are_translated() and op.getopnum() == -124:
                self.genop_force_spill(op)
            # Can we omit the operation completely?
            elif op.has_no_side_effect() and op.result not in self.longevity:
                self.bldr.emit_comment("OMMITTED USELESS JIT OP: %s" % (op,))
            # Do we need to emit it in conjunction with a guard?
            elif self._op_needs_guard(op):
                if SANITYCHECK:
                    assert self.pos + 1 < len(self.operations)
                step = 2
                guardop = self.operations[self.pos + 1]
                if SANITYCHECK:
                    assert guardop.is_guard()
                self.bldr.emit_comment("BEGIN JIT OP: %s" % (op,))
                self.bldr.emit_comment("  WITH GUARD OP: %s" % (guardop,))
                genop_withguard_list[op.getopnum()](self, op, guardop)
            # Do we need to write complex code for it?
            elif not self._op_is_simple_expr(op):
                self.bldr.emit_comment("BEGIN JIT OP: %s" % (op,))
                genop_list[op.getopnum()](self, op)
            # It's just a simple expression.
            # Maybe we can fold it into the next op?
            else:
                self.bldr.emit_comment("BEGIN JIT EXPR OP: %s" % (op,))
                expr = genop_expr_list[op.getopnum()](self, op)
                # XXX TODO: this causes test_caching_setfield to fail.
                if False and self._is_final_use(op.result, self.pos + 1):
                    if SANITYCHECK:
                        assert op.result not in self.box_to_jsval
                    self.box_to_jsval[op.result] = expr
                    self.bldr.emit_comment("FOLDED JIT EXPR OP")
                else:
                    boxvar = self._get_jsval(op.result)
                    self.bldr.emit_assignment(boxvar, expr)
            # Free vars for boxes that are no longer needed.
            # XXX TODO: need to free boxes from the guard, if any.
            for j in range(op.numargs()):
                self._maybe_free_boxvar(op.getarg(j))
            self._maybe_free_boxvar(op.result)
            self.pos += step
        # Generate the final jump, if any.
        # For jumps to local loop, we can send arguments via boxes.
        # For jumps to a different function, we spill to the frame.
        jumpdescr = self.outtoken
        if jumpdescr is None:
            pass
        else:
            assert isinstance(jumpdescr, TargetToken)
            if jumpdescr._asmjs_funcid == self.clt.compiled_funcid:
                comment = "JUMP LOCAL [%d]"
                comment = comment % (jumpdescr._asmjs_label,)
                self.bldr.emit_comment(comment)
                self._genop_local_jump(jumpdescr._asmjs_label, self.outputargs)
            else:
                comment = "JUMP TO ANOTHER LOOP [%d %d]"
                comment = comment % (jumpdescr._asmjs_funcid, jumpdescr._asmjs_label)
                self.bldr.emit_comment(comment)
                if SANITYCHECK:
                    target_clt = jumpdescr._asmjs_clt
                    target_blk = target_clt.compiled_blocks[jumpdescr._asmjs_label]
                    assert len(self.outputargs) == len(target_blk.inputargs)
                self._genop_write_output_args(self.outputargs)
                self._genop_set_frame_next_call(js.frame,
                                          js.ConstInt(jumpdescr._asmjs_funcid),
                                          js.ConstInt(jumpdescr._asmjs_label))
                self.bldr.emit_exit()
        # It's now safe to free all but the input vars
        # so they can be re-used by other blocks.
        for box, jsval in self.box_to_jsval.iteritems():
            if box not in self.inputargs and jsval != js.frame:
                if isinstance(jsval, js.DoubleVar):
                    self.bldr.free_doublevar(jsval)
                elif isinstance(jsval, js.IntVar):
                    self.bldr.free_intvar(jsval)
        # Safety net to prevent infinite loop.
        # Real code should never reach here, but tests generate some
        # partial traces that don't have a proper FINISH or JUMP at end.
        if SANITYCHECK:
            self.bldr.emit_exit()

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

    def _maybe_free_boxvar(self, box):
        if isinstance(box, Box):
            if self._is_final_use(box, self.pos) or box not in self.longevity:
                if SANITYCHECK:
                    assert box not in self.outputargs
                # Here we are happy to pop inputarg boxes from the dict,
                # as it means we don't have to keep them alive.  But we
                # must not free the underlying js variables.
                boxvar = self.box_to_jsval.pop(box, None)
                if box not in self.inputargs:
                    if boxvar is not None and boxvar != js.frame:
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

    def _genop_set_frame_next_call(self, frame, funcid, label):
        return self.assembler._genop_set_frame_next_call(frame, funcid, label)

    def _genop_get_frame_next_call(self, resvar, frame):
        return self.assembler._genop_get_frame_next_call(resvar, frame)

    def _genop_write_output_args(self, outputargs, locations=None):
        if locations is None:
            locations = self._get_frame_locations(outputargs)
        assert len(outputargs) == len(locations)
        with ctx_spill_to_frame(self):
            for i in xrange(len(outputargs)):
                box = outputargs[i]
                offset = locations[i]
                curval = self.spilled_frame_values.get(offset, None)
                if curval is not None:
                    if SANITYCHECK:
                        assert curval == box
                else:
                    self._genop_spill_to_frame(box, offset)
            self._genop_store_gcmap()
        return locations

    def _get_frame_locations(self, arguments, offset=-1):
        """Allocate locations in the frame for all the given arguments."""
        locations = [-1] * len(arguments)
        if offset < 0:
            offset = self.spilled_frame_offset
        for i in xrange(len(arguments)):
            box = arguments[i]
            typ = js.HeapType.from_box(box)
            alignment = offset % typ.size
            if alignment:
                offset += typ.size - alignment
            locations[i] = offset
            offset += typ.size
        self.clt.ensure_frame_depth(offset)
        return locations

    def _genop_spill_to_frame(self, box, offset=-1):
        typ = js.HeapType.from_box(box)
        # Allocate it a position at the next available offset.
        # Align each value to a multiple of its size.
        if offset == -1:
            offset = self.spilled_frame_offset
            alignment = offset % typ.size
            if alignment:
                offset += typ.size - alignment
        if offset + typ.size > self.spilled_frame_offset:
            self.spilled_frame_offset = offset + typ.size
        # Generate code to write the value into the frame.
        addr = js.FrameSlotAddr(offset)
        if isinstance(box, Box):
            boxexpr = self._genop_realize_box(box)
        else:
            boxexpr = self._get_jsval(box)
        self.bldr.emit_store(boxexpr, addr, typ)
        # Record where we spilled it.
        if box not in self.spilled_frame_locations:
            self.spilled_frame_locations[box] = []
        self.spilled_frame_locations[box].append(offset)
        self.spilled_frame_values[offset] = box
        return offset

    #
    #  Code-Generating dispatch methods.
    #  There's a method here for every resop we support.
    #  They are built into dispatch tables by code at the end of this file.
    #

    def genop_label(self, op):
        pass

    def genop_expr_strgetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        assert itemsize == 1
        itemoffset = js.Plus(js.ConstInt(basesize), offset)
        return js.HeapData(js.UInt8, js.Plus(base, itemoffset))

    def genop_strsetitem(self, op):
        base = self._get_jsval(op.getarg(0))
        offset = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        assert itemsize == 1
        itemoffset = js.Plus(js.ConstInt(basesize), offset)
        self.bldr.emit_store(value, js.Plus(base, itemoffset), js.UInt8)

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

    def genop_raw_load(self, op):
        itemsize, offset, signed = unpack_arraydescr(op.getdescr())
        base = self._get_jsval(op.getarg(0))
        which = self._get_jsval(op.getarg(1))
        addr = js.Plus(base, js.Plus(js.ConstInt(offset), which))
        typ = js.HeapType.from_size_and_sign(itemsize, signed)
        self.bldr.emit_load(self._get_jsval(op.result), addr, typ)

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

    def genop_raw_store(self, op):
        itemsize, offset, signed = unpack_arraydescr(op.getdescr())
        base = self._get_jsval(op.getarg(0))
        where = self._get_jsval(op.getarg(1))
        value = self._get_jsval(op.getarg(2))
        itemoffset = js.Plus(js.ConstInt(offset), where)
        addr = js.Plus(base, itemoffset)
        typ = js.HeapType.from_size_and_sign(itemsize, signed)
        self.bldr.emit_store(value, addr, typ)

    def _genop_expr_int_unaryop(operator):
        def genop_expr_int_unaryop(self, op):
            return operator(self._get_jsval(op.getarg(0)))
        return genop_expr_int_unaryop

    genop_expr_int_is_zero = _genop_expr_int_unaryop(js.UNot)
    genop_expr_int_neg = _genop_expr_int_unaryop(js.UMinus)
    genop_expr_int_invert = _genop_expr_int_unaryop(js.UNeg)

    def genop_expr_int_is_true(self, op):
        return js.UNot(js.UNot(self._get_jsval(op.getarg(0))))

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
        # XXX TODO: we don't have longlongs, but tests require this.
        os.write(2, "WARNING: genop_expr_convert_float_bytes_to_longlong\n")
        return self._get_jsval(op.getarg(0))

    def genop_expr_convert_longlong_bytes_to_float(self, op):
        # XXX TODO: we don't have longlongs, but tests require this.
        os.write(2, "WARNING: genop_expr_convert_longlong_bytes_to_float\n")
        return self._get_jsval(op.getarg(0))

    def genop_expr_cast_float_to_int(self, op):
        return js.SignedCast(self._get_jsval(op.getarg(0)))

    def genop_expr_cast_int_to_float(self, op):
        return js.DoubleCast(self._get_jsval(op.getarg(0)))

    def genop_read_timestamp(self, op):
        # Simulate processor time using gettimeofday().
        # XXX TODO: Probably this is all sorts of technically incorrect.
        # It needs to write into the heap, so we use the frame as scratch.
        os.write(2, "WARNING: genop_read_timestamp probably doesn't work\n")
        self.clt.ensure_frame_depth(2*WORD)
        addr = js.FrameSlotAddr(0)
        self.bldr.emit_expr(js.CallFunc("gettimeofday", [addr]))
        secs = js.HeapData(js.Int32, addr)
        micros = js.HeapData(js.Int32, js.FrameSlotAddr(WORD))
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
        # See if we can special-case this call with a builtin.
        # XXX TODO: wtf is "oop" anyway?
        effectinfo = descr.get_extra_info()
        oopspecindex = effectinfo.oopspecindex
        if oopspecindex == EffectInfo.OS_MATH_SQRT:
            self._genop_math_sqrt(op)
        else:
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
            self._genop_propagate_exception()

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
        release_addr = js.ConstInt(self.assembler.release_gil_addr)
        reacquire_addr = js.ConstInt(self.assembler.reacquire_gil_addr)
        with ctx_guard_not_forced(self, guardop):
            with ctx_allow_gc(self):
                self.bldr.emit_expr(js.DynCallFunc("v", release_addr, []))
                self._genop_call(op, descr, addr, args)
                self.bldr.emit_expr(js.DynCallFunc("v", reacquire_addr, []))

    def genop_withguard_call_assembler(self, op, guardop):
        cpu = self.cpu
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
            # XXX TODO: does it push a gcmap? does it need to?
            exeaddr = self.assembler.execute_trampoline_addr
            target_clt = descr.compiled_loop_token
            assert isinstance(target_clt, CompiledLoopTokenASMJS)
            funcid = js.ConstInt(target_clt.compiled_funcid)
            self._genop_set_frame_next_call(frame, funcid, js.zero)
            resvar = self.bldr.allocate_intvar()
            with ctx_allow_gc(self):
                call = js.DynCallFunc("ii", js.ConstInt(exeaddr), [frame])
                self.bldr.emit_assignment(resvar, call)
            # Load the descr resulting from that call.
            addr = js.FrameDescrAddr(resvar)
            resdescr = js.HeapData(js.Int32, addr)
            # Check if it's equal to done-with-this-frame.
            # The particular brand of DWTF depends on the result type.
            if op.result is None:
                dwtf = cpu.done_with_this_frame_descr_void
            else:
                kind = op.result.type
                if kind == INT:
                    dwtf = cpu.done_with_this_frame_descr_int
                elif kind == REF:
                    dwtf = cpu.done_with_this_frame_descr_ref
                elif kind == FLOAT:
                    dwtf = cpu.done_with_this_frame_descr_float
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
                    descr = cpu.getarraydescr_for_frame(kind)
                    offset = cpu.unpack_arraydescr(descr)
                    addr = js.Plus(resvar, js.ConstInt(offset))
                    typ = js.HeapType.from_kind(kind)
                    self.bldr.emit_load(self._get_jsval(op.result), addr, typ)
            # If not, then we need to invoke a helper function.
            with self.bldr.emit_else_block():
                if op.result is None:
                    callsig = "vii"
                elif op.result.type == FLOAT:
                    callsig = "dii"
                else:
                    callsig = "iii"
                args = [resvar, virtref]
                helpaddr = cpu.cast_adr_to_int(jd.assembler_helper_adr)
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
        sigmap = {"i": "i", "r": "i", "f": "d", "v": "v"}
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
                if descr.get_result_type() == "i":
                    # Trim the result if it's a less-than-full-sized integer,
                    result_size = descr.get_result_size()
                    if result_size < WORD:
                        nbits = 8 * result_size - 1
                        mask = js.ConstInt((2 << nbits) - 1)
                        call = js.And(call, mask)
                    # Cast result to appropriate signedness.
                    if descr.is_result_signed():
                        call = js.SignedCast(call)
                    else:
                        call = js.UnsignedCast(call)
                self.bldr.emit_comment("CALL:")
                self.bldr.emit_comment("  ARGS: %s" % (descr.arg_classes,))
                self.bldr.emit_comment("  RES: %s" % (descr.get_result_type(),))
                self.bldr.emit_comment("  SZ: %s" % (descr.get_result_size(),))
                self.bldr.emit_comment("  SGN: %s" % (descr.is_result_signed(),))
                self.bldr.emit_assignment(self._get_jsval(op.result), call)

    def _genop_math_sqrt(self, op):
        assert op.numargs() == 2
        arg = self._get_jsval(op.getarg(1))
        res = self._get_jsval(op.result)
        self.bldr.emit_assignment(res, js.CallFunc("sqrt", [arg]))

    def genop_force_token(self, op):
        if op.result is not None:
            self.box_to_jsval[op.result] = js.frame

    def _genop_local_jump(self, label, inputargs):
        target_block = self.clt.compiled_blocks[label]
        assert len(target_block.inputargs) == len(inputargs)
        # We're going to use a bunch of temporary variables to swap the
        # the new values of the variables into the old, so that we don't
        # mess with the state of any boxes that we might need for a future
        # assignment.
        tempvars = [None] * len(target_block.inputargs)
        outvars = [None] * len(target_block.inputargs)
        num_int_vars = 0
        num_double_vars = 0
        for i in range(len(inputargs)):
            box = inputargs[i]
            if box.type == FLOAT:
                tempvars[i] = self.bldr.allocate_doublevar()
                outvars[i] = self.assembler.input_vars_double[num_double_vars]
                num_double_vars += 1
            else:
                tempvars[i] = self.bldr.allocate_intvar()
                outvars[i] = self.assembler.input_vars_int[num_int_vars]
                num_int_vars += 1
            boxvar = self._get_jsval(box)
            self.bldr.emit_assignment(tempvars[i], boxvar)
        if SANITYCHECK:
            assert num_int_vars == target_block.num_int_args
            assert num_double_vars == target_block.num_double_args
        for i in range(len(inputargs)):
            box = inputargs[i]
            self.bldr.emit_assignment(outvars[i], tempvars[i])
            if box.type == FLOAT:
                self.bldr.free_doublevar(tempvars[i])
            else:
                self.bldr.free_intvar(tempvars[i])
        if label != self.label:
            self.bldr.emit_assignment(self.assembler.label_var,
                                      js.ConstInt(label))
        self.bldr.emit_continue_loop()

    def genop_finish(self, op):
        descr = op.getdescr()
        descr = cast_instance_to_gcref(descr)
        rgc._make_sure_does_not_move(descr)
        # Write return value into the frame.
        self._genop_write_output_args(op.getarglist())
        # Write the descr into the frame slot.
        addr = js.FrameDescrAddr()
        self.bldr.emit_store(js.ConstPtr(descr), addr, js.Int32)
        self._genop_set_frame_next_call(js.frame, js.zero, js.zero)
        self.bldr.emit_exit()

    #
    # Guard-related things.
    #

    def genop_guard_true(self, op):
        test = js.UNot(self._get_jsval(op.getarg(0)))
        self._genop_guard_failure(test, op)

    def genop_guard_isnull(self, op):
        test = js.NotEqual(self._get_jsval(op.getarg(0)), js.zero)
        self._genop_guard_failure(test, op)

    def genop_guard_nonnull(self, op):
        test = js.UNot(self._get_jsval(op.getarg(0)))
        self._genop_guard_failure(test, op)

    def genop_guard_false(self, op):
        test = self._get_jsval(op.getarg(0))
        self._genop_guard_failure(test, op)

    def genop_guard_value(self, op):
        test = js.NotEqual(self._get_jsval(op.getarg(0)),
                           self._get_jsval(op.getarg(1)))
        self._genop_guard_failure(test, op)

    def genop_guard_class(self, op):
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
        pos_exctyp = js.ConstInt(self.cpu.pos_exception())
        exctyp = js.HeapData(js.Int32, pos_exctyp)
        test = js.NotEqual(exctyp, js.zero)
        self._genop_guard_failure(test, op)

    def genop_guard_not_invalidated(self, op):
        translate_support_code = self.cpu.translate_support_code
        offset, size = symbolic.get_field_token(INVALIDATION_COUNTER,
                                                "counter",
                                                translate_support_code)
        assert size == js.Int32.size
        invalidation = rffi.cast(lltype.Signed, self.clt.invalidation)
        cur_val = js.HeapData(js.Int32, js.Plus(js.ConstInt(invalidation),
                                                js.ConstInt(offset)))
        orig_val = js.ConstInt(self.initial_invalidation_counter)
        test = js.NotEqual(cur_val, orig_val)
        self._genop_guard_failure(test, op)

    def _genop_guard_failure(self, test, op, faillocs=None):
        descr = op.getdescr()
        rgc._make_sure_does_not_move(descr)
        assert isinstance(descr, AbstractFailDescr)
        assert descr._asmjs_funcid == self.clt.compiled_funcid
        failargs = op.getfailargs()
        with self.bldr.emit_if_block(test):
            #self.bldr.emit_debug("GUARD FAILED %s" % (op,))
            #self.bldr.emit_debug("  ARGS:", [self._get_jsval(a) for a in op.getarglist()])
            #self.bldr.emit_debug("  FAILARGS:", [self._get_jsval(a) for a in failargs])
            # If the guard has been compiled into a bridge, jump to that label.
            # If not, then spill to the frame and exit.
            if descr._asmjs_glbl.label != 0:
                if SANITYCHECK:
                    assert self.has_been_compiled
                outputargs = [arg for arg in failargs if arg is not None]
                self._genop_local_jump(descr._asmjs_glbl.label, outputargs)
                # During initial compilation we would have pushed a gcmap here.
                # We don't need it now, so skip over it.  Note that we can't
                # free it in case older versions of the loop are still running.
                self.next_compiled_gcmap += 1
            else:
                # Tricky case: we may have compiled a bridge for this guard
                # while the code was executing.  We need to jump into the
                # newly-compiled code at the appropriate label.
                translate_support_code = self.cpu.translate_support_code
                offset, size = symbolic.get_field_token(GUARD_LABEL,
                                                        "label",
                                                        translate_support_code)
                assert size == js.Int32.size
                glbl = rffi.cast(lltype.Signed, descr._asmjs_glbl)
                cur_label_adr = js.Plus(js.ConstInt(glbl), js.ConstInt(offset))
                cur_label = js.HeapData(js.Int32, cur_label_adr)
                with self.bldr.emit_if_block(js.NotEqual(cur_label, js.zero)):
                    self.bldr.emit_comment("INVOKING NEWLY-COMPILED BRIDGE")
                    # Spill jump args to the frame as inputargs for guard.
                    outputargs = [arg for arg in failargs if arg is not None]
                    self._genop_write_output_args(outputargs)
                    # Directly re-invoke the new version of this function.
                    # We can't use the trampoline here, as there may be
                    # an active exception that the guard must capture.
                    this_funcid = js.ConstInt(self.clt.compiled_funcid)
                    self._genop_set_frame_next_call(js.frame,
                                                    this_funcid,
                                                    cur_label)
                    call = js.CallFunc("jitInvoke", [this_funcid, js.frame])
                    self.bldr.emit_assignment(js.frame, call)
                    self.bldr.emit_exit()
                # return _jitInvoke(this-func-id, newly-compiled-label, frame)
                # If there might be an exception, capture it to the frame.
                if self._guard_might_have_exception(op):
                    cpu = self.cpu
                    pos_exctyp = js.ConstInt(cpu.pos_exception())
                    pos_excval = js.ConstInt(cpu.pos_exc_value())
                    exctyp = js.HeapData(js.Int32, pos_exctyp)
                    excval = js.HeapData(js.Int32, pos_excval)
                    with self.bldr.emit_if_block(exctyp):
                        addr = js.FrameGuardExcAddr()
                        self.bldr.emit_store(excval, addr, js.Int32)
                        self.bldr.emit_store(js.zero, pos_exctyp, js.Int32)
                        self.bldr.emit_store(js.zero, pos_excval, js.Int32)
                # Store the failargs into the frame.
                self.bldr.emit_comment("SPILL %d FAILARGS" % (len(failargs),))
                if faillocs is None:
                    faillocs = self._get_frame_locations(failargs)
                self._genop_write_output_args(failargs, faillocs)
                descr._asmjs_faillocs = faillocs
                # Write the fail_descr into the frame.
                fail_descr = cast_instance_to_gcref(descr)
                addr = js.FrameDescrAddr()
                self.bldr.emit_store(js.ConstPtr(fail_descr), addr, js.Int32)
                # Bail back to the interpreter to deal with it.
                self._genop_set_frame_next_call(js.frame, js.zero, js.zero)
                self.bldr.emit_exit()

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
        pos_exctyp = js.ConstInt(self.cpu.pos_exception())
        exctyp = js.HeapData(js.Int32, pos_exctyp)
        with self.bldr.emit_if_block(exctyp):
            self._genop_propagate_exception()

    def _genop_propagate_exception(self):
        cpu = self.cpu
        if not cpu.propagate_exception_descr:
            return
        pos_exctyp = js.ConstInt(cpu.pos_exception())
        pos_excval = js.ConstInt(cpu.pos_exc_value())
        excval = js.HeapData(js.Int32, pos_excval)
        # Store the exception on the frame, and clear it.
        self.bldr.emit_store(excval, js.FrameGuardExcAddr(), js.Int32)
        self.bldr.emit_store(js.zero, pos_exctyp, js.Int32)
        self.bldr.emit_store(js.zero, pos_excval, js.Int32)
        # Store the special propagate-exception descr on the frame.
        descr = cast_instance_to_gcref(cpu.propagate_exception_descr)
        addr = js.FrameDescrAddr()
        self.bldr.emit_store(js.ConstPtr(descr), addr, js.Int32)
        # Bail back to the invoking code to deal with it.
        self._genop_store_gcmap()
        self._genop_set_frame_next_call(js.frame, js.zero, js.zero)
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
            # The tests sometimes require that we pass along the jitfame.
            mallocfn = rffi.cast(lltype.Signed, self.assembler.gc_malloc_nursery_addr)
            if hasattr(gc_ll_descr, 'passes_frame'):
                callsig = "iii"
                args = [sizevar, js.frame]
            else:
                callsig = "ii"
                args = [sizevar]
            with ctx_allow_gc(self, exclude=[op.result]):
                call = js.DynCallFunc(callsig, js.ConstInt(mallocfn), args)
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
        lengthvar = self._genop_realize_box(lengthbox)
        # Figure out the total size to be allocated.
        # It's gcheader + basesize + length*itemsize, maybe with some padding.
        if hasattr(gc_ll_descr, 'gcheaderbuilder'):
            size_of_header = gc_ll_descr.gcheaderbuilder.size_gc_header
        else:
            size_of_header = WORD
        constsize = size_of_header + arraydescr.basesize
        calc_totalsize = js.Plus(js.ConstInt(constsize),
                                 js.IMul(lengthvar, js.ConstInt(itemsize)))
        totalsize = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(totalsize, calc_totalsize)
        # Round up the total size to a whole multiple of wordize.
        if itemsize % WORD != 0:
            padsize = self.bldr.allocate_intvar()
            self.bldr.emit_assignment(padsize,
                                      js.Mod(totalsize, js.word))
            with self.bldr.emit_if_block(js.NotEqual(padsize, js.zero)):
                self.bldr.emit_assignment(totalsize,
                                          js.Plus(totalsize,
                                                  js.Minus(js.word, padsize)))
            self.bldr.free_intvar(padsize)
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
        chk_not_overflowed = js.And(js.LessThanEq(new_nfree, ntop),
                                    js.LessThan(totalsize, maxsize))
        with self.bldr.emit_if_block(chk_not_overflowed):
            # If we fit in the nursery, we're all good!
            # Increment nursery_free and set type flags on the object.
            self.bldr.emit_store(new_nfree, nfree_addr, js.Int32)
            self.bldr.emit_store(js.ConstInt(arraydescr.tid), resvar, js.Int32)
        with self.bldr.emit_else_block():
            # If it didn't fit in the nursery, we have to call out to malloc.
            if kind == rewrite.FLAG_ARRAY:
                args = [js.ConstInt(itemsize),
                        js.ConstInt(arraydescr.tid),
                        lengthvar]
                callsig = "iiii"
                mallocfn = self.assembler.gc_malloc_array_addr
            else:
                args = [lengthvar]
                callsig = "ii"
                if kind == rewrite.FLAG_STR:
                    mallocfn = self.assembler.gc_malloc_str_addr
                else:
                    assert kind == rewrite.FLAG_UNICODE
                    mallocfn = self.assembler.gc_malloc_unicode_addr
            mallocfn = js.ConstInt(rffi.cast(lltype.Signed, mallocfn))
            with ctx_allow_gc(self, exclude=[op.result]):
                call = js.DynCallFunc(callsig, mallocfn, args)
                self.bldr.emit_assignment(resvar, call)
            self._genop_check_and_propagate_exception()
        # That's it!  Cleanup temp variables.
        self.bldr.free_intvar(new_nfree)
        self.bldr.free_intvar(totalsize)

    def genop_cond_call_gc_wb(self, op):
        assert op.result is None
        self._genop_write_barrier(op.getarglist(), op.getdescr())

    def genop_cond_call_gc_wb_array(self, op):
        assert op.result is None
        self._genop_write_barrier(op.getarglist(), op.getdescr(), array=True)

    def _genop_write_barrier(self, arguments, wbdescr=None, array=False):
        # Decode and grab the necessary function pointer.
        # If it's zero, the GC doesn't need a write barrier here.
        cpu = self.cpu
        if wbdescr is None:
            wbdescr = cpu.gc_ll_descr.write_barrier_descr
        if wbdescr is None:
            return
        if we_are_translated():
            cls = cpu.gc_ll_descr.has_write_barrier_class()
            assert cls is not None and isinstance(wbdescr, cls)
        card_marking = False
        if array and wbdescr.jit_wb_cards_set != 0:
            assert (wbdescr.jit_wb_cards_set_byteofs ==
                    wbdescr.jit_wb_if_flag_byteofs)
            card_marking = True
        if not card_marking:
            wbfunc = wbdescr.get_write_barrier_fn(cpu)
        else:
            wbfunc = wbdescr.get_write_barrier_from_array_fn(cpu)
        wbfunc = rffi.cast(lltype.Signed, wbfunc)
        if wbfunc == 0:
            return
        self.bldr.emit_comment("WRITE BARRIER")
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
        flagaddrvar = self.bldr.allocate_intvar()
        self.bldr.emit_assignment(flagaddrvar, flagaddr)
        flagbyte = js.HeapData(js.Int8, flagaddrvar)
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
                    with ctx_temp_intvar(self, byte_addr) as byte_addr:
                        old_byte_data = js.HeapData(js.Int8, byte_addr)
                        new_byte_data = js.Or(old_byte_data, byte_mask)
                        self.bldr.emit_store(new_byte_data, byte_addr, js.Int8)
        self.bldr.free_intvar(flagbytevar)
        self.bldr.free_intvar(flagaddrvar)

    def _genop_store_gcmap(self, writebarrier=True):
        """Push a gcmap representing current spilled state of frame."""
        # If we've already compiled this code once, re-use existing gcmap.
        if self.has_been_compiled:
            gcmap = self.compiled_gcmaps[self.next_compiled_gcmap]
            self.next_compiled_gcmap += 1
        else:
            # Make a new gcmap sized to match current size of frame.
            gcmap = self._allocate_gcmap(self.spilled_frame_offset)
            # Set a bit for every REF that has been spilled.
            num_refs = 0
            for pos, box in self.spilled_frame_values.iteritems():
                if box and box.type == REF:
                    pos = r_uint(pos // WORD)
                    gcmap[pos // WORD // 8] |= r_uint(1) << (pos % (WORD * 8))
                    num_refs += 1
            self.compiled_gcmaps.append(gcmap)
        # Store the appropriate gcmap on the frame.
        comment = "STORE GCMAP"
        if SANITYCHECK:
            for i in xrange(len(gcmap)):
                comment = comment + " %d" % (gcmap[i],)
        self.bldr.emit_comment(comment)
        gcmapref = js.ConstInt(self.cpu.cast_ptr_to_int(gcmap))
        self.bldr.emit_store(gcmapref, js.FrameGCMapAddr(), js.Int32)
        # We might have just stored some young pointers into the frame.
        # Emit a write barrier just in case.
        if writebarrier:
            self._genop_write_barrier([js.frame])

    def genop_debug_merge_point(self, op):
        pass

    def genop_jit_debug(self, op):
        pass

    def genop_keepalive(self, op):
        pass

    def genop_force_spill(self, op):
        # This is used by tests.
        # The item will stay spilled to the frame forever.
        self._genop_spill_to_frame(op.getarg(0))

    def not_implemented_op_withguard(self, op, guardop):
        self.not_implemented_op(op)
        self.not_implemented_op(guardop)

    def not_implemented_op_expr(self, op):
        self.not_implemented_op(op)
        return js.zero

    def not_implemented_op(self, op):
        self._print_op(op)
        raise NotImplementedError

    def _print_op(self, op):
        print "OPERATION:", op
        for i in range(op.numargs()):
            print "  ARG:", op.getarg(i)
        print "  RES:", op.result


class ctx_spill_to_frame(object):

    def __init__(self, block):
        self.block = block
        self.orig_spilled_frame_offset = 0

    def _get_jsval(self, jitval):
        return self.block._get_jsval(jitval)

    def __enter__(self):
        # Remember the current offset, so we can pop all items
        # after this when we exit the context.
        self.orig_spilled_frame_offset = self.block.spilled_frame_offset
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        # Pop any items that were pushed in this context.
        orig_offset = self.orig_spilled_frame_offset
        self.block.clt.ensure_frame_depth(self.block.spilled_frame_offset)
        for pos, box in self.block.spilled_frame_values.items():
            if pos >= orig_offset:
                del self.block.spilled_frame_values[pos]
                self.block.spilled_frame_locations[box].remove(pos)
                if not self.block.spilled_frame_locations[box]:
                    del self.block.spilled_frame_locations[box]
        self.block.spilled_frame_offset = orig_offset

    def is_spilled(self, box):
        try:
            self.block.spilled_frame_locations[box]
        except KeyError:
            return False
        else:
            return True

    def genop_spill_to_frame(self, box, offset=-1):
        self.block._genop_spill_to_frame(box, offset)


class ctx_guard_not_forced(ctx_spill_to_frame):

    def __init__(self, block, guardop):
        ctx_spill_to_frame.__init__(self, block)
        self.faillocs = None
        self.guardop = guardop

    def __enter__(self):
        ctx_spill_to_frame.__enter__(self)
        bldr = self.block.bldr
        # Store the force-descr where forcing code can find it.
        descr = self.guardop.getdescr()
        faildescr = js.ConstPtr(cast_instance_to_gcref(descr))
        bldr.emit_store(faildescr, js.FrameForceDescrAddr(), js.Int32)
        # Write the potential failargs into the frame.
        # We have to spill them here because the forcing logic might
        # need to read them out to populate the virtualizable.
        # They must be spilled at their final output location.
        assert self.orig_spilled_frame_offset == 0
        failargs = self.guardop.getfailargs()
        self.faillocs = self.block._get_frame_locations(failargs)
        assert len(failargs) == len(self.faillocs)
        for i in xrange(len(failargs)):
            failarg = failargs[i]
            # Careful, some boxes may not have a value yet.
            if failarg and failarg not in self.block.box_to_jsval:
                continue
            self.genop_spill_to_frame(failarg, self.faillocs[i])
        # XXX TODO: this gets overwritten by the ctx_allow_gc
        # that follows the contained call() op.
        self.block._genop_store_gcmap()
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        # Emit the guard check, testing for whether jf_descr has been set.
        descr = js.HeapData(js.Int32, js.FrameDescrAddr())
        test = js.NotEqual(descr, js.zero)
        self.block._genop_guard_failure(test, self.guardop, self.faillocs)
        self.faillocs = None
        # It's now safe to pop from the frame as usual.
        ctx_spill_to_frame.__exit__(self, exc_typ, exc_val, exc_tb)


class ctx_allow_gc(ctx_spill_to_frame):

    def __init__(self, block, exclude=None):
        ctx_spill_to_frame.__init__(self, block)
        self.exclude = exclude

    def __enter__(self):
        ctx_spill_to_frame.__enter__(self)
        bldr = self.block.bldr
        # Spill any active REF boxes into the frame.
        for box, jsval in self._get_live_boxes_in_spill_order():
            if not box or box.type != REF:
                continue
            if self.exclude is not None and box in self.exclude:
                continue
            if jsval == js.frame:
                continue
            if self.block._is_final_use(box, self.block.pos):
                continue
            if not self.is_spilled(box):
                self.genop_spill_to_frame(box)
        self.block._genop_store_gcmap()
        # Push the jitframe itself onto the gc shadowstack.
        # We do the following:
        #   * get a pointer to the pointer to the root-stack top.
        #   * deref it to get the root-stack top, and write the frame there.
        #   * in-place increment root-stack top via its pointer.
        gcrootmap = self.block.assembler.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rstaddr = js.ConstInt(gcrootmap.get_root_stack_top_addr())
            rst = js.HeapData(js.Int32, rstaddr)
            bldr.emit_store(js.frame, rst, js.Int32)
            newrst = js.Plus(rst, js.word)
            bldr.emit_store(newrst, rstaddr, js.Int32)
        return self

    def __exit__(self, exc_typ, exc_val, exc_tb):
        bldr = self.block.bldr
        # Pop the jitframe from the root-stack.
        # This is an in-place decrement of root-stack top.
        gcrootmap = self.block.assembler.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rstaddr = js.ConstInt(gcrootmap.get_root_stack_top_addr())
            rst = js.HeapData(js.Int32, rstaddr)
            newrst = js.Minus(rst, js.word)
            bldr.emit_store(newrst, rstaddr, js.Int32)
            # For moving GCs, the address of the jitframe may have changed.
            # Read the possibly-updated address out of root-stack top.
            # NB: this instruction re-evaluates the HeapData expression in rst.
            bldr.emit_assignment(js.frame, js.HeapData(js.Int32, rst))
        # Similarly, read potential new addresss of any spilled boxes.
        # XXX TODO: don't double-load boxes that appear multiple times.
        for pos, box in self.block.spilled_frame_values.iteritems():
            if not box or box.type != REF:
                continue
            if self.exclude is not None and box in self.exclude:
                continue
            addr = js.FrameSlotAddr(pos)
            bldr.emit_load(self._get_jsval(box), addr, js.Int32)
        # It's now safe to pop from the frame as usual.
        ctx_spill_to_frame.__exit__(self, exc_typ, exc_val, exc_tb)

    def _get_live_boxes_in_spill_order(self):
        # Some tests expect boxes to be spilled in order of use.
        # We fake it by ordering them lexicographically by name.
        if we_are_translated():
            for item in self.block.box_to_jsval.iteritems():
                yield item
        else:
            items = list(self.block.box_to_jsval.iteritems())
            items.sort(key=lambda i: str(i[0]))
            for item in items:
                yield item


class ctx_preserve_exception(object):

    def __init__(self, assembler):
        self.assembler = assembler
        self.bldr = self.assembler.bldr
        self.pos_exctyp = js.ConstInt(assembler.cpu.pos_exception())
        self.pos_excval = js.ConstInt(assembler.cpu.pos_exc_value())
        self.var_exctyp = self.bldr.allocate_intvar()

    def __enter__(self):
        exctyp = js.HeapData(js.Int32, self.pos_exctyp)
        excval = js.HeapData(js.Int32, self.pos_excval)
        self.bldr.emit_assignment(self.var_exctyp, exctyp)
        with self.bldr.emit_if_block(self.var_exctyp):
            self.bldr.emit_store(excval, js.FrameGuardExcAddr(), js.Int32)
            self.bldr.emit_store(js.zero, self.pos_exctyp, js.Int32)
            self.bldr.emit_store(js.zero, self.pos_excval, js.Int32)

    def __exit__(self, exc_typ, exc_val, exc_tb):
        excval = js.HeapData(js.Int32, js.FrameGuardExcAddr())
        with self.bldr.emit_if_block(self.var_exctyp):
            self.bldr.emit_store(self.var_exctyp, self.pos_exctyp, js.Int32)
            self.bldr.emit_store(excval, self.pos_excval, js.Int32)
            self.bldr.emit_store(js.zero, js.FrameGuardExcAddr(), js.Int32)
        self.bldr.free_intvar(self.var_exctyp)


class ctx_temp_intvar(object):

    def __init__(self, block, expr=None):
        self.block = block
        self.bldr = block.bldr
        self.variable = None
        self.expr = expr

    def __enter__(self):
        self.variable = self.bldr.allocate_intvar()
        if self.expr is not None:
            self.bldr.emit_assignment(self.variable, self.expr)
        return self.variable

    def __exit__(self, exc_typ, exc_val, exc_tb):
        self.bldr.free_intvar(self.variable)


# Build a dispatch table mapping opnums to the method that emits code for them.
# There are two different method signatures, depending on whether we can work
# with a single opcode or require an opcode/guard pair.

genop_list = [CompiledBlockASMJS.not_implemented_op] * rop._LAST
genop_expr_list = [CompiledBlockASMJS.not_implemented_op_expr] * rop._LAST
genop_withguard_list = [CompiledBlockASMJS.not_implemented_op_withguard] * rop._LAST

for name, value in CompiledBlockASMJS.__dict__.iteritems():
    if name == "genop_force_spill":
        continue
    elif name.startswith('genop_withguard_'):
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
