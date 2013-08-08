
from rpython.jit.backend.llsupport import symbolic, jitframe
from rpython.jit.backend.llsupport.regalloc import compute_vars_longevity
from rpython.jit.backend.llsupport.descr import unpack_fielddescr
from rpython.rtyper.lltypesystem import lltype, rffi, rstr
from rpython.rtyper.annlowlevel import cast_instance_to_gcref
from rpython.jit.backend.model import CompiledLoopToken
from rpython.jit.metainterp.resoperation import rop
from rpython.jit.metainterp.history import (AbstractFailDescr, ConstInt,
                                            ConstPtr, Box, TargetToken)

from rpython.jit.backend.asmjs import support
from rpython.jit.backend.asmjs.jsbuilder import (ASMJSBuilder, IntBinOp,
                                                 HeapType, Int8, Int32,
                                                 JitFrameAddr_base,
                                                 JitFrameAddr_descr)


class AssemblerASMJS(object):
    """Class for assembling a Trace into a compiled ASMJS function."""

    def __init__(self, cpu):
        self.cpu = cpu

    def set_debug(self, v):
        return False

    def setup_once(self):
        # XXX TODO: we use a fixed, shared JITFRAMEINFO for now.
        # Just until we figure out exactly how we're going to use the frame.
        # We give it enough size to handle a realistic number of args.
        frame_info = lltype.malloc(jitframe.JITFRAMEINFO, flavor="raw")
        self.frame_info = rffi.cast(jitframe.JITFRAMEINFOPTR, frame_info)
        baseofs = self.cpu.get_baseofs_of_frame_field()
        self.frame_info.update_frame_depth(baseofs, 32)

    def finish_once(self):
        lltype.free(self.frame_info, flavor="raw")

    def setup(self, looptoken):
        self.current_clt = looptoken.compiled_loop_token
        self.jsbuilder = ASMJSBuilder(self.cpu)
        self.pending_target_tokens = []

    def teardown(self):
        self.output = None
        self.jsbuilder = None
        self.pending_target_tokens = None

    def assemble_loop(self, loopname, inputargs, operations, looptoken, log):
        """Assemble and compile a new loop function from the given trace.

        This method takes the recorded trace, generates asmjs source code from
        it, then invokes an external helper to compile that source into a new
        function object.  It attached a "function id" to the CompiledLoopToken
        which can be used to invoke the new function.
        """
        clt = CompiledLoopToken(self.cpu, looptoken.number)
        looptoken.compiled_loop_token = clt
        self.setup(looptoken)
        clt.frame_info = self.frame_info
        clt.allgcrefs = []
        operations = self._prepare_loop(inputargs, operations, looptoken,
                                        clt.allgcrefs)
        self._find_loop_entry_token(operations)
        self._allocate_variables(inputargs, operations)
        self._assemble(inputargs, operations)
        clt._compiled_function_id = self._compile()
        self._finalize_pending_target_tokens(clt._compiled_function_id)
        self.teardown()

    def assemble_bridge(self, faildescr, inputargs, operations,
                        original_loop_token, log):
        """Assemble, compile and link a new bridge from the given trace.

        NOT WORKING YET.  BELOW IS VAGUE PLANNED THEORY OF OPERATION.

        This method (will) take the recorded trace, generate asmjs source code
        from it, then invoke and external helper to compile it over the top
        of the existing function for the bridged guard.
        """
        assert isinstance(faildescr, AbstractFailDescr)
        self.setup(original_loop_token)
        operations = self._prepare_bridge(inputargs, operations,
                                          self.current_clt.allgcrefs,
                                          self.current_clt.frame_info)
        self._find_loop_entry_token(operations)
        self._allocate_variables(inputargs, operations)
        self._assemble(inputargs, operations)
        self._recompile(faildescr._asmjs_funcid)
        self._finalize_pending_target_tokens(faildescr._asmjs_funcid)
        self.teardown()

    def _prepare_loop(self, inputargs, operations, looptoken, allgcrefs):
        operations = self._prepare(inputargs, operations, allgcrefs)
        # Allocate frame locations for the input arguments.
        # They just go right upon the base offset of the frame.
        locs = []
        offset = 0
        for box in inputargs:
            assert isinstance(box, Box)
            typ = HeapType.from_box(box)
            # XXX TODO: alignment for multi-word inputs
            locs.append(offset)
            offset += typ.size
        # Store the list of locs on the token, so that the calling
        # code knows where to write them to.
        # XXX TODO: this is an llmodel idiom, feels kinda janky here.
        # We can probably simplify this somewhat.
        looptoken.compiled_loop_token._ll_initial_locs = locs
        return operations

    def _prepare_bridge(self, inputargs, operations, allgcrefs, frame_info):
        operations = self._prepare(inputargs, operations, allgcrefs)
        # XXX TODO: figure out what sort of re-writing will be required,
        # if any.  The x86 backend has a bunch of frame-remapping stuff here.
        return operations

    def _prepare(self, inputargs, operations, allgcrefs):
        cpu = self.cpu
        return cpu.gc_ll_descr.rewrite_assembler(cpu, operations, allgcrefs)

    def _find_loop_entry_token(self, operations):
        """Find the looptoken targetted by the final JUMP, if any.

        This method finds and marks the target of the backwards jump at the
        end of the loop.  It costs us an iteration over the ops list, but it
        allows the jsbuilder to more efficiently generate its looping structure
        by knowing this information up-front.
        """
        for i in range(len(operations) - 1):
            assert operations[i].getopnum() != rop.JUMP
        final_op = operations[-1]
        if final_op.getopnum() == rop.JUMP:
            descr = final_op.getdescr()
            assert isinstance(descr, TargetToken)
            self.jsbuilder.set_loop_entry_token(descr)

    def _allocate_variables(self, inputargs, operations):
        """Allocate an asmjs variable for every box in the trace.

        This method enumerates all the boxes in the trace, including input
        arguments, op arguments, and op return values.  It allocates each
        a variable in the generated code, using lifetime information to share
        variables between boxes that don't overlap.

        If you squint, it's kinda like a linear-scan register allocation
        with an unbounded number of registers...
        """
        longevity, _ = compute_vars_longevity(inputargs, operations)
        for box in inputargs:
            self.jsbuilder.allocate_variable(box)
        for i in range(len(operations)):
            op = operations[i]
            # Allocate variables for all the arg boxes, and the result box.
            for j in range(op.numargs()):
                box = op.getarg(j)
                if box is not None and isinstance(box, Box):
                    self.jsbuilder.allocate_variable(box)
            if op.result and isinstance(op.result, Box):
                self.jsbuilder.allocate_variable(op.result)
            # Free vars for boxes where this is their last use.
            for j in range(op.numargs()):
                box = op.getarg(j)
                if box is not None and isinstance(box, Box):
                    if longevity[box][1] <= i:
                        self.jsbuilder.free_variable(box)
            if op.result and isinstance(op.result, Box):
                if longevity[op.result][1] <= i:
                    self.jsbuilder.free_variable(op.result)

    def _assemble(self, inputargs, operations):
        """Generate the body of the asmjs function for the given trace."""
        # Load input arguments from frame.
        # XXX TODO: de-duplicate this iteration logic from _prepare_loop
        offset = 0
        for box in inputargs:
            assert isinstance(box, Box)
            addr = IntBinOp("+", JitFrameAddr_base(), ConstInt(offset))
            typ = HeapType.from_box(box)
            # XXX TODO: alignment for multi-word arguments
            self.jsbuilder.emit_load(box, addr, typ)
            offset += typ.size
        # Walk the list of operations, emitting code for each.
        # We do absolutely no optimizations on the trace, with the
        # expectation that the host JIT will do that for us.
        i = 0
        while i < len(operations):
            op = operations[i]
            genop_list[op.getopnum()](self, op)
            i += 1

    def _compile(self):
        """Compile the generated source into an invokable asmjs function.

        This method passes the generated source to an external javascript
        helper, which compiles it into a function and returns an opaque
        integer "function id" that can be used to invoke the code.
        """
        jssrc = self.jsbuilder.finish()
        return support.jitCompile(jssrc)

    def _recompile(self, function_id):
        """Compile generated source as replacement for an existing function.

        This method passes the generated source to an external javascript
        helper, which compiles it into a function and uses that to replace
        any existing function associated with the given function id.
        """
        jssrc = self.jsbuilder.finish()
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
    #  (Which frankly is not that many at this point in time!)
    #  These are built into a jump-table by code at the end of this file.
    #

    def genop_label(self, op):
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        # Attach the argument boxes to the descr, so that
        # jumps to this label know where to put their arguments.
        # XXX TODO: attach just the varnames, let the boxes be GC'd?
        inputargs = op.getarglist()
        argboxes = [None] * len(inputargs)
        for i in range(len(inputargs)):
            argbox = inputargs[i]
            assert isinstance(argbox, Box)
            argboxes[i] = argbox
        descr._asmjs_argboxes = argboxes
        descr._asmjs_label = self.jsbuilder.emit_token_label(descr)
        descr._asmjs_funcid = 0  # placeholder; this gets set after compilation
        self.pending_target_tokens.append(descr)

    def genop_strgetitem(self, op):
        base = op.getarg(0)
        offset = op.getarg(1)
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        assert itemsize == 1
        addr = IntBinOp("+", base, IntBinOp("+", ConstInt(basesize), offset))
        self.jsbuilder.emit_load(op.result, addr, Int8)

    def genop_strlen(self, op):
        base = op.getarg(0)
        arraytoken = symbolic.get_array_token(rstr.STR,
                                              self.cpu.translate_support_code)
        basesize, itemsize, len_offset = arraytoken
        addr = IntBinOp("+", base, ConstInt(len_offset))
        self.jsbuilder.emit_load(op.result, addr, Int32)

    def genop_getfield_gc(self, op):
        base = op.getarg(0)
        offset, fieldsize, signed = unpack_fielddescr(op.getdescr())
        addr = IntBinOp("+", base, ConstInt(offset))
        typ = HeapType.from_size_and_sign(fieldsize, signed)
        self.jsbuilder.emit_load(op.result, addr, typ)

    genop_getfield_raw = genop_getfield_gc
    genop_getfield_raw_pure = genop_getfield_gc
    genop_getfield_gc_pure = genop_getfield_gc

    def genop_setfield_gc(self, op):
        base = op.getarg(0)
        value = op.getarg(1)
        offset, fieldsize, signed = unpack_fielddescr(op.getdescr())
        addr = IntBinOp("+", base, ConstInt(offset))
        typ = HeapType.from_size_and_sign(fieldsize, signed)
        self.jsbuilder.emit_store(value, addr, typ)

    genop_setfield_raw = genop_setfield_gc

    def _genop_int_binop(binop):
        def genop_int_binop(self, op):
            abval = IntBinOp(binop, op.getarg(0), op.getarg(1))
            self.jsbuilder.emit_assignment(op.result, abval)
        return genop_int_binop

    genop_int_lt = _genop_int_binop("<")
    genop_int_le = _genop_int_binop("<=")
    genop_int_eq = _genop_int_binop("==")
    genop_int_ne = _genop_int_binop("!=")
    genop_int_gt = _genop_int_binop(">")
    genop_int_ge = _genop_int_binop(">=")
    genop_int_add = _genop_int_binop("+")  # XXX TODO: semantics of these?
    genop_int_sub = _genop_int_binop("-")  # esp. re: overflow
    genop_int_mul = _genop_int_binop("Math.imul")
    genop_int_and = _genop_int_binop("&")
    genop_int_or = _genop_int_binop("|")

    def genop_jump(self, op):
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        # Write arguments into the allocated boxes.
        argboxes = descr._asmjs_argboxes
        if argboxes is None:
            # XXX TODO: how do we write args for non-local jumps?
            # Do we need to, or will bridge take care of that?
            raise NotImplementedError("cant send args to non-local jump")
        assert len(argboxes) == op.numargs()
        for i in range(op.numargs()):
            box = op.getarg(i)
            self.jsbuilder.emit_assignment(argboxes[i], box)
        self.jsbuilder.emit_jump(descr)

    def genop_finish(self, op):
        descr = op.getdescr()
        fail_descr = cast_instance_to_gcref(descr)
        # If there's a return value, write it into the frame.
        # It is always the first value in the frame.
        if op.numargs() == 1:
            return_value = op.getarg(0)
            return_addr = JitFrameAddr_base()
            typ = HeapType.from_box(return_value)
            self.jsbuilder.emit_store(return_value, return_addr, typ)
        # Write the descr into the frame slot.
        addr = JitFrameAddr_descr()
        self.jsbuilder.emit_store(ConstPtr(fail_descr), addr, Int32)
        # XXX TODO: gc stuff goes here, as seen in x86 backend.
        self.jsbuilder.emit_exit()

    def genop_guard_true(self, op):
        self.jsbuilder.emit("if(!")
        self.jsbuilder.emit_value(op.getarg(0))
        self.jsbuilder.emit("){\n")
        self._genop_guard_failure(op)
        self.jsbuilder.emit("}\n")

    def _genop_guard_failure(self, op):
        descr = op.getdescr()
        assert isinstance(descr, AbstractFailDescr)
        # Write the fail_descr into the frame slot.
        fail_descr = cast_instance_to_gcref(descr)
        addr = JitFrameAddr_descr()
        self.jsbuilder.emit_store(ConstPtr(fail_descr), addr, Int32)
        # Write the failargs into the frame.
        offset = 0
        failargs = op.getfailargs()
        faillocs = [-1] * len(failargs)
        for i in range(len(failargs)):
            failarg = failargs[i]
            # XXX TODO: alignment issues for multi-word values
            addr = IntBinOp("+", JitFrameAddr_base(), ConstInt(offset))
            typ = HeapType.from_box(failarg)
            self.jsbuilder.emit_store(failarg, addr, typ)
            faillocs[i] = offset
            offset += typ.size
        # Put the failloc info on the descr, where the runner
        # and any future bridges can find them.
        descr._asmjs_faillocs = faillocs
        # Allocate a new function for this guard, and goto it.
        # Initially this is a quick exit, but it might get recompiled
        # into a bridge if the exit gets hot.
        descr._asmjs_funcid = support.jitReserve()
        self.jsbuilder.emit_goto(descr._asmjs_funcid)

    def not_implemented_op(self, op):
        self._print_op(op)
        #os.write(2, '[asmjs] %s\n' % msg)
        self.jsbuilder.emit("// NOT IMPLEMENTED: %s\n" % (op,))
        #raise NotImplementedError("not implemented: " + str(op))

    def _print_op(self, op):
        print "OPERATION:", op
        for i in range(op.numargs()):
            print "  ARG:", op.getarg(i)
        print "  RES:", op.result


# Build a dispatch table mapping opnums to the method that emits code them.

genop_list = [AssemblerASMJS.not_implemented_op] * rop._LAST

for name, value in AssemblerASMJS.__dict__.iteritems():
    if name.startswith('genop_'):
        opname = name[len('genop_'):]
        num = getattr(rop, opname.upper())
        genop_list[num] = value
