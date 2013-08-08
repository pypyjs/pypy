
from rpython.rlib.rstring import StringBuilder
from rpython.jit.metainterp.history import (ConstInt, ConstFloat, ConstPtr,
                                            Box, TargetToken, AbstractValue,
                                            INT, REF, FLOAT)


class HeapType(object):
    """Object representing different types that can be loaded from the heap.

    Instances of HeapType are used as simple type markers when generating
    heap loads and stores.  They control the particular view of the heap used
    as well as the size of the resulting object.
    """

    def __init__(self, lltype, heap_name, size, shift):
        self.lltype = lltype
        self.heap_name = heap_name
        self.size = size
        self.shift = shift
        assert 2**shift == size

    @staticmethod
    def from_box(box):
        """Determine the HeapType marker that matches the given Box."""
        assert isinstance(box, Box)
        if box.type == FLOAT:
            return Float64
        if box.type == INT:
            return Int32
        if box.type == REF:
            return UInt32
        raise NotImplementedError("unsupported box type: %s" % (box.type,))

    @staticmethod
    def from_size(size):
        """Determine the HeapType marker for objects of the given size."""
        if size == 1:
            return Int8
        if size == 2:
            return Int16
        if size == 4:
            return Int32
        if size == 8:
            return Float64
        raise NotImplementedError("unsupported box size: %d" % (size,))

    @staticmethod
    def from_size_and_sign(size, sign):
        """Determine HeapType marker matching the given size and signedness."""
        if size == 1:
            if not sign:
                return UInt8
            return Int8
        if size == 2:
            if not sign:
                return UInt16
            return Int16
        if size == 4:
            if not sign:
                return UInt32
            return Int32
        if size == 8:
            if not sign:
                raise NotImplementedError("unsupported 8-byte unsigned")
            return Float64
        raise NotImplementedError("unsupported box size: %d" % (size,))


Int8 = HeapType(INT, "H8", 1, 0)
Int16 = HeapType(INT, "H16", 2, 1)
Int32 = HeapType(INT, "H32", 4, 2)
UInt8 = HeapType(INT, "HU8", 1, 0)
UInt16 = HeapType(INT, "HU16", 2, 1)
UInt32 = HeapType(INT, "HU32", 4, 2)
Float32 = HeapType(FLOAT, "HF32", 4, 2)
Float64 = HeapType(FLOAT, "HF64", 8, 3)


class JitFrameAddr(AbstractValue):
    """AbstractValue representing the address of the active jitframe object.

    The builder will render this as a reference to the special 'jitframe'
    variable, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass


class JitFrameAddr_base(AbstractValue):
    """AbstractValue representing the address of jitframe scratch space.

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to its scratch space, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass


class JitFrameAddr_descr(AbstractValue):
    """AbstractValue representing the address of jitframe.jf_descr

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to the given field, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass


class AbstractBinOp(AbstractValue):
    """AbstractValue representing a binary operation on two other values.

    This provides a simple way to build up value expressions and have them
    rendered inline by the ASMJSBuilder.
    """
    binop = ""
    lhs = None
    rhs = None
    _attrs_ = ('binop', 'lhs', 'rhs')

    def __init__(self, binop, lhs, rhs):
        self.binop = binop
        self.lhs = lhs
        self.rhs = rhs


class IntBinOp(AbstractBinOp):
    """AbstractBinOp between two integers, with an integer value."""
    type = INT


class FloatBinOp(AbstractBinOp):
    """AbstractBinOp between two floats, with a float value."""
    type = FLOAT


class ASMJSBuilder(object):
    """Class for building ASMJS assembly of a trace.

    This class can be used to build up the source for a single-function ASMJS
    module.  The structure of the code is optimized for a labelled trace with
    a single jump backwards at the end.  It renders the code as a prelude
    followed by a single while-loop. Jumps into labeled points in the code are
    facilitated by conditionals and an input argument named "goto":

        function (jitframe, goto) {
          if (goto <= 0) {
            <block zero>
          }
          ...
          if (goto <= M) {
            <block M>
          }
          while(true) {
            if (goto <= M+1) {
              <block M+1>
            }
            ...
            if (goto <= N) {
              <block N>
            }
            goto = 0;
          }
        }

    According to the emscripten technical paper, it is quite important to use
    high-level control-flow structures rather than a more general switch-in-
    a-loop.  An alternative would be to import and use the whole emscripten
    relooper algorithm, but that sounds painful and expensive.

    XXX TODO: is this too general, or too broad?  try to do better!
    XXX TODO: benchmark against a generic switch-in-a-loop construct.

    The compiled function always returns an integer.  If it returns zero then
    execution has completed.  If it returns a positive integer, this identifies
    another compiled function into which execution should jump.  We expect to
    be run inside some sort of trampoline that can execute these jumps without
    blowing the call stack.

    Trace boxes are assigned to variables of the appopriate type, and all such
    boxes must be declared before trying to emit any code.  You would typically
    do something like the following:

        jsbuilder = ASMJSBuilder(cpu)

        for [each box in the trace]:
            jsbuilder.allocate_variable(box)
            if [this is the last use of that box]:
                jsbuilder.free_variable(box)

        if [the last operation is a local JUMP]:
            jsbuilder.set_loop_entry_token(jump_token)

        for [each operation in the trace]:
            jsbuilder.emit_[whatever]() as needed.

        src = jsbuilder.finish()

    XXX TODO: produce more concise code once we're sure it's working,
              e.g. by removing newlines and semicolons.
    """

    def __init__(self, cpu):
        self.cpu = cpu
        self.srcbuffer = StringBuilder()
        self.num_variables = 0
        self.box_variables = {}
        self.free_variables_intish = []
        self.free_variables_double = []
        self.token_labels = {}
        self.loop_entry_token = None
        self._write_prelude()

    def finish(self):
        self._write_epilog()
        return self.srcbuffer.build()

    def _write_prelude(self):
        # Standard asmjs prelude stuff.
        self.emit('function(global, foreign, heap){\n')
        self.emit('"use asm";\n')
        self.emit('var H8 = new global.Int8Array(heap);\n')
        self.emit('var H16 = new global.Int16Array(heap);\n')
        self.emit('var H32 = new global.Int32Array(heap);\n')
        self.emit('var HU8 = new global.Uint8Array(heap);\n')
        self.emit('var HU16 = new global.Uint16Array(heap);\n')
        self.emit('var HU32 = new global.Uint32Array(heap);\n')
        self.emit('var HF32 = new global.Float32Array(heap);\n')
        self.emit('var HF64 = new global.Float64Array(heap);\n')
        # Extra useful functions from the Module object.
        self.emit('var jitInvoke = foreign._jitInvoke;\n')
        self.emit('function F(jitframe, goto){\n')
        self.emit('jitframe=jitframe|0;\n')
        self.emit('goto=goto|0;\n')

    def _write_epilog(self):
        # End the final labelled block, if any.
        if len(self.token_labels) > 0:
            self.emit("}\n")
        # End the loop, if we're in one.
        if self.loop_entry_token is not None:
            self.emit("goto = 0|0;\n")
            self.emit("}\n")
        # The compiled function always returns the jitframe pointer.
        self.emit('return jitframe|0;\n')
        self.emit('}\n')
        # Export the singleton compiled function.
        self.emit('return F;\n')
        self.emit('}\n')

    def allocate_variable(self, box):
        """Allocate a variable to be used for the given box.

        If there is a previously-freed variable of appropriate type then
        that variable is re-used; otherwise a fresh variable name is
        allocated and emitted.
        """
        varname = self.box_variables.get(box, None)
        if varname is None:
            # Try to re-use an existing variable.
            if box.type == FLOAT:
                if len(self.free_variables_double) > 0:
                    varname = self.free_variables_double.pop()
            else:
                if len(self.free_variables_intish) > 0:
                    varname = self.free_variables_intish.pop()
            # Nope, allocate a new one of the appropriate type.
            if varname is None:
                self.num_variables += 1
                varname = "box" + str(self.num_variables)
                if box.type == FLOAT:
                    initial_value = "0.0"
                else:
                    initial_value = "0"
                self.emit_statement("var %s=%s" % (varname, initial_value))
            self.box_variables[box] = varname

    def free_variable(self, box):
        """Free up the variable allocated to the given box.

        This should be called during variable allocation if it is known that
        the box will no longer be used.  Any variable currently allocated to
        that box will be freed up for use by future boxes with non-overlapping
        lifetimes.
        """
        assert isinstance(box, Box)
        varname = self.box_variables.get(box, None)
        if varname is not None:
            if box.type == FLOAT:
                self.free_variables_double.append(varname)
            else:
                self.free_variables_intish.append(varname)

    def set_loop_entry_token(self, token):
        """Indicate which token is the target of the final JUMP operation.

        If the trace being compiled ends in a JUMP operation, this method
        should be called to indicate the target of that jump.  It causes the
        generated code to contain an appropriate loop statement.
        """
        assert self.loop_entry_token is None
        self.loop_entry_token = token

    def emit_token_label(self, token):
        """Emit a label indicating the start of this token's block.

        This function can be be used to split up the generated code into
        distinct labelled blocks.  It returns an integer identifying the
        label for the token; passing this integer as the 'goto' argument
        will jump control directly to that label.
        """
        assert isinstance(token, TargetToken)
        # Close the previous block, if any.
        if len(self.token_labels):
            self.emit("}\n")
        # If this is the loop token, enter the loop.
        if token == self.loop_entry_token:
            self.emit("while(1){\n")
        # Assign a label and begin the new block.
        label = len(self.token_labels) + 1
        self.token_labels[token] = label
        self.emit("if(goto|0 <= %d|0){\n" % (label,))
        return label

    def emit(self, code):
        """Emit the given string directly into the generated code."""
        self.srcbuffer.append(code)

    def emit_statement(self, code):
        """Emit the given string as a statement into.

        This sanity-checks and appends an appropriate statement terminator.
        """
        self.emit(code)
        if code.endswith("}"):
            self.emit("\n")
        else:
            self.emit(";\n")

    def emit_int_literal(self, value):
        """Emit an integer literal value in the generated source."""
        self.emit(str(value))
        self.emit("|0")

    def emit_float_literal(self, value):
        """Emit a float literal value in the generated source."""
        value_as_str = str(value)
        self.emit(value_as_str)
        if "." not in value_as_str:
            self.emit(".0")

    def emit_value(self, abval):
        """Emit a reference to an AbstractValue in the generated source.

        The provided value could be a Box, a Const, a BinOp expression or a
        reference to some of our internal state.  Each is detected and rendered
        with appropriate type coercions and parens.
        """
        if isinstance(abval, JitFrameAddr):
            self.emit("jitframe|0")
        elif isinstance(abval, JitFrameAddr_base):
            offset = self.cpu.get_baseofs_of_frame_field()
            self.emit("((jitframe|0) + (%d|0))" % (offset,))
        elif isinstance(abval, JitFrameAddr_descr):
            offset = self.cpu.get_ofs_of_frame_field("jf_descr")
            self.emit("((jitframe|0) + (%d|0))" % (offset,))
        elif isinstance(abval, ConstInt):
            self.emit_int_literal(abval.value)
        elif isinstance(abval, ConstFloat):
            self.emit_float_literal(str(abval.value))
        elif isinstance(abval, ConstPtr):
            self.emit_int_literal(self.cpu.cast_ptr_to_int(abval.value))
        elif isinstance(abval, Box):
            if abval.type == FLOAT:
                self.emit("+")
            self.emit(self.box_variables[abval])
            if abval.type == INT or abval.type == REF:
                self.emit("|0")
        elif isinstance(abval, AbstractBinOp):
            if isinstance(abval, FloatBinOp):
                self.emit("+")
            self.emit("(")
            if abval.binop[0].isalpha():
                self.emit(abval.binop)
                self.emit("(")
                self.emit_value(abval.lhs)
                self.emit(",")
                self.emit_value(abval.rhs)
                self.emit(")")
            else:
                self.emit("(")
                self.emit_value(abval.lhs)
                self.emit(")")
                self.emit(abval.binop)
                self.emit("(")
                self.emit_value(abval.rhs)
                self.emit(")")
            self.emit(")")
            if isinstance(abval, IntBinOp):
                self.emit("|0")
        else:
            raise NotImplementedError("unsupported abstract value")

    def emit_assignment(self, resbox, value):
        """Emit an assignment of a value to the given box."""
        varname = self.box_variables[resbox]
        self.emit(varname)
        self.emit("=")
        self.emit_value(value)
        self.emit(";\n")

    def emit_load(self, resbox, addr, typ=None):
        """Emit a typed load operation from the heap into a box.

        Given a result-storing box, a heap address, and an optional HeapType,
        this method generates a genric "load from heap" instruction of the
        the following form:

            boxvar = HEAPVIEW[(addr) >> shift];

        """
        if typ is None:
            typ = HeapType.from_box(resbox)
        varname = self.box_variables[resbox]
        self.emit(varname)
        self.emit("=")
        if typ.lltype == FLOAT:
            self.emit("+")
        self.emit(typ.heap_name)
        self.emit("[(")
        self.emit_value(addr)
        self.emit(") >> ")
        self.emit(str(typ.shift))
        self.emit("]")
        if typ.lltype == INT:
            self.emit("|0")
        self.emit(";\n")

    def emit_store(self, value, addr, typ):
        """Emit a typed store operation of a value into the heap.

        Given an abstract value, a heap address, and a HeapType, this method
        generates a generic "store to heap" instruction of the following form:

            HEAPVIEW[(addr) >> shift] = value;

        """
        self.emit(typ.heap_name)
        self.emit("[(")
        self.emit_value(addr)
        self.emit(") >> ")
        self.emit(str(typ.shift))
        self.emit("]=")
        self.emit_value(value)
        self.emit(";\n")

    def emit_jump(self, token):
        """Emit a jump to the given token.

        For jumps to the loop-entry token of the current function, this just
        moves to the next iteration of the loop.  For other targets it looks
        up the compiled functionid and goto address, and returns that value
        to the trampoline.
        """
        assert isinstance(token, TargetToken)
        if token == self.loop_entry_token:
            # A local loop, just move to the next iteration.
            self.emit_statement("continue")
        else:
            # An external loop, need to goto via trampoline.
            self.emit_goto(token._asmjs_funcid, token._asmjs_label)

    def emit_goto(self, funcid, label=0):
        """Emit a goto to the funcid and label.

        Essentially we just encode the funcid and label into a single integer
        and return it.  We expect to be runninng inside some sort of trampoline
        that can actually execute the goto.
        """
        assert funcid < 2**24
        assert label < 0xFF
        next_call = (funcid << 8) | label
        self.emit_statement("return %d" % (next_call,))

    def emit_exit(self):
        """Emit an immediate return from the function."""
        self.emit("return 0;\n")
