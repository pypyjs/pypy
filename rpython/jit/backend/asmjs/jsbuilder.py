
import struct
from binascii import unhexlify
from rpython.memory.gctypelayout import GCData
from rpython.rtyper.lltypesystem import lltype, rffi, llmemory
from rpython.rtyper.lltypesystem.lloperation import llop
from rpython.jit.metainterp.history import (ConstInt, ConstFloat, ConstPtr,
                                            TargetToken, AbstractValue, Box,
                                            Const, INT, REF, FLOAT, BoxInt)

from rpython.jit.backend.asmjs.arch import DEBUGMODE


class HeapType(object):
    """Object representing different types that can be loaded from the heap.

    Instances of HeapType are used as simple type markers when generating
    heap loads and stores.  They control the particular view of the heap used
    as well as the size of the resulting value.
    """

    def __init__(self, lltype, heap_name, size, shift):
        self.lltype = lltype
        self.heap_name = heap_name
        self.size = size
        self.shift = shift
        assert 2**shift == size

    @staticmethod
    def from_kind(kind):
        """Determine the HeapType marker that matches the given ll kind."""
        if kind == FLOAT:
            return Float64
        if kind == INT:
            return Int32
        if kind == REF:
            return UInt32
        print "unsupported kind: %s" % (kind,)
        raise RuntimeError("unsupported kind: %s" % (kind,))

    @staticmethod
    def from_box(box):
        """Determine the HeapType marker that matches the given Box."""
        if box is None:
            # Treating this a NULL pointer simplifies some stuff.
            return Int32
        elif isinstance(box, Box) or isinstance(box, Const):
            if box.type == FLOAT:
                return Float64
            if box.type == INT:
                return Int32
            if box.type == REF:
                return UInt32
            print "unsupported box type: %s" % (box.type,)
            raise NotImplementedError("unsupported box type: %s" % (box.type,))
        else:
            print "from_box does not support %s" % (box,)
            raise NotImplementedError("from_box does not support %s" % (box,))

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
            # XXX TODO: is signedness relevant for float types?
            #if not sign:
            #    raise NotImplementedError("unsupported 8-byte unsigned")
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


class ASMJSValue(AbstractValue):
    """An AbstractValue that knows how to render its own asmjs code."""

    def emit_value(self, jsbuilder):
        raise NotImplementedError


class Placeholder(ASMJSValue):
    """Placeholder integer, for values we want to fill in later."""

    def __init__(self, varname):
        self.varname = varname

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(%s|0)" % (self.varname,))


class TempDoublePtr(ASMJSValue):
    """ASMJSValue representing the address of double storage scratch-space."""
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(tempDoublePtr|0)")


class JitFrameAddr(ASMJSValue):
    """ASMJSValue representing the address of the active jitframe object."""
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(jitframe|0)")


class JitFrameAddr_base(ASMJSValue):
    """ASMJSalue representing the address of jitframe scratch space."""
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        offset = jsbuilder.cpu.get_baseofs_of_frame_field()
        jsbuilder.emit("(((jitframe|0) + (%d|0))|0)" % (offset,))


class JitFrameAddr_descr(ASMJSValue):
    """ASMJSValue representing the address of jitframe.jf_descr

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to the given field, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        offset = jsbuilder.cpu.get_ofs_of_frame_field("jf_descr")
        jsbuilder.emit("(((jitframe|0) + (%d|0))|0)" % (offset,))


class JitFrameAddr_force_descr(ASMJSValue):
    """ASMJSValue representing the address of jitframe.jf_force_descr

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to the given field, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        offset = jsbuilder.cpu.get_ofs_of_frame_field("jf_force_descr")
        jsbuilder.emit("(((jitframe|0) + (%d|0))|0)" % (offset,))


class JitFrameAddr_guard_exc(ASMJSValue):
    """ASMJSValue representing the address of jitframe.jf_guard_exc

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to the given field, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        offset = jsbuilder.cpu.get_ofs_of_frame_field("jf_guard_exc")
        jsbuilder.emit("(((jitframe|0) + (%d|0))|0)" % (offset,))


class JitFrameAddr_gcmap(ASMJSValue):
    """ASMJSValue representing the address of jitframe.jf_gcmap

    The builder will render this as a reference to the special 'jitframe'
    variable, offset to point to the given field, appropriately cooerced.
    """
    type = REF

    def __init__(self):
        pass

    def emit_value(self, jsbuilder):
        offset = jsbuilder.cpu.get_ofs_of_frame_field("jf_gcmap")
        jsbuilder.emit("(((jitframe|0) + (%d|0))|0)" % (offset,))


class HeapData(ASMJSValue):
    """ASMJSValue representing data read from the heap."""
    type = REF  # XXX TODO: need to split this for different types
    heaptype = None
    addr = None

    def __init__(self, heaptype, addr):
        self.heaptype = heaptype
        self.addr = addr
        # Alignment issues mean we can't reliably read multi-word data...
        assert heaptype.size <= 4

    def emit_value(self, jsbuilder):
        if self.type == FLOAT:
            jsbuilder.emit("+")
        jsbuilder.emit(self.heaptype.heap_name)
        jsbuilder.emit("[(")
        jsbuilder.emit_value(self.addr)
        jsbuilder.emit(") >> ")
        jsbuilder.emit(str(self.heaptype.shift))
        jsbuilder.emit("]")
        if self.type != INT:
            jsbuilder.emit("|0")


class IntCast(ASMJSValue):
    """Explicitly cast a value to type int."""
    type = INT
    value = None

    def __init__(self, value):
        self.value = value

    def emit_value(self, jsbuilder):
        jsbuilder.emit("((")
        jsbuilder.emit_value(self.value)
        jsbuilder.emit(")|0)")


class SIntCast(ASMJSValue):
    """Explicitly cast a value to type signed-int."""
    type = INT
    value = None

    def __init__(self, value):
        self.value = value

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(~(~(")
        jsbuilder.emit_value(self.value)
        jsbuilder.emit("|0)))")


class UIntCast(ASMJSValue):
    """Explicitly cast a value to type unsigned-int."""
    type = INT
    value = None

    def __init__(self, value):
        self.value = value

    def emit_value(self, jsbuilder):
        jsbuilder.emit("((")
        jsbuilder.emit_value(self.value)
        jsbuilder.emit(") >>> 0)")


class DblCast(ASMJSValue):
    """Explicitly cast a value to type double."""
    type = FLOAT
    value = None

    def __init__(self, value):
        self.value = value

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(+(")
        jsbuilder.emit_value(self.value)
        jsbuilder.emit("))")


class ClassPtrTypeID(ASMJSValue):
    """ASMJSValue representing the typeid for a class pointer.

    This is a special value for handling class comparisons when we're not
    using type pointers.  It extracts an "expected type id" form the class
    pointer, which can be compared agaist the first half-word in the object
    pointer.  Logic shamelessly cargo-culted from x86 backend.
    """
    type = INT
    classptr = None

    def __init__(self, classptr):
        self.classptr = classptr

    def emit_value(self, jsbuilder):
        sizeof_ti = rffi.sizeof(GCData.TYPE_INFO)
        type_info_group = llop.gc_get_type_info_group(llmemory.Address)
        type_info_group = rffi.cast(lltype.Signed, type_info_group)
        expected_typeid = IntBinOp("-", self.classptr,
                                   ConstInt(sizeof_ti + type_info_group))
        expected_typeid = IntBinOp(">>", expected_typeid, ConstInt(2))
        expected_typeid = IntBinOp("&", expected_typeid, ConstInt(0xFFFF))
        jsbuilder.emit_value(expected_typeid)


class ASMJSOp(ASMJSValue):
    """ASMJSValue representing an operation on existing values.

    This provides a simple way to build up value expressions and have them
    rendered inline by the ASMJSBuilder.
    """
    pass


class ASMJSUnaryOp(ASMJSOp):
    """ASMJSOp representing a unary operation on a single value."""
    operator = ""
    operand = None

    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(")
        if self.operator[0].isalpha():
            jsbuilder.emit(self.operator)
            jsbuilder.emit("(")
            jsbuilder.emit_value(self.operand)
            jsbuilder.emit(")")
        else:
            jsbuilder.emit(self.operator)
            jsbuilder.emit("(")
            jsbuilder.emit_value(self.operand)
            jsbuilder.emit(")")
        jsbuilder.emit(")")


class IntUnaryOp(ASMJSUnaryOp):
    """ASMJSUnaryOp with an integer value."""
    type = INT

    def emit_value(self, jsbuilder):
        ASMJSUnaryOp.emit_value(self, jsbuilder)
        jsbuilder.emit("|0")


class DblUnaryOp(ASMJSUnaryOp):
    """ASMJSUnaryOp with a float value."""
    type = FLOAT

    def emit_value(self, jsbuilder):
        jsbuilder.emit("+")
        ASMJSUnaryOp.emit_value(self, jsbuilder)


class ASMJSBinOp(ASMJSOp):
    """ASMJSOp representing a binary operation on two other values."""
    binop = ""
    lhs = None
    rhs = None

    def __init__(self, binop, lhs, rhs):
        self.binop = binop
        self.lhs = lhs
        self.rhs = rhs

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(")
        if self.binop[0].isalpha():
            jsbuilder.emit(self.binop)
            jsbuilder.emit("(")
            jsbuilder.emit_value(self.lhs)
            jsbuilder.emit(",")
            jsbuilder.emit_value(self.rhs)
            jsbuilder.emit(")")
        else:
            jsbuilder.emit("(")
            jsbuilder.emit_value(self.lhs)
            jsbuilder.emit(")")
            jsbuilder.emit(self.binop)
            jsbuilder.emit("(")
            jsbuilder.emit_value(self.rhs)
            jsbuilder.emit(")")
        jsbuilder.emit(")")


class IntBinOp(ASMJSBinOp):
    """ASMJSBinOp with an integer value."""
    type = INT

    def emit_value(self, jsbuilder):
        ASMJSBinOp.emit_value(self, jsbuilder)
        jsbuilder.emit("|0")


class DblBinOp(ASMJSBinOp):
    """ASMJSBinOp with a float value."""
    type = FLOAT

    def emit_value(self, jsbuilder):
        jsbuilder.emit("+")
        ASMJSBinOp.emit_value(self, jsbuilder)


class IntCallFunc(ASMJSValue):
    """ASMJSValue representing result of external function call."""
    type = INT

    def __init__(self, funcname, callsig, arguments):
        assert callsig[0] == "i"
        self.funcname = funcname
        self.callsig = callsig
        self.arguments = arguments

    def emit_value(self, jsbuilder):
        funcname = "_" + self.funcname
        jsbuilder.imported_functions[funcname] = funcname
        jsbuilder.emit("(")
        jsbuilder.emit(funcname)
        jsbuilder.emit("(")
        for i in xrange(len(self.arguments)):
            if i > 0:
                jsbuilder.emit(",")
            jsbuilder.emit_value(self.arguments[i])
        jsbuilder.emit(")")
        jsbuilder.emit("|0)")


class IntScaleOp(ASMJSOp):
    """Special-case operation for multiplying by a small integer.

    ASMJS requires a raw numeric literal when multiplying by an integer
    constant, so we have to render it without any of the usual brackets
    or typecasting, then cast the entire expression at the end.
    """
    type = INT
    value = None
    scale = 1

    def __init__(self, value, scale):
        self.value = value
        self.scale = scale

    def emit_value(self, jsbuilder):
        jsbuilder.emit("(((")
        jsbuilder.emit_value(self.value)
        jsbuilder.emit(")*")
        jsbuilder.emit(str(self.scale))
        jsbuilder.emit(")|0)")


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
            continue
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
        self.source_chunks = []
        self.num_variables = 0
        self.box_variables = {}
        self.immortal_boxes = {}
        self.placeholder_variables = {}
        self.free_variables_intish = []
        self.free_variables_double = []
        self.token_labels = {}
        self.loop_entry_token = None
        self.imported_functions = {}

    def finish(self):
        return "".join(self._build_prelude() +
                       self.source_chunks +
                       self._build_epilog())

    def _build_prelude(self):
        chunks = []
        # Standard asmjs prelude stuff.
        chunks.append('function(stdlib, foreign, heap){\n')
        chunks.append('"use asm";\n')
        chunks.append('var H8 = new stdlib.Int8Array(heap);\n')
        chunks.append('var H16 = new stdlib.Int16Array(heap);\n')
        chunks.append('var H32 = new stdlib.Int32Array(heap);\n')
        chunks.append('var HU8 = new stdlib.Uint8Array(heap);\n')
        chunks.append('var HU16 = new stdlib.Uint16Array(heap);\n')
        chunks.append('var HU32 = new stdlib.Uint32Array(heap);\n')
        chunks.append('var HF32 = new stdlib.Float32Array(heap);\n')
        chunks.append('var HF64 = new stdlib.Float64Array(heap);\n')
        chunks.append('var imul = stdlib.Math.imul;\n')
        # Import any required names from the Module object.
        chunks.append('var tempDoublePtr = foreign.tempDoublePtr|0;\n')
        for funcname in self.imported_functions:
            chunks.append('var %s = foreign.%s;\n' % (funcname, funcname))
        # The function definition, including variable declarations.
        chunks.append('function F(jitframe, goto){\n')
        chunks.append('jitframe=jitframe|0;\n')
        chunks.append('goto=goto|0;\n')
        for box, varname in self.box_variables.iteritems():
            if box.type == FLOAT:
                chunks.append("var %s=0.0;\n" % (varname,))
            else:
                chunks.append("var %s=0;\n" % (varname,))
        for varname in self.free_variables_double:
            chunks.append("var %s=0.0;\n" % (varname,))
        for varname in self.free_variables_intish:
            chunks.append("var %s=0;\n" % (varname,))
        for varname, value in self.placeholder_variables.items():
            chunks.append("var %s=%d;\n" % (varname, value))
        chunks.append("if((goto|0) <= (0|0)){\n")
        return chunks

    def _build_epilog(self):
        chunks = []
        # End the final labelled block.
        chunks.append("}\n")
        # End the loop, if we're in one.
        if self.loop_entry_token is not None:
            chunks.append("goto = 0|0;\n")
            chunks.append("}\n")
        # The compiled function always returns zero when it's finished.
        chunks.append('return 0;\n')
        chunks.append('}\n')
        # Export the singleton compiled function.
        chunks.append('return F;\n')
        chunks.append('}\n')
        return chunks

    def allocate_variable(self, box, immortal=False):
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
            self.box_variables[box] = varname
        if immortal:
            self.immortal_boxes[box] = True
        return varname

    def free_variable(self, box):
        """Free up the variable allocated to the given box.

        This should be called during variable allocation if it is known that
        the box will no longer be used.  Any variable currently allocated to
        that box will be freed up for use by future boxes with non-overlapping
        lifetimes.
        """
        assert isinstance(box, Box)
        if box in self.immortal_boxes:
            return
        varname = self.box_variables.pop(box, None)
        if varname is not None:
            if box.type == FLOAT:
                self.free_variables_double.append(varname)
            else:
                self.free_variables_intish.append(varname)

    def allocate_placeholder(self):
        """Allocate a placeholder integer, which you can give a value later."""
        varname = "p%d" % (len(self.placeholder_variables) + 1,)
        self.placeholder_variables[varname] = 0
        return Placeholder(varname)

    def fill_placeholder(self, placeholder, value):
        self.placeholder_variables[placeholder.varname] = value

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
        # Close the previous block.
        self.emit("}\n")
        # If this is the loop token, enter the loop.
        if token == self.loop_entry_token:
            self.emit("while(1){\n")
        # Assign a label and begin the new block.
        label = len(self.token_labels) + 1
        self.token_labels[token] = label
        self.emit("if((goto|0) <= (%d|0)){\n" % (label,))
        return label

    def emit(self, code):
        """Emit the given string directly into the generated code."""
        self.source_chunks.append(code)

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
        if "." not in value_as_str:
            value_as_str += ".0"
        self.emit(value_as_str)

    def emit_value(self, abval):
        """Emit a reference to an AbstractValue in the generated source.

        The provided value could be a Box, a Const, an AbstractOp expression
        or a reference to some of our internal state.  Each is detected and
        rendered with appropriate type coercions and parens.
        """
        if abval is None:
            self.emit_int_literal(0)
        elif isinstance(abval, ASMJSValue):
            abval.emit_value(self)
        elif isinstance(abval, ConstInt):
            intval = rffi.cast(lltype.Signed, abval.getint())
            self.emit_int_literal(intval)
        elif isinstance(abval, ConstFloat):
            self.emit_float_literal(abval.getfloat())
        elif isinstance(abval, ConstPtr):
            ptrval = rffi.cast(lltype.Signed, abval.getref_base())
            self.emit_int_literal(ptrval)
        elif isinstance(abval, Box):
            if abval.type == FLOAT:
                self.emit("+")
            if abval not in self.box_variables:
                print "ERROR: box doesn't have a variable", abval
                raise RuntimeError("box doesnt have a variable: %s" % (abval,))
            self.emit(self.box_variables[abval])
            if abval.type == INT or abval.type == REF:
                self.emit("|0")
        else:
            raise NotImplementedError("unsupported abstract value")

    def emit_assignment(self, resbox, value):
        """Emit an assignment of a value to the given box."""
        varname = self.allocate_variable(resbox)
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
        # For doubles, we can't guarantee that the data is aligned.
        # Read it as two 32bit ints into a properly aligned chunk.
        if typ is Float64:
            tempaddr = BoxInt()
            self.emit_assignment(tempaddr, addr)
            addr = TempDoublePtr()
            first_word = HeapData(Int32, tempaddr)
            self.emit_store(first_word, addr, Int32)
            word = ConstInt(4)
            second_word = HeapData(Int32, IntBinOp("+", tempaddr, word))
            self.emit_store(second_word, IntBinOp("+", addr, word), Int32)
            self.free_variable(tempaddr)
        # Now we can do the actual load.
        varname = self.allocate_variable(resbox)
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

    def emit_assign_jitframe(self, value):
        self.emit("jitframe")
        self.emit("=")
        self.emit_value(value)
        self.emit(";\n")

    def emit_store(self, value, addr, typ):
        """Emit a typed store operation of a value into the heap.

        Given an abstract value, a heap address, and a HeapType, this method
        generates a generic "store to heap" instruction of the following form:

            HEAPVIEW[(addr) >> shift] = value;

        """
        # For doubles, we can't guarantee that the data is aligned.
        # We have to store it into a properly aligned chunk, then
        # copy to final destination as two 32-bit ints.
        tempaddr = None
        if typ is Float64:
            tempaddr = BoxInt()
            self.emit_assignment(tempaddr, addr)
            addr = TempDoublePtr()
        self.emit(typ.heap_name)
        self.emit("[(")
        self.emit_value(addr)
        self.emit(") >> ")
        self.emit(str(typ.shift))
        self.emit("]=")
        self.emit_value(value)
        self.emit(";\n")
        if typ is Float64:
            first_word = HeapData(Int32, addr)
            self.emit_store(first_word, tempaddr, Int32)
            word = ConstInt(4)
            second_word = HeapData(Int32, IntBinOp("+", addr, word))
            self.emit_store(second_word, IntBinOp("+", tempaddr, word), Int32)
            self.free_variable(tempaddr)

    def emit_call(self, funcname, callsig, arguments, resbox):
        """Emit a call to a named function."""
        assert len(callsig) == len(arguments) + 1
        # XXX TODO: we can give these shorter local names.
        funcname = "_" + funcname
        self.imported_functions[funcname] = funcname
        if resbox is not None:
            varname = self.allocate_variable(resbox)
            self.emit(varname)
            self.emit("=")
        if callsig[0] == "f":
            self.emit("+")
        self.emit(funcname)
        self.emit("(")
        for i in xrange(len(arguments)):
            if i > 0:
                self.emit(",")
            self.emit_value(arguments[i])
        self.emit(")")
        if callsig[0] == "i":
            self.emit("|0")
        self.emit(";\n")

    def emit_dyncall(self, funcaddr, callsig, arguments, resbox=None):
        """Emit a call to a function pointer."""
        assert len(callsig) == len(arguments) + 1
        # XXX TODO: we can give these shorter local names.
        dyncall = "dynCall_" + callsig
        self.imported_functions[dyncall] = dyncall
        if resbox is not None:
            varname = self.allocate_variable(resbox)
            self.emit(varname)
            self.emit("=")
        if callsig[0] == "f":
            self.emit("+")
        self.emit(dyncall)
        self.emit("(")
        self.emit_value(funcaddr)
        for i in xrange(len(arguments)):
            self.emit(",")
            self.emit_value(arguments[i])
        self.emit(")")
        if callsig[0] == "i":
            self.emit("|0")
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
            self.emit_statement("goto = 0|0")
            self.emit_statement("continue")
        else:
            # An external loop, need to goto via trampoline.
            self.emit_goto(token._asmjs_funcid, token._asmjs_label)

    def emit_goto(self, funcid, label=0):
        """Emit a goto to the funcid and label.

        Essentially we just encode the funcid and label into a single integer
        and return it.  We expect to be running inside some sort of trampoline
        that can actually execute the goto.
        """
        assert funcid < 2**24
        assert label < 0xFF
        next_call = (funcid << 8) | label
        self.emit_statement("return %d" % (next_call,))

    def emit_exit(self):
        """Emit an immediate return from the function."""
        self.emit("return 0;\n")

    def emit_comment(self, msg):
        self.emit("// %s\n" % (msg,))

    def emit_debug(self, msg, values=None):
        if DEBUGMODE:
            self.emit("log(\"%s\"" % (msg,))
            if values is not None:
                for value in values:
                    self.emit(",")
                    self.emit_value(value)
            self.emit(");\n")
