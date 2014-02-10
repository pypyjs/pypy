
import os
from rpython.rlib.objectmodel import we_are_translated
from rpython.rtyper.lltypesystem import lltype, rffi

from rpython.jit.backend.asmjs import jsvalue as jsval
from rpython.jit.backend.asmjs.arch import SANITYCHECK


class ASMJSFragment(object):

    def __init__(self, source, intvars, doublevars, functions):
        self.source = source
        self.all_intvars = intvars
        self.all_doublevars = doublevars
        self.imported_functions = functions


class ASMJSBuilder(object):
    """Class for building ASMJS assembly of a trace."""

    def __init__(self, cpu):
        self.cpu = cpu
        self.source_chunks = []
        self.all_intvars = {}
        self.all_doublevars = {}
        self.free_intvars = []
        self.free_doublevars = []
        self.imported_functions = {}

    def finish(self):
        return "".join(self._build_prelude() +
                       self.source_chunks +
                       self._build_epilog())

    def _build_prelude(self):
        chunks = []
        # Standard asmjs prelude stuff.
        chunks.append('function M(stdlib, foreign, heap){\n')
        chunks.append('"use asm";\n')
        chunks.append('var HI8 = new stdlib.Int8Array(heap);\n')
        chunks.append('var HI16 = new stdlib.Int16Array(heap);\n')
        chunks.append('var HI32 = new stdlib.Int32Array(heap);\n')
        chunks.append('var HU8 = new stdlib.Uint8Array(heap);\n')
        chunks.append('var HU16 = new stdlib.Uint16Array(heap);\n')
        chunks.append('var HU32 = new stdlib.Uint32Array(heap);\n')
        chunks.append('var HF32 = new stdlib.Float32Array(heap);\n')
        chunks.append('var HF64 = new stdlib.Float64Array(heap);\n')
        chunks.append('var imul = stdlib.Math.imul;\n')
        chunks.append('var sqrt = stdlib.Math.sqrt;\n')
        # Import any required names from the Module object.
        chunks.append('var tempDoublePtr = foreign.tempDoublePtr|0;\n')
        for funcname in self.imported_functions:
            chunks.append('var %s = foreign.%s;\n' % (funcname, funcname))
        # The function definition, including variable declarations.
        chunks.append('function F(label, frame){\n')
        chunks.append('label=label|0;\n')
        chunks.append('frame=frame|0;\n')
        for varname, init_int in self.all_intvars.iteritems():
            chunks.append("var %s=%d;\n" % (varname, init_int))
        for varname, init_double in self.all_doublevars.iteritems():
            assert "." in ("%f" % (init_double,))
            chunks.append("var %s=%f;\n" % (varname, init_double))
        return chunks

    def _build_epilog(self):
        chunks = []
        # The compiled function always returns frame when it's finished.
        chunks.append('return frame|0;\n')
        chunks.append('}\n')
        # Export the singleton compiled function.
        chunks.append('return F;\n')
        chunks.append('}\n')
        return chunks

    def allocate_intvar(self, num=-1):
        """Allocate a variable of type int.

        If there is a previously-freed variable of appropriate type then
        that variable is re-used; otherwise a fresh variable name is
        allocated and emitted.
        """
        if num < 0 and len(self.free_intvars) > 0:
            varname = self.free_intvars.pop()
        else:
            if num < 0:
                num = len(self.all_intvars)
                varname = "i%d" % (num,)
            else:
                varname = "i%d" % (num,)
                try:
                    self.free_intvars.remove(varname)
                except ValueError:
                    pass
            self.all_intvars[varname] = 0
        return jsval.IntVar(varname)

    def allocate_doublevar(self, num=-1):
        """Allocate a variable of type double.

        If there is a previously-freed variable of appropriate type then
        that variable is re-used; otherwise a fresh variable name is
        allocated and emitted.
        """
        if num < 0 and len(self.free_doublevars) > 0:
            varname = self.free_doublevars.pop()
        else:
            if num < 0:
                num = len(self.all_doublevars)
                varname = "f%d" % (num,)
            else:
                varname = "f%d" % (num,)
                try:
                    self.free_doublevars.remove(varname)
                except ValueError:
                    pass
            self.all_doublevars[varname] = 0.0
        return jsval.DoubleVar(varname)

    def free_intvar(self, var):
        """Free up the given int variable for future re-use."""
        assert isinstance(var, jsval.IntVar)
        self.free_intvars.append(var.varname)

    def free_doublevar(self, var):
        """Free up the given double variable for future re-use."""
        assert isinstance(var, jsval.DoubleVar)
        self.free_doublevars.append(var.varname)

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

    def emit_value(self, val):
        """Emit a reference to a ASMJSValue in the generated source."""
        if val is None:
            self.emit_value(jsval.zero)
        elif isinstance(val, jsval.ASMJSValue):
            val.emit_value(self)
        elif isinstance(val, jsval.ConstInt):
            intval = (rffi.cast(lltype.Signed, val.getint()))
            self.emit(str(intval))
        elif isinstance(val, jsval.ConstPtr):
            refval = (rffi.cast(lltype.Signed, val.getref_base()))
            self.emit(str(refval))
        elif isinstance(val, jsval.ConstFloat):
            # RPython doesn't have %r format for float repr.
            # XXX TODO: how to do this with no loss of precsion?
            if not we_are_translated():
                self.emit("%r" % (val.getfloat(),))
            else:
                self.emit("%f" % (val.getfloat(),))
        else:
            os.write(2, "Unknown js value type: %s" % (val,))
            raise RuntimeError("Unknown js value type: %s" % (val,))

    def emit_expr(self, val):
        self.emit_value(val)
        self.emit(";\n")

    def emit_assignment(self, target, value):
        """Emit an assignment of a value to the given target."""
        assert isinstance(target, jsval.Variable)
        self.emit(target.varname)
        self.emit("=")
        if jsval.istype(target, jsval.Double):
            if not jsval.istype(value, jsval.Double):
                self.emit("+")
        self.emit("(")
        self.emit_value(value)
        self.emit(")")
        if jsval.istype(target, jsval.Int):
            if not jsval.istype(value, jsval.Int):
                self.emit("|0")
        self.emit(";\n")

    def emit_load(self, target, addr, typ):
        """Emit a typed load operation from the heap into a target.

        Given a result-storing target, a heap address, and an optional heap
        type, this method generates a generic "load from heap" instruction of
        the following form:

            target = HEAPVIEW[(addr) >> shift];

        """
        assert isinstance(addr, jsval.AbstractValue)
        # For doubles, we can't guarantee that the data is aligned.
        # Read it as two 32bit ints into a properly aligned chunk.
        #if typ is jsval.Float64:
        #    tempaddr = self.allocate_intvar()
        #    self.emit_assignment(tempaddr, addr)
        #    addr = jsval.tempDoublePtr
        #    pt1 = jsval.HeapData(jsval.Int32, tempaddr)
        #    self.emit_store(pt1, addr, jsval.Int32)
        #    pt2 = jsval.HeapData(jsval.Int32, jsval.Plus(tempaddr, jsval.word))
        #    self.emit_store(pt2, jsval.Plus(addr, jsval.word), jsval.Int32)
        #    self.free_intvar(tempaddr)
        #self.emit_assert(jsval.Equal(jsval.Mod(addr, jsval.ConstInt(typ.size)), jsval.zero), 'unaligned read', [addr, jsval.ConstInt(typ.size)])
        # Now we can do the actual load.
        assert isinstance(target, jsval.Variable)
        self.emit(target.varname)
        self.emit("=")
        if jsval.istype(target, jsval.Double):
            self.emit("+")
        self.emit("(")
        self.emit(typ.heap_name)
        self.emit("[(")
        self.emit_value(addr)
        self.emit(")>>")
        self.emit(str(typ.shift))
        self.emit("])")
        if jsval.istype(target, jsval.Int):
            self.emit("|0")
        self.emit(";\n")

    def emit_store(self, value, addr, typ):
        """Emit a typed store operation of a value into the heap.

        Given a jsvalue, a heap address, and a HeapType, this method generates
        a generic "store to heap" instruction of the following form:

            HEAPVIEW[(addr) >> shift] = value;

        """
        assert isinstance(addr, jsval.AbstractValue)
        # For doubles, we can't guarantee that the data is aligned.
        # We have to store it into a properly aligned chunk, then
        # copy to final destination as two 32-bit ints.
        #tempaddr = None
        #if typ is jsval.Float64:
        #    tempaddr = self.allocate_intvar()
        #    self.emit_assignment(tempaddr, addr)
        #    addr = jsval.tempDoublePtr
        #self.emit_assert(jsval.Equal(jsval.Mod(addr, jsval.ConstInt(typ.size)), jsval.zero), 'unaligned write', [addr, jsval.ConstInt(typ.size)])
        self.emit(typ.heap_name)
        self.emit("[(")
        self.emit_value(addr)
        self.emit(")>>")
        self.emit(str(typ.shift))
        self.emit("]=")
        self.emit_value(value)
        self.emit(";\n")
        #if typ is jsval.Float64:
        #    pt1 = jsval.HeapData(jsval.Int32, addr)
        #    self.emit_store(pt1, tempaddr, jsval.Int32)
        #    pt2 = jsval.HeapData(jsval.Int32, jsval.Plus(addr, jsval.word))
        #    self.emit_store(pt2, jsval.Plus(tempaddr, jsval.word), jsval.Int32)
        #    self.free_intvar(tempaddr)

    def emit_continue_loop(self):
        self.emit_statement("continue")

    def emit_exit(self):
        """Emit an immediate return from the function."""
        self.emit("return frame|0;\n")

    def emit_assert(self, check, msg="ASSERTION FAILED", args=None):
        if SANITYCHECK:
            with self.emit_if_block(jsval.UNot(check)):
                self.emit_debug(msg, args)
                self.emit_expr(jsval.CallFunc("abort", []))

    def emit_comment(self, msg):
        if SANITYCHECK:
            self.emit("// %s\n" % (msg,))

    def emit_debug(self, msg, values=None):
        if SANITYCHECK:
            if we_are_translated():
                self.emit("print(\"%s\"" % (msg,))
            else:
                self.emit("log(\"%s\"" % (msg,))
            if values:
                for i in xrange(len(values)):
                    self.emit(",")
                    self.emit_value(values[i])
            self.emit(");\n")

    def emit_if_block(self, test):
        return ctx_if_block(self, test)

    def emit_else_block(self):
        return ctx_else_block(self)

    def emit_while_block(self, test):
        return ctx_while_block(self, test)

    def emit_switch_block(self, value):
        return ctx_switch_block(self, value)

    def emit_case_block(self, value):
        return ctx_case_block(self, value)

    def emit_fragment(self, fragment):
        self.source_chunks.append(fragment.source)
        self.all_intvars.update(fragment.all_intvars)
        self.all_doublevars.update(fragment.all_doublevars)
        self.imported_functions.update(fragment.imported_functions)

    def capture_fragment(self):
        fragment = ASMJSFragment("".join(self.source_chunks),
                                 self.all_intvars, self.all_doublevars,
                                 self.imported_functions)
        del self.source_chunks[:]
        return fragment


class ctx_if_block(object):

    def __init__(self, bldr, test):
        self.bldr = bldr
        if not jsval.istype(test, jsval.Int):
            test = jsval.IntCast(test)
        self.test = test

    def __enter__(self):
        self.bldr.emit("if(")
        self.bldr.emit_value(self.test)
        self.bldr.emit("){\n")
        return self.bldr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bldr.emit("}\n")


class ctx_else_block(object):

    def __init__(self, bldr):
        self.bldr = bldr

    def __enter__(self):
        self.bldr.emit("else {\n")
        return self.bldr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bldr.emit("}\n")


class ctx_while_block(object):

    def __init__(self, bldr, test):
        self.bldr = bldr
        if not jsval.istype(test, jsval.Int):
            test = jsval.IntCast(test)
        self.test = test

    def __enter__(self):
        self.bldr.emit("while(")
        self.bldr.emit_value(self.test)
        self.bldr.emit("){\n")
        return self.bldr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bldr.emit("}\n")


class ctx_switch_block(object):

    def __init__(self, bldr, value):
        self.bldr = bldr
        if not jsval.istype(value, jsval.Signed):
            value = jsval.SignedCast(value)
        self.value = value

    def __enter__(self):
        self.bldr.emit("switch(")
        self.bldr.emit_value(self.value)
        self.bldr.emit("){\n")
        return self.bldr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bldr.emit("default:\n")
        self.bldr.emit_exit()
        self.bldr.emit("}\n")


class ctx_case_block(object):

    def __init__(self, bldr, value):
        self.bldr = bldr
        if not jsval.istype(value, jsval.Signed):
            value = jsval.SignedCast(value)
        self.value = value

    def __enter__(self):
        self.bldr.emit("case (")
        self.bldr.emit_value(self.value)
        self.bldr.emit("): {\n")
        return self.bldr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bldr.emit("break;\n}\n")
