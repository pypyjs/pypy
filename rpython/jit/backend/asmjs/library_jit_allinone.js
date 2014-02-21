//
//  emscripten helper library for JIT-compiling asmjs functions.
//
//  This is a re-implementation of library_jit.js that compiles all jitted
//  functions into a single asmjs module.  Yes, that means it recompiles all
//  jitted code whenever any jitted code has changed.  This is an experiment
//  in reducing function call overhead.
//

var LibraryJIT = {

  //  JIT-compile a single function.
  //
  //  The input argument must be the heap address of a string containing
  //  asmjs source code, defining a single function that takes two integer
  //  arguments and returns an integer.
  //
  //  The source will be loaded, compiled, and linked with the main Module.
  //  An opaque integer "function id" will be returned, which can be passed
  //  to jitInvoke to invoke the newly-compiled function.
  //
  jitCompile__deps: ['jitReserve', 'jitRecompile'],
  jitCompile: function(addr) {
    addr = addr|0;
    var id = _jitReserve()|0;
    return _jitRecompile(id, addr);
  },

  //  Reserve a function id for later use.
  //
  //  Rather than creating a new function straight away, this simply allocates
  //  and returns a new function id.  The code can be filled in later by a
  //  call to jitRecompile().
  //
  //  Attempts to invoke a not-yet-defined function will immediately return
  //  zero.
  //
  jitReserve: function() {
    if (!Module._jitCompiledFunctions) {
      // We never allocate a function ID of zero.
      // In theory we could use zero to report compilation failure, but
      // the try-catch may prevent optimization of this function.
      Module._jitCompiledFunctions = [null];
    }
    var id = Module._jitCompiledFunctions.length;
    Module._jitCompiledFunctions[id] = null;
    return id;
  },


  //  Check if a compiled function exists with given id.
  //
  jitExists: function(id) {
    id = id|0;
    if (!Module._jitCompiledFunctions) {
      return 0;
    }
    if (Module._jitCompiledFunctions[id]) {
      return 1;
    }
    return 0;
  },

  //  Re-compile a JIT-compiled function with new source code.
  //
  //  The input arguments are an existing function id and the heap address
  //  of a string containing asmjs source code.
  //
  //  The source will be loaded, compiled, and linked with the main Module.
  //  An opaque integer "function id" will be returned, which can be passed
  //  to jitInvoke to invoke the newly-compiled function.
  //  
  jitRecompile: function(id, addr) {
    id = id|0;
    addr = addr|0;
    // Read js source from the heap, as a C-style string.
    var sourceChars = [];
    var i = addr;
    while (HEAP8[i] != 0) {
      sourceChars.push(String.fromCharCode(HEAP8[i]));
      i++;
    }
    Module._jitCompiledFunctions[id] = sourceChars.join("");
    // Re-compile everything into a great bit asmjs module.
    modsource = "function M(stdlib, foreign, heap){\n"+
        '"use asm";\n' +
        'var HI8 = new stdlib.Int8Array(heap);\n' + 
        'var HI16 = new stdlib.Int16Array(heap);\n' + 
        'var HI32 = new stdlib.Int32Array(heap);\n' + 
        'var HU8 = new stdlib.Uint8Array(heap);\n' + 
        'var HU16 = new stdlib.Uint16Array(heap);\n' + 
        'var HU32 = new stdlib.Uint32Array(heap);\n' + 
        'var HF32 = new stdlib.Float32Array(heap);\n' + 
        'var HF64 = new stdlib.Float64Array(heap);\n' + 
        'var imul = stdlib.Math.imul;\n' +
        'var sqrt = stdlib.Math.sqrt;\n' +
        'var tempDoublePtr = foreign.tempDoublePtr|0;\n' +
        'var abort = foreign.abort;\n';
    var doneImports = {'tempDoublePtr': 1, 'abort': 1, 'jitInvoke': 1};
    for (var id=1; id < Module._jitCompiledFunctions.length; id++) {
      var importRE = expr = /var ([a-zA-Z0-0_]+) = foreign\.[a-zA-Z0-9_]+;/g;
      var funcsrc = Module._jitCompiledFunctions[id];
      var importRes;
      while ((importRes = importRE.exec(funcsrc)) !== null) {
        if (!doneImports[importRes[1]]) {
          modsource += importRes[0] + "\n";
          doneImports[importRes[1]] = 1;
        }
      }
    }
    var funcTableSize = Module._jitCompiledFunctions.length;
    funcTableSize = Math.ceil(Math.log(funcTableSize) / Math.log(2))
    funcTableSize = Math.pow(2, funcTableSize);
    for (var id=1; id < Module._jitCompiledFunctions.length; id++) {
      var funcsrc = Module._jitCompiledFunctions[id];
      var bodystart = funcsrc.indexOf("function ", 10)
      var bodystart = funcsrc.indexOf("(", bodystart)
      var bodyend = funcsrc.lastIndexOf("}")
      var bodyend = funcsrc.lastIndexOf("}", bodyend - 2)
      var funcbody = funcsrc.substring(bodystart, bodyend + 1)
      // Avoid redirecting through jitInvoke helper.
      funcbody = funcbody.replace(/jitInvoke\(([0-9]+),/g, function(m, id) {
        return "FUNCS[" + id + " & " + (funcTableSize-1) + "]("
      })
      modsource += "function F" + id + funcbody + "\n";
    }
      modsource += "function jitInvoke(id, frame, label) {\n" +
          "id = id|0;\n" +
          "frame = frame|0;\n" +
          "label = label|0;\n" +
          "return FUNCS[id & " + (funcTableSize-1) + "](frame|0,label|0)|0;\n" +
          "}\n" +
          "function jitAbort(frame,label) {\n" +
          "frame = frame|0;\n" +
          "label = label|0;\n" +
          "return abort()|0;\n" +
          "}\n" +
          "var FUNCS = [jitAbort"
      var id = 1;
      for (; id < Module._jitCompiledFunctions.length; id++) {
        modsource += ",F" + id;
      }
      for (; id < funcTableSize; id++) {
        modsource += ",jitAbort";
      }
      modsource += "];\n" +
          "return jitInvoke;\n" +
          "}"
    var mkfunc = new Function("return (" + modsource + ")");
    var stdlib = {
      "Math": Math,
      "Int8Array": Int8Array,
      "Int16Array": Int16Array,
      "Int32Array": Int32Array,
      "Uint8Array": Uint8Array,
      "Uint16Array": Uint16Array,
      "Uint32Array": Uint32Array,
      "Float32Array": Float32Array,
      "Float64Array": Float64Array
    };
    if (typeof Module.tempDoublePtr === "undefined") {
      if (typeof tempDoublePtr === "undefined") {
          throw "NO TEMP DOUBLE PTR";
      }
      Module.tempDoublePtr = tempDoublePtr;
    }
    Module._jitCompiledInvoke = mkfunc()(stdlib, Module, buffer);
    return id
  },

  // Invoke a JIT-compiled function.
  //
  // All JIT-compiled functions accept two integer arguments and produce an
  // integer result.  You'll probably want to treat these like a void* to pass
  // around data, but that's up to you.
  //
  // If you pass an id that does not have compiled code associated with it,
  // it will produce a return value of zero.
  //
  jitInvoke: function(id, frame, label) {
    id = id|0;
    label = label|0;
    frame = frame|0;
    var src = Module._jitCompiledFunctions[id];
    if (src && Module._jitCompiledInvoke) {
        return Module._jitCompiledInvoke(id, frame, label)|0;
    } else {
        return 0|0;
    }
  },

  // Free a JIT-compiled function.
  //
  jitFree: function(id) {
    id = id|0;
    Module._jitCompiledFunctions[id] = null;
  }
}

mergeInto(LibraryManager.library, LibraryJIT);
