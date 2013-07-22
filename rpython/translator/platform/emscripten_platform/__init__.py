import py, os

from rpython.translator.platform.posix import BasePosix, rpydir


pypy_root_dir = str(py.path.local(rpydir).join('..'))


class EmscriptenPlatform(BasePosix):
    """Platform for compiling to javascript via emscripten.

    Emscripten gives us the ability to compile from C to JavaScript while
    emulating a POSIX-like environment.  There are some subtlties, but it's
    mostly just a matter of using the right CC, CFLAGS, etc on top of a
    standard posixy build process.
    """

    name = "emscripten"

    exe_ext = 'js'
    DEFAULT_CC = 'emcc'
    standalone_only = []
    link_flags = ()
    cflags = [
      # Misc helpful/sensible flags.
      "-s", "DISABLE_EXCEPTION_CATCHING=1",
      "-s", "GC_SUPPORT=0",
      # Optimizations!
      # These are things that we've found to work OK with the generated code.
      # Try switching them off if the resulting javascript mis-behaves.
      "-O2",
      "-s", "FORCE_ALIGNED_MEMORY=1",
      # This prevents llvm optimization from throwing stuff away.
      # XXX TODO: probably there's a more nuanced way to achieve this...
      "-s", "EXPORT_ALL=1",
      # Growable memory is not compatible with asm.js.
      # Set it to the largest value that seems to be supported by nodejs.
      # XXX TODO: figure out a better memory-size story.
      # XXX TODO: automatically limit the GC to this much memory.
      # XXX TODO: ensure that pypy GC can detect when this runs out.
      #"-s", "ALLOW_MEMORY_GROWTH=1",
      "-s", "TOTAL_MEMORY=536870912",
      # For compiling with JIT.
      "--js-library", os.path.join(pypy_root_dir, "rpython/jit/backend/asmjs/library_jit.js"),
      # Extra sanity-checking.
      # Enable these if things go wrong.
      #"-s", "ASSERTIONS=1",
      #"-s", "SAFE_HEAP=1",
      #"-s", "CORRUPTION_CHECK=1",
      #"-s", "CHECK_HEAP_ALIGN=1",
      #"-s", "CHECK_OVERFLOWS=1",
      #"-s", "CHECK_SIGNED_OVERFLOWS=1",
    ]
    link_flags = cflags

    def execute(self, executable, args=None, *rest):
        # The generated file is just javascript, so it's not executable.
        # Instead we arrange for it to be run with nodejs.
        if args is None:
            args = []
        args = [str(executable)] + args
        executable = "node"
        return super(EmscriptenPlatform, self).execute(executable, args, *rest)

    def include_dirs_for_libffi(self):
        # libffi not supported; maybe we can hack around it?
        # For now we let the code read some defns out of the standard header.
        return [os.path.dirname(__file__)]

    def library_dirs_for_libffi(self):
        # libffi not supported; maybe we can hack around it?
        return []
