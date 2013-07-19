import py

from rpython.translator.platform.posix import BasePosix, rpydir


pypy_root_dir = str(py.path.local(rpydir).join('..'))


class EmscriptenPlatform(BasePosix):
    """Platform for compiling to javascript via emscripten.

    Emscripten gives us the ability to compile from C to JavaScript while
    emulating a POSIX-like environment.  There are some subtlties, but it's
    usually just a matter of using the right CC, CFLAGS, etc on top of a
    standard posixy build process.
    """

    name = "emscripten"

    exe_ext = 'js'
    DEFAULT_CC = 'emcc'
    standalone_only = []
    link_flags = ()
    # XXX TODO: custom -s SHELL_FILE
    cflags = [
      #"-O1",  # enabling asm.js causes much error
      #"--llvm-opts", "1",  # enabling these opts seems to corrupt memory somehow
      "-s", "VERBOSE=1",
      "-s", "ASSERTIONS=1",
      "-s", "SAFE_HEAP=1",
      "-s", "CORRUPTION_CHECK=1",
      "-s", "DISABLE_EXCEPTION_CATCHING=1",
      "-s", "GC_SUPPORT=0",
      "-s", "ASM_JS=0",
      #"-s", "ALLOW_MEMORY_GROWTH=1",
      "-s", "CORRECT_SIGNS=1",
      "-s", "RELOOP=1",
      "-s", "CHECK_HEAP_ALIGN=1",
      "-s", "CHECK_OVERFLOWS=1",
      "-s", "CHECK_SIGNED_OVERFLOWS=1",
      "--closure", "0"
      # XXX TODO:  embed the pypy root directory in the generated code?
    ]

    # XXX TODO: the generated file is unexecutable javascript.
    # So we have to run it under node.
    # It would be better to make it a #!/usr/bin/env node file.
    def execute(self, executable, args=None, *extra):
        if args is None:
            args = []
        args = [str(executable)] + args
        executable = "node"
        return super(EmscriptenPlatform, self).execute(executable, args, *extra)

    def include_dirs_for_libffi(self):
        raise NotImplementedError("libffi not supported")

    def library_dirs_for_libffi(self):
        raise NotImplementedError("libffi not supported")
