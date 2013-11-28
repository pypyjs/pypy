import py, os

from rpython.translator.platform.posix import BasePosix, rpydir, GnuMakefile
from rpython.translator.platform import log, _run_subprocess


pypy_root_dir = str(py.path.local(rpydir).join('..'))


def find_executable(filename, environ=None):
    """Find an executable by searching the current $PATH."""
    if environ is None:
        environ = os.environ
    path = environ.get("PATH", "/usr/local/bin:/usr/bin:/bin").split(":")
    for dirpath in path:
        dirpath = os.path.abspath(dirpath.strip())
        filepath = os.path.normpath(os.path.join(dirpath, filename))
        if os.path.exists(filepath):
            return filepath
    return None


class EmscriptenPlatform(BasePosix):
    """Platform for compiling to javascript via emscripten.

    Emscripten gives us the ability to compile from C to JavaScript while
    emulating a POSIX-like environment.  There are some subtlties, but it's
    mostly just a matter of using the right CC, CFLAGS, etc on top of a
    standard posixy build process.
    """

    name = "emscripten"

    exe_ext = 'js'
    so_ext = 'so'
    DEFAULT_CC = 'emcc'
    standalone_only = []
    shared_only = []

    # Environment variables required for proper compilation.
    # XXX TODO: it would be more reliable to set these via command-line
    extra_environ = {
        # PyPy allocates at wordsize boundaries, so we can't assume that
        # doubles are properly aligned.  This forces emscripten to take
        # more care when loading/storing doubles.
        # XXX TODO: fix pypy to do properly-aligned allocations
        "EMCC_LLVM_TARGET": "i386-pc-linux-gnu",
    }

    cflags = [
      # Misc helpful/sensible flags.
      #"-s", "VERBOSE=1",
      "-s", "DISABLE_EXCEPTION_CATCHING=1",
      "-s", "GC_SUPPORT=0",
      # Optimizations!
      # These are things that we've found to work OK with the generated code.
      # Try switching them off if the resulting javascript mis-behaves.
      "-O2",
      # Things to try:
      #    This *should* be ok, but needs extensive testing.
      #    "-s", "FUNCTION_POINTER_ALIGNMENT=1",
      #    This may be worth trying for a small speedup.
      #    "-s", "CORRECT_OVERFLOWS=0",
      #    This may be worth trying for a small speedup.
      #    "-s", "CORRECT_SIGNS=0",
      #    This may be worth trying for a small speedup.
      #    "-s", "ASSERTIONS=0",
      # Some parts of the JIT assume that a function is uniquely identified
      # by its pointer.  This makes it so, at the cost of a lot of extra
      # padding in the function type tables.
      "-s", "ALIASING_FUNCTION_POINTERS=0",
      # This prevents llvm optimization from throwing stuff away.
      # XXX TODO: probably there's a more nuanced way to achieve this...
      "-s", "EXPORT_ALL=1",
      # Growable memory is not compatible with asm.js yet.
      # Set it to the largest value that seems to be supported by nodejs.
      # XXX TODO: figure out a better memory-size story.
      # XXX TODO: automatically limit the GC to this much memory.
      # XXX TODO: ensure that pypy GC can detect when this runs out.
      "-s", "TOTAL_MEMORY=536870912",
      # Some dummy includes to convince things to compile properly.
      # XXX TODO: only include these when needed.
      "-I", os.path.dirname(__file__),
      # For compiling with JIT.
      # XXX TODO: only include this when jit is enabled.
      "--js-library", os.path.join(pypy_root_dir, "rpython/jit/backend/asmjs/library_jit.js"),
      # PyPy usually produces a couple of very large functions, particularly
      # with the JIT included.  Try to break them up a little.
      "-s", "OUTLINING_LIMIT=150000",
      # Extra sanity-checking.
      # Enable these if things go wrong.
      #"-s", "ASSERTIONS=1",
      #"-s", "SAFE_HEAP=1",
      #"-s", "CORRUPTION_CHECK=1",
      #"-s", "CHECK_HEAP_ALIGN=1",
      #"-s", "CHECK_OVERFLOWS=1",
      #"-s", "CHECK_SIGNED_OVERFLOWS=1",
    ]

    link_flags = cflags + [
      # This preserves sensible names in the generated JS.
      # Useful for debugging, but turn this off in production.
      "-g2",
      # Necessary for ctypes support.
      #"-s", "DLOPEN_SUPPORT=1",
      #"-s", "INCLUDE_FULL_LIBRARY=1",
      # XXX TODO: use DEFAULT_LIBRARY_FUNCS_TO_INCLUDE to force the ffi lib
    ]

    def __init__(self, *args, **kwds):
        os.environ.update(self.extra_environ)
        super(EmscriptenPlatform, self).__init__(*args, **kwds)

    def execute(self, executable, args=None, env=None, *a, **k):
        if env is None:
            os.environ.update(self.extra_environ)
        else:
            env.update(self.extra_environ)
        # The generated file is just javascript, so it's not executable.
        # Instead we arrange for it to be run with a javascript shell.
        if args is None:
            args = []
        if isinstance(args, basestring):
            args = str(executable) + " " + args
        else:
            args = [str(executable)] + args
        jsshell = find_executable("js")
        if jsshell is None:
            jsshell = find_executable("node")
            if jsshell is None:
                raise RuntimeError("Could not find javascript shell")
        super_cls = super(EmscriptenPlatform, self)
        return super_cls.execute(jsshell, args, env, *a, **k)

    def include_dirs_for_libffi(self):
        return []

    def library_dirs_for_libffi(self):
        return []

    def gen_makefile(self, *args, **kwds):
        m = super(EmscriptenPlatform, self).gen_makefile(*args, **kwds)
        # Embed parts of the build dir as files in the final executable.
        # This is necessary so the pypy interpreter can find its files,
        # but is useless for generic rpython apps.
        # XXX TODO: find a more elegant way to achieve this only when needed.
        # There's apparently a "VFS" system under development for emscripten
        # which might make the need for this go away.
        ldflags_def = m.lines[m.defs["LDFLAGS"]]
        ldflags_def.value.extend([
          "--embed-file", os.path.join(str(pypy_root_dir), "lib-python") + "@" + os.path.join(str(pypy_root_dir), "lib-python")[1:],
          "--embed-file", os.path.join(str(pypy_root_dir), "lib_pypy") + "@" + os.path.join(str(pypy_root_dir), "lib_pypy")[1:],
        #  "--embed-file", os.path.join(str(pypy_root_dir), "rpython") + "@" + os.path.join(str(pypy_root_dir), "rpython")[1:],
          "--embed-file", os.path.join(str(pypy_root_dir), "pytest.py") + "@" + os.path.join(str(pypy_root_dir), "pytest.py")[1:],
        ])
        return m 

    def execute_makefile(self, path_to_makefile, extra_opts=[]):
        if isinstance(path_to_makefile, GnuMakefile):
            path = path_to_makefile.makefile_dir
        else:
            path = path_to_makefile
        log.execute('make %s in %s' % (" ".join(extra_opts), path))
        env = os.environ.copy()
        returncode, stdout, stderr = _run_subprocess(
            self.make_cmd, ['-C', str(path)] + extra_opts, env=env)
        self._handle_error(returncode, stdout, stderr, path.join('make'))

    def _args_for_shared(self, args):
        return args
