import py, os, sys

import rpython
from rpython.translator.platform import Platform, CompilationError
from rpython.tool.runsubprocess import run_subprocess

from rpython.translator.platform.posix import BasePosix, GnuMakefile


rpydir = str(py.path.local(rpython.__file__).join('..'))


class EmscriptenPlatform(BasePosix):
    """ Platform for compiling to javascript via emscripten."""

    name = "emscripten"

    # XXX TODO: custom -s SHELL_FILE

    cflags = [
      #"-O1",  # enabling asm.js causes much error
      "--llvm-opts", "1",  # enabling these opts seems to corrupt memory somehow
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
    ]
    
    def __init__(self, cc=None):
        self.cc = cc or 'emcc'
        # XXX TODO: check that emcc is available and runnable
        # XXX TODO: check that nodejs is available and runnable

    def compile(self, cfilenames, eci, outputfilename=None, standalone=True):
        if outputfilename is None:
            outputfilename = py.path.local(cfilenames[0]).new(ext='.emcc')
        else:
            outputfilename = py.path.local(outputfilename)
        args = list(self.cflags)
        args.extend(str(c) for c in cfilenames)
        if eci.include_dirs:
            for include_dir in eci.include_dirs:
                args.extend(['-I', str(include_dir)])

        # Always ask emcc to generate a .js file, since it automatically
        # detects output format based on filename extension.
        rawoutputfile = str(outputfilename) + '.js'
        args.extend(['-o', rawoutputfile])

        print "RUNNING", self.cc, args
        returncode, stdout, stderr = run_subprocess(self.cc, args)
        if returncode:
            raise CompilationError('', stderr)

        # Add a #! line to the generated file, so its executable.
        with open(str(outputfilename), 'w') as f_out:
            f_out.write('#!/usr/bin/env node\n')
            with open(rawoutputfile, 'r') as f_in:
                f_out.write(f_in.read())
        os.chmod(str(outputfilename), 0700)

        return outputfilename

    def gen_makefile(self, cfiles, eci, exe_name=None, path=None,
                     shared=False):
        cfiles = self._all_cfiles(cfiles, eci)
        if path is None:
            path = cfiles[0].dirpath()
        if exe_name is None:
            exe_name = cfiles[0].new(ext='.js')
        else:
            exe_name = exe_name.new(ext='.js')
        m = GnuMakefile(path)
        m.exe_name = path.join(exe_name.basename)
        m.eci = eci

        rpypath = py.path.local(rpydir)

        def rpyrel(fpath):
            lpath = py.path.local(fpath)
            rel = lpath.relto(rpypath)
            if rel:
                return os.path.join('$(RPYDIR)', rel)
            m_dir = m.makefile_dir
            if m_dir == lpath:
                return '.'
            if m_dir.dirpath() == lpath:
                return '..'
            return fpath

        rel_cfiles = [m.pathrel(cfile) for cfile in cfiles]
        m.cfiles = rel_cfiles

        rel_includedirs = [rpyrel(incldir) for incldir in
                           self.preprocess_include_dirs(eci.include_dirs)]

        cflags = list(self.cflags)
        # Embed the pypy build directory in the JS.
        # This is needed when running the full interpreter.
        # XXX TODO: It would be great to do something more clever here.
        #cflags.extend(["--embed-file", os.path.dirname(rpydir)])

        m.comment("automatically generated makefile")
        definitions = [
            ('RPYDIR', '"%s"' % rpydir),
            ('TARGET', exe_name.basename),
            ('DEFAULT_TARGET', exe_name.basename),
            ('SOURCES', rel_cfiles),
            ('INCLUDEDIRS', self._includedirs(rel_includedirs)),
            ('CC', self.cc),
            ('CFLAGS', cflags),
            ]
        for args in definitions:
            m.definition(*args)
        rules = [
            ('all', '$(DEFAULT_TARGET)', []),
            ('$(TARGET)', '$(SOURCES)', '$(CC) $(CFLAGS) $(INCLUDEDIRS) -o $@ $(SOURCES)'),
            ]
        for rule in rules:
            m.rule(*rule)

        return m

    def include_dirs_for_libffi(self):
        raise NotImplementedError("libffi not supported")

    def library_dirs_for_libffi(self):
        raise NotImplementedError("libffi not supported")

