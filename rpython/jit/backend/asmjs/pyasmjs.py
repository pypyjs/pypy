"""

A toy asmjs-to-python converter.

This module provides the ability to compile asmjs source code into a native
python function offering equivalent functionality.  It is mostly useful for
testing purposes.

It translates only a subset of asmjs, and does not guarantee to throw errors
on non-asmjs code.  It would be nice to be a lot stricter here because it's
useful for testing, but this is enough to get started for now.

Some known limitations:

  * Newlines and semicolons are expected to be used roughly C-style (or
    jshint-style) so code that depends on automatic semicolon insertion may
    fail to compile, or may compile incorrectly.

  * Minimal typechecking is done, so code that is not technically valid asmjs
    (but is still valid javascript!) may still get successfully compiled by
    this module.

  * Javascript code that uses python builtins as identifiers will probably
    fail to compile; we don't do that so there's no need to work around it yet.

  * Comments are not supported, because they would complicate the parsing.

  * Function tables are not supported, because the JIT doesn't use them.

"""

import re
import struct
import collections


def load_asmjs(jssource, stdlib=None, foreign=None, heap=None):
    """Compile asmjs source code into an equivalent python function."""
    func = compile_asmjs(jssource)
    if stdlib is None:
        stdlib = STDLIB
    if foreign is None:
        foreign = FOREIGN
    if heap is None:
        heap = collections.defaultdict(lambda: 0)
    return func(stdlib, foreign, heap)


def compile_asmjs(jssource):
    """Compile asmjs source code into an equivalent python function."""
    print jssource
    print "=-=-=-=-=-=-=-=-=-=-="
    jstokens = tokenize(jssource)
    pysource = "".join(translate_module(jstokens))
    print pysource
    ns = {}
    exec pysource in {}, ns
    assert len(ns) == 1
    return ns.itervalues().next()


# Here is an ad-hoc tokenize for the subset of javascript that we need.

TOKEN_PATTERNS = {
  "IF": "if",
  "ELSE": "else",
  "VAR": "var",
  "RETURN": "return",
  "VOID": "void",
  "FOR": "for",
  "WHILE": "while",
  "BREAK": "break",
  "FUNCTION": "function",
  "SWITCH": "switch",
  "NEW": "new",
  "COMMA": ",",
  "SEMICOLON": ";",
  "DOT": ".",
  "COLON": ":",
  "PLUS": "+",
  "MINUS": "-",
  "INVERT": "~",
  "NOT": "!",
  "MULTIPLY": "*",
  "DIVIDE": "/",
  "MODULO": "%",
  "BINARYOR": "|",
  "BINARYAND": "&",
  "BINARYXOR": "^",
  "LEFTSHIFT": "<<",
  "RIGHTSHIFT": ">>",
  "URIGHTSHIFT": ">>>",
  "LESSTHAN": "<",
  "LESSTHANEQ": "<=",
  "GREATERTHAN": ">",
  "GREATERTHANEQ": ">=",
  "EQUAL": "==",
  "NOTEQUAL": "!=",
  "ASSIGN": "=",
  "INTCAST": "|0",
  "LEFTPAREN": "(",
  "RIGHTPAREN": ")",
  "LEFTBRACE": "{",
  "RIGHTBRACE": "}",
  "LEFTSQBRACKET": "[",
  "RIGHTSQBRACKET": "]",
  "ASMJSDECL": re.compile("('|\")use asm('|\")"),
  "NUMBER": re.compile("[0-9]+(.[0-9_]+)?"),
  "IDENTIFIER": re.compile("[a-zA-Z][a-zA-Z0-9_]*")
}


class Token(object):

    name = None
    pattern = None

    def __init__(self, value):
        self.value = value

    @classmethod
    def match(cls, string):
        match = cls.pattern.match(string)
        if match is None:
            return None
        return cls(match.group(0))


# Process each type of token into a class, and generate a list of
# token classes to be tried in order, based on the length of their pattern.
# This lets us do some simple greedy matching to disambiguate a couple of
# overlapping patterns e.g. ">>" and ">>>".

TOKEN_NAMES = TOKEN_PATTERNS.keys()
TOKEN_NAMES.sort(key=lambda v: len(TOKEN_PATTERNS[v]) if isinstance(TOKEN_PATTERNS[v], str) else 0, reverse=True)

TOKEN_CLASSES = []
for token in TOKEN_NAMES:
    pattern = TOKEN_PATTERNS[token]
    if isinstance(pattern, str):
       if not pattern.isalnum():
           pattern = "".join("\\" + c for c in pattern)
       pattern = re.compile(pattern)
    name = "T_" + token.upper()
    globals()[name] = type(name, (Token,), {"name": token, "pattern": pattern})
    TOKEN_CLASSES.append(globals()[name])


def _tokenize(source):
    """Generator producing a stream of tokens from the given source string."""
    source = source.lstrip()
    while source:
        for tokencls in TOKEN_CLASSES:
            token = tokencls.match(source)
            if token is not None:
                source = source[len(token.value):].lstrip()
                yield token
                break
        else:
            raise ValueError("Invalid asmjs source: %r" % (source[:20],))


class tokenize(object):
    """A little class for handily walking a stream of tokens."""

    def __init__(self, source):
        self._tokenstream = iter(_tokenize(source))
        self._next_token = None

    def peek(self):
        if self._next_token is None:
            try:
                self._next_token = self._tokenstream.next()
            except StopIteration:
                pass
        return self._next_token

    def consume(self, cls=None):
        next_token = self.peek()
        if cls is not None and not isinstance(next_token, cls):
            msg = "Tokenization error: expected %s, found %s"
            raise ValueError(msg % (cls, next_token,))
        self._next_token = None
        return next_token

    def maybe_consume(self, cls):
        next_token = self.peek()
        if not isinstance(next_token, cls):
            return None
        self._next_token = None
        return next_token

    def finish(self):
        if self.peek() is not None:
            raise ValueError("tokens remain after finish")


# Here we have a simple one-pass translator that takes a stream of tokens
# and yields chunks of python sourcecode.  The statement structure and
# identifiers stay the same


def translate_module(tokens):
    # Outer function definition, including "use asm" declaration.
    tokens.consume(T_FUNCTION)
    yield "def "
    name = tokens.maybe_consume(T_IDENTIFIER)
    if name is not None:
        yield name.value
    else:
        yield "PYASMJS"
    tokens.consume(T_LEFTPAREN)
    yield "("
    while not tokens.maybe_consume(T_RIGHTPAREN):
        yield tokens.consume().value
    yield "):\n"
    tokens.consume(T_LEFTBRACE)
    tokens.consume(T_ASMJSDECL)
    tokens.consume(T_SEMICOLON)
    # Global variable definitions.
    # Fortunately the dotted-lookup syntax works the same as in python,
    # so we just have to cope with "new" and the numeric cast operators.
    while tokens.maybe_consume(T_VAR):
        yield "    "
        yield tokens.consume(T_IDENTIFIER).value
        tokens.consume(T_ASSIGN)
        tokens.maybe_consume(T_NEW)
        yield " = "
        cast = ""
        expr = ""
        if tokens.maybe_consume(T_PLUS):
            cast = "float"
        while not isinstance(tokens.peek(), (T_SEMICOLON, T_INTCAST)):
            expr += tokens.consume().value
        if tokens.maybe_consume(T_INTCAST):
            assert not cast
            cast = "int"
        yield "%s(%s)\n" % (cast, expr)
        tokens.consume(T_SEMICOLON)
    # Function definitions.
    # Each is a function prelude followed by some indented statements.
    while tokens.maybe_consume(T_FUNCTION):
        yield "    def "
        yield tokens.consume(T_IDENTIFIER).value
        tokens.consume(T_LEFTPAREN)
        yield "("
        while not tokens.maybe_consume(T_RIGHTPAREN):
            yield tokens.consume().value
        yield "):\n"
        for chunk in translate_function_body(tokens, "        "):
            yield chunk
    # Return expression.
    # It can be either a single function, or a dict of functions.
    tokens.consume(T_RETURN)
    yield "    return "
    if tokens.maybe_consume(T_LEFTBRACE):
        yield "{'"
        while not tokens.maybe_consume(T_RIGHTBRACE):
            yield tokens.consume(T_IDENTIFIER).value
            tokens.consume(T_COLON)
            yield "':"
            yield tokens.consume(T_IDENTIFIER).value
            if tokens.maybe_consume(T_COMMA):
                yield ","
        yield "}"
    else:
        yield tokens.consume(T_IDENTIFIER).value
    tokens.consume(T_SEMICOLON)
    yield "\n"
    tokens.consume(T_RIGHTBRACE)
    tokens.finish()


def translate_function_body(tokens, indent):
    yield indent
    yield "raise NotImplementedError"
    yield "\n"
    tokens.consume(T_LEFTBRACE)
    brace_count = 1
    while brace_count > 0:
        next_token = tokens.consume()
        if isinstance(next_token, T_LEFTBRACE):
            brace_count += 1
        elif isinstance(next_token, T_RIGHTBRACE):
            brace_count -= 1


# Here we have a re-implementation of the asmjs standard library.
# Of particular interest are the ArrayView types which provide different
# views onto the underlying shared heap.

def makeHeapView(format):
    """Given native-endian struct pack format, build a heapview class."""

    itemsize = struct.calcsize(format)

    class _HeapView(object):

        def __init__(self, heap):
            self.heap = heap

        def __getitem__(self, addr):
            data = ""
            for i in xrange(itemsize):
                data += chr(self.heap[(itemsize * addr) + i])
            return struct.unpack(format, data)[0]

        def __setitem__(self, addr, value):
            data = struct.pack(format, value)
            for i in xrange(len(data)):
                self.heap[(itemsize * addr) + i] = ord(data[i])

    return _HeapView


class STDLIB(object):

    Int8Array = makeHeapView("b")
    Int16Array = makeHeapView("h")
    Int32Array = makeHeapView("i")
    Uint8Array = makeHeapView("B")
    Uint16Array = makeHeapView("H")
    Uint32Array = makeHeapView("I")
    Float32Array = makeHeapView("f")
    Float64Array = makeHeapView("d")

    class Math(object):

        @staticmethod
        def imul(a, b):
            return (int(a) * int(b)) & 0xFFFFFFFF

STDLIB = STDLIB()


# Here we have a basic implementation of the foreign-function object as it
# might look when generated by emscripten.  The various dynCall_XXX methods
# and generated automatically when accessed.

class FOREIGN(object):

    def __getattr__(self, name):
        if name.startswith("dynCall_"):
            def dynCall(*args):
                raise NotImplementedError
            return dynCall
        raise AttributeError

FOREIGN = FOREIGN()
