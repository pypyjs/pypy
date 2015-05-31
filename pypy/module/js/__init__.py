
from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):
    """\
This module provides access to the host javascript environment.  It can be
used to introspect objects and call functions in the host environment, and
to create callbacks allowing the host environment to call back into the pypy
interpreter.
    """

    def startup(self, space):
        from pypy.module.js import interp_js
        interp_js.getstate(space).startup(space)

    interpleveldefs = {
        # Type heirarchy, as python classes.
        'Value': 'interp_js.W_Value',
        'Undefined': 'interp_js.W_Undefined',
        'Boolean': 'interp_js.W_Boolean',
        'Number': 'interp_js.W_Number',
        'String': 'interp_js.W_String',
        'Object': 'interp_js.W_Object',
        'Array': 'interp_js.W_Array',
        'Function': 'interp_js.W_Function',
        'Method': 'interp_js.W_Method',
        # An Exception subclass to encapsulate javascript-level errors.
        'Error': 'space.fromcache(interp_js.State).w_jserror',
        # Standard constants, for easy access.
        'undefined': 'interp_js.undefined',
        'null': 'interp_js.null',
        'false': 'interp_js.false',
        'true': 'interp_js.true',
        'globals': 'space.fromcache(interp_js.State).w_globals',
        # Helper functions for things that don't map nicely to python.
        'convert': 'interp_js.convert',
        'eval': 'interp_js.eval',
        'bind': 'interp_js.bind',
        'new': 'interp_js.new',
        'equal': 'interp_js.equal',
        'instanceof': 'interp_js.instanceof',
        'urshift': 'interp_js.urshift',
        'uint32': 'interp_js.uint32',
        }

    appleveldefs = {
        }
