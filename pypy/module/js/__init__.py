
from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):
    """\
This module provides access to the host javascript environment.  It can be
used to introspect objects and call functions in the host environment, and
to create callbacks allowing the host environment to call back into the pypy
interpreter.
    """

    interpleveldefs = {
        'Error': 'space.fromcache(interp_js.Cache).w_error',
        'Value': 'interp_js.W_Value',
        # XXX TODO: make these actual subclasses of W_Value
        # that we can use in lieu of 'typeof'.
        'Undefined': 'interp_js.Undefined',
        'Boolean': 'interp_js.Boolean',
        'Number': 'interp_js.Number',
        'String': 'interp_js.String',
        'Object': 'interp_js.Object',
        'Array': 'interp_js.Array',
        'Function': 'interp_js.Function',
        'undefined': 'interp_js.undefined',
        'null': 'interp_js.null',
        'false': 'interp_js.false',
        'true': 'interp_js.true',
        'globals': 'interp_js.globals',
        'eval': 'interp_js.eval',
        'new': 'interp_js.new',
        'instanceof': 'interp_js.instanceof',
        'urshift': 'interp_js.urshift',
        'uint32': 'interp_js.uint32',
        }

    appleveldefs = {
        }
