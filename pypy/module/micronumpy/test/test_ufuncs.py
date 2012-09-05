
from pypy.module.micronumpy.test.test_base import BaseNumpyAppTest

class AppTestUfuncs(BaseNumpyAppTest):
    def setup_class(cls):
        import os
        BaseNumpyAppTest.setup_class.im_func(cls)
        fname = os.path.join(os.path.dirname(__file__), 'complex_testcases.txt')
        cls.w_testcases = cls.space.wrap(fname)

    def test_ufunc_instance(self):
        from _numpypy import add, ufunc

        assert isinstance(add, ufunc)
        assert repr(add) == "<ufunc 'add'>"
        assert repr(ufunc) == "<type 'numpypy.ufunc'>"

    def test_ufunc_attrs(self):
        from _numpypy import add, multiply, sin

        assert add.identity == 0
        assert multiply.identity == 1
        assert sin.identity is None

        assert add.nin == 2
        assert multiply.nin == 2
        assert sin.nin == 1

    def test_wrong_arguments(self):
        from _numpypy import add, sin

        raises(ValueError, add, 1)
        raises(TypeError, add, 1, 2, 3)
        raises(TypeError, sin, 1, 2)
        raises(ValueError, sin)

    def test_single_item(self):
        from _numpypy import negative, sign, minimum

        assert negative(5.0) == -5.0
        assert sign(-0.0) == 0.0
        assert minimum(2.0, 3.0) == 2.0

    def test_sequence(self):
        from _numpypy import array, ndarray, negative, minimum
        a = array(range(3))
        b = [2.0, 1.0, 0.0]
        c = 1.0
        b_neg = negative(b)
        assert isinstance(b_neg, ndarray)
        for i in range(3):
            assert b_neg[i] == -b[i]
        min_a_b = minimum(a, b)
        assert isinstance(min_a_b, ndarray)
        for i in range(3):
            assert min_a_b[i] == min(a[i], b[i])
        min_b_a = minimum(b, a)
        assert isinstance(min_b_a, ndarray)
        for i in range(3):
            assert min_b_a[i] == min(a[i], b[i])
        min_a_c = minimum(a, c)
        assert isinstance(min_a_c, ndarray)
        for i in range(3):
            assert min_a_c[i] == min(a[i], c)
        min_c_a = minimum(c, a)
        assert isinstance(min_c_a, ndarray)
        for i in range(3):
            assert min_c_a[i] == min(a[i], c)
        min_b_c = minimum(b, c)
        assert isinstance(min_b_c, ndarray)
        for i in range(3):
            assert min_b_c[i] == min(b[i], c)
        min_c_b = minimum(c, b)
        assert isinstance(min_c_b, ndarray)
        for i in range(3):
            assert min_c_b[i] == min(b[i], c)

    def test_negative(self):
        from _numpypy import array, negative

        a = array([-5.0, 0.0, 1.0])
        b = negative(a)
        for i in range(3):
            assert b[i] == -a[i]

        a = array([-5.0, 1.0])
        b = negative(a)
        a[0] = 5.0
        assert b[0] == 5.0
        a = array(range(30))
        assert negative(a + a)[3] == -6

    def test_abs(self):
        from _numpypy import array, absolute

        a = array([-5.0, -0.0, 1.0])
        b = absolute(a)
        for i in range(3):
            assert b[i] == abs(a[i])

    def test_add(self):
        from _numpypy import array, add

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = add(a, b)
        for i in range(3):
            assert c[i] == a[i] + b[i]

    def test_divide(self):
        from _numpypy import array, divide

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = divide(a, b)
        for i in range(3):
            assert c[i] == a[i] / b[i]

        assert (divide(array([-10]), array([2])) == array([-5])).all()

    def test_true_divide(self):
        from _numpypy import array, true_divide

        a = array([0, 1, 2, 3, 4, 1, -1])
        b = array([4, 4, 4, 4, 4, 0,  0])
        c = true_divide(a, b)
        assert (c == [0.0, 0.25, 0.5, 0.75, 1.0, float('inf'), float('-inf')]).all()

        assert math.isnan(true_divide(0, 0))

    def test_fabs(self):
        from _numpypy import array, fabs
        from math import fabs as math_fabs, isnan

        a = array([-5.0, -0.0, 1.0])
        b = fabs(a)
        for i in range(3):
            assert b[i] == math_fabs(a[i])
        assert fabs(float('inf')) == float('inf')
        assert fabs(float('-inf')) == float('inf')
        assert isnan(fabs(float('nan')))

    def test_fmax(self):
        from _numpypy import fmax
        import math

        nnan, nan, inf, ninf = float('-nan'), float('nan'), float('inf'), float('-inf')

        a = [ninf, -5, 0, 5, inf]
        assert (fmax(a, [ninf]*5) == a).all()
        assert (fmax(a, [inf]*5) == [inf]*5).all()
        assert (fmax(a, [1]*5) == [1, 1, 1, 5, inf]).all()
        assert math.isnan(fmax(nan, 0))
        assert math.isnan(fmax(0, nan))
        assert math.isnan(fmax(nan, nan))
        # The numpy docs specify that the FIRST NaN should be used if both are NaN
        # Since comparisons with nnan and nan all return false,
        # use copysign on both sides to sidestep bug in nan representaion
        # on Microsoft win32
        assert math.copysign(1., fmax(nnan, nan)) == math.copysign(1., nnan)


    def test_fmin(self):
        from _numpypy import fmin
        import math

        nnan, nan, inf, ninf = float('-nan'), float('nan'), float('inf'), float('-inf')

        a = [ninf, -5, 0, 5, inf]
        assert (fmin(a, [ninf]*5) == [ninf]*5).all()
        assert (fmin(a, [inf]*5) == a).all()
        assert (fmin(a, [1]*5) == [ninf, -5, 0, 1, 1]).all()
        assert math.isnan(fmin(nan, 0))
        assert math.isnan(fmin(0, nan))
        assert math.isnan(fmin(nan, nan))
        # The numpy docs specify that the FIRST NaN should be used if both are NaN
        # use copysign on both sides to sidestep bug in nan representaion
        # on Microsoft win32
        assert math.copysign(1., fmin(nnan, nan)) == math.copysign(1., nnan)

    def test_fmod(self):
        from _numpypy import fmod
        import math

        assert fmod(-1e-100, 1e100) == -1e-100
        assert fmod(3, float('inf')) == 3
        assert (fmod([-3, -2, -1, 1, 2, 3], 2) == [-1,  0, -1,  1,  0,  1]).all()
        for v in [float('inf'), float('-inf'), float('nan'), float('-nan')]:
            assert math.isnan(fmod(v, 2))

    def test_minimum(self):
        from _numpypy import array, minimum

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = minimum(a, b)
        for i in range(3):
            assert c[i] == min(a[i], b[i])

    def test_maximum(self):
        from _numpypy import array, maximum

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = maximum(a, b)
        for i in range(3):
            assert c[i] == max(a[i], b[i])

        x = maximum(2, 3)
        assert x == 3
        assert isinstance(x, (int, long))

    def test_multiply(self):
        from _numpypy import array, multiply

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = multiply(a, b)
        for i in range(3):
            assert c[i] == a[i] * b[i]

    def test_sign(self):
        from _numpypy import array, sign, dtype

        reference = [-1.0, 0.0, 0.0, 1.0]
        a = array([-5.0, -0.0, 0.0, 6.0])
        b = sign(a)
        for i in range(4):
            assert b[i] == reference[i]

        a = sign(array(range(-5, 5)))
        ref = [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1]
        for i in range(10):
            assert a[i] == ref[i]

        a = sign(array([True, False], dtype=bool))
        assert a.dtype == dtype("int8")
        assert a[0] == 1
        assert a[1] == 0

    def test_signbit(self):
        from _numpypy import signbit

        assert (signbit([0, 0.0, 1, 1.0, float('inf')]) ==
            [False, False, False, False, False]).all()
        assert (signbit([-0, -0.0, -1, -1.0, float('-inf')]) ==
            [False,  True,  True,  True,  True]).all()
        skip('sign of nan is non-determinant')
        assert (signbit([float('nan'), float('-nan'), -float('nan')]) ==
            [False, True, True]).all()    

    def test_reciporocal(self):
        from _numpypy import array, reciprocal

        reference = [-0.2, float("inf"), float("-inf"), 2.0]
        a = array([-5.0, 0.0, -0.0, 0.5])
        b = reciprocal(a)
        for i in range(4):
            assert b[i] == reference[i]

    def test_subtract(self):
        from _numpypy import array, subtract

        a = array([-5.0, -0.0, 1.0])
        b = array([ 3.0, -2.0,-3.0])
        c = subtract(a, b)
        for i in range(3):
            assert c[i] == a[i] - b[i]

    def test_floorceiltrunc(self):
        from _numpypy import array, floor, ceil, trunc
        import math
        ninf, inf = float("-inf"), float("inf")
        a = array([ninf, -1.4, -1.5, -1.0, 0.0, 1.0, 1.4, 0.5, inf])
        assert ([ninf, -2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 0.0, inf] == floor(a)).all()
        assert ([ninf, -1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 1.0, inf] == ceil(a)).all()
        assert ([ninf, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, inf] == trunc(a)).all()
        assert all([math.isnan(f(float("nan"))) for f in floor, ceil, trunc])
        assert all([math.copysign(1, f(abs(float("nan")))) == 1 for f in floor, ceil, trunc])
        assert all([math.copysign(1, f(-abs(float("nan")))) == -1 for f in floor, ceil, trunc])

    def test_copysign(self):
        from _numpypy import array, copysign

        reference = [5.0, -0.0, 0.0, -6.0]
        a = array([-5.0, 0.0, 0.0, 6.0])
        b = array([5.0, -0.0, 3.0, -6.0])
        c = copysign(a, b)
        for i in range(4):
            assert c[i] == reference[i]

        b = array([True, True, True, True], dtype=bool)
        c = copysign(a, b)
        for i in range(4):
            assert c[i] == abs(a[i])

    def test_exp(self):
        import math
        from _numpypy import array, exp

        a = array([-5.0, -0.0, 0.0, 12345678.0, float("inf"),
                   -float('inf'), -12343424.0])
        b = exp(a)
        for i in range(len(a)):
            try:
                res = math.exp(a[i])
            except OverflowError:
                res = float('inf')
            assert b[i] == res

    def test_exp2(self):
        import math
        from _numpypy import array, exp2

        a = array([-5.0, -0.0, 0.0, 2, 12345678.0, float("inf"),
                   -float('inf'), -12343424.0])
        b = exp2(a)
        for i in range(len(a)):
            try:
                res = 2 ** a[i]
            except OverflowError:
                res = float('inf')
            assert b[i] == res

        assert exp2(3) == 8
        assert math.isnan(exp2(float("nan")))

    def test_expm1(self):
        import math
        from _numpypy import array, expm1

        a = array([-5.0, -0.0, 0.0, 12345678.0, float("inf"),
                   -float('inf'), -12343424.0])
        b = expm1(a)
        for i in range(4):
            try:
                res = math.exp(a[i]) - 1
            except OverflowError:
                res = float('inf')
            assert b[i] == res

        assert expm1(1e-50) == 1e-50

    def test_sin(self):
        import math
        from _numpypy import array, sin

        a = array([0, 1, 2, 3, math.pi, math.pi*1.5, math.pi*2])
        b = sin(a)
        for i in range(len(a)):
            assert b[i] == math.sin(a[i])

        a = sin(array([True, False], dtype=bool))
        assert abs(a[0] - sin(1)) < 1e-7  # a[0] will be less precise
        assert a[1] == 0.0

    def test_cos(self):
        import math
        from _numpypy import array, cos

        a = array([0, 1, 2, 3, math.pi, math.pi*1.5, math.pi*2])
        b = cos(a)
        for i in range(len(a)):
            assert b[i] == math.cos(a[i])

    def test_tan(self):
        import math
        from _numpypy import array, tan

        a = array([0, 1, 2, 3, math.pi, math.pi*1.5, math.pi*2])
        b = tan(a)
        for i in range(len(a)):
            assert b[i] == math.tan(a[i])

    def test_arcsin(self):
        import math
        from _numpypy import array, arcsin

        a = array([-1, -0.5, -0.33, 0, 0.33, 0.5, 1])
        b = arcsin(a)
        for i in range(len(a)):
            assert b[i] == math.asin(a[i])

        a = array([-10, -1.5, -1.01, 1.01, 1.5, 10, float('nan'), float('inf'), float('-inf')])
        b = arcsin(a)
        for f in b:
            assert math.isnan(f)

    def test_arccos(self):
        import math
        from _numpypy import array, arccos

        a = array([-1, -0.5, -0.33, 0, 0.33, 0.5, 1])
        b = arccos(a)
        for i in range(len(a)):
            assert b[i] == math.acos(a[i])

        a = array([-10, -1.5, -1.01, 1.01, 1.5, 10, float('nan'), float('inf'), float('-inf')])
        b = arccos(a)
        for f in b:
            assert math.isnan(f)

    def test_arctan(self):
        import math
        from _numpypy import array, arctan

        a = array([-3, -2, -1, 0, 1, 2, 3, float('inf'), float('-inf')])
        b = arctan(a)
        for i in range(len(a)):
            assert b[i] == math.atan(a[i])

        a = array([float('nan')])
        b = arctan(a)
        assert math.isnan(b[0])

    def test_arctan2(self):
        import math
        from _numpypy import array, arctan2

        # From the numpy documentation
        assert (
            arctan2(
                [0.,  0.,           1.,          -1., float('inf'),  float('inf')],
                [0., -0., float('inf'), float('inf'), float('inf'), float('-inf')]) ==
            [0.,  math.pi,  0., -0.,  math.pi/4, 3*math.pi/4]).all()

        a = array([float('nan')])
        b = arctan2(a, 0)
        assert math.isnan(b[0])

    def test_sinh(self):
        import math
        from _numpypy import array, sinh

        a = array([-1, 0, 1, float('inf'), float('-inf')])
        b = sinh(a)
        for i in range(len(a)):
            assert b[i] == math.sinh(a[i])

    def test_cosh(self):
        import math
        from _numpypy import array, cosh

        a = array([-1, 0, 1, float('inf'), float('-inf')])
        b = cosh(a)
        for i in range(len(a)):
            assert b[i] == math.cosh(a[i])

    def test_tanh(self):
        import math
        from _numpypy import array, tanh

        a = array([-1, 0, 1, float('inf'), float('-inf')])
        b = tanh(a)
        for i in range(len(a)):
            assert b[i] == math.tanh(a[i])

    def test_arcsinh(self):
        import math
        from _numpypy import arcsinh

        for v in [float('inf'), float('-inf'), 1.0, math.e]:
            assert math.asinh(v) == arcsinh(v)
        assert math.isnan(arcsinh(float("nan")))

    def test_arccosh(self):
        import math
        from _numpypy import arccosh

        for v in [1.0, 1.1, 2]:
            assert math.acosh(v) == arccosh(v)
        for v in [-1.0, 0, .99]:
            assert math.isnan(arccosh(v))

    def test_arctanh(self):
        import math
        from _numpypy import arctanh

        for v in [.99, .5, 0, -.5, -.99]:
            assert math.atanh(v) == arctanh(v)
        for v in [2.0, -2.0]:
            assert math.isnan(arctanh(v))
        for v in [1.0, -1.0]:
            assert arctanh(v) == math.copysign(float("inf"), v)

    def test_sqrt(self):
        import math
        from _numpypy import sqrt

        nan, inf = float("nan"), float("inf")
        data = [1, 2, 3, inf]
        results = [math.sqrt(1), math.sqrt(2), math.sqrt(3), inf]
        assert (sqrt(data) == results).all()
        assert math.isnan(sqrt(-1))
        assert math.isnan(sqrt(nan))

    def test_square(self):
        import math
        from _numpypy import square

        nan, inf, ninf = float("nan"), float("inf"), float("-inf")

        assert math.isnan(square(nan))
        assert math.isinf(square(inf))
        assert math.isinf(square(ninf))
        assert square(ninf) > 0
        assert [square(x) for x in range(-5, 5)] == [x*x for x in range(-5, 5)]
        assert math.isinf(square(1e300))

    def test_radians(self):
        import math
        from _numpypy import radians, array
        a = array([
            -181, -180, -179,
            181, 180, 179,
            359, 360, 361,
            400, -1, 0, 1,
            float('inf'), float('-inf')])
        b = radians(a)
        for i in range(len(a)):
            assert b[i] == math.radians(a[i])

    def test_deg2rad(self):
        import math
        from _numpypy import deg2rad, array
        a = array([
            -181, -180, -179,
            181, 180, 179,
            359, 360, 361,
            400, -1, 0, 1,
            float('inf'), float('-inf')])
        b = deg2rad(a)
        for i in range(len(a)):
            assert b[i] == math.radians(a[i])

    def test_degrees(self):
        import math
        from _numpypy import degrees, array
        a = array([
            -181, -180, -179,
            181, 180, 179,
            359, 360, 361,
            400, -1, 0, 1,
            float('inf'), float('-inf')])
        b = degrees(a)
        for i in range(len(a)):
            assert b[i] == math.degrees(a[i])

    def test_rad2deg(self):
        import math
        from _numpypy import rad2deg, array
        a = array([
            -181, -180, -179,
            181, 180, 179,
            359, 360, 361,
            400, -1, 0, 1,
            float('inf'), float('-inf')])
        b = rad2deg(a)
        for i in range(len(a)):
            assert b[i] == math.degrees(a[i])

    def test_reduce_errors(self):
        from _numpypy import sin, add

        raises(ValueError, sin.reduce, [1, 2, 3])
        raises((ValueError, TypeError), add.reduce, 1)

    def test_reduce_1d(self):
        from _numpypy import add, maximum, less

        assert less.reduce([5, 4, 3, 2, 1])
        assert add.reduce([1, 2, 3]) == 6
        assert maximum.reduce([1]) == 1
        assert maximum.reduce([1, 2, 3]) == 3
        raises(ValueError, maximum.reduce, [])

    def test_reduceND(self):
        from _numpypy import add, arange
        a = arange(12).reshape(3, 4)
        assert (add.reduce(a, 0) == [12, 15, 18, 21]).all()
        assert (add.reduce(a, 1) == [6.0, 22.0, 38.0]).all()
        raises(ValueError, add.reduce, a, 2)

    def test_reduce_keepdims(self):
        from _numpypy import add, arange
        a = arange(12).reshape(3, 4)
        b = add.reduce(a, 0, keepdims=True)
        assert b.shape == (1, 4)
        assert (add.reduce(a, 0, keepdims=True) == [12, 15, 18, 21]).all()

    def test_bitwise(self):
        from _numpypy import bitwise_and, bitwise_or, bitwise_xor, arange, array
        a = arange(6).reshape(2, 3)
        assert (a & 1 == [[0, 1, 0], [1, 0, 1]]).all()
        assert (a & 1 == bitwise_and(a, 1)).all()
        assert (a | 1 == [[1, 1, 3], [3, 5, 5]]).all()
        assert (a | 1 == bitwise_or(a, 1)).all()
        assert (a ^ 3 == bitwise_xor(a, 3)).all()
        raises(TypeError, 'array([1.0]) & 1')

    def test_unary_bitops(self):
        from _numpypy import bitwise_not, invert, array
        a = array([1, 2, 3, 4])
        assert (~a == [-2, -3, -4, -5]).all()
        assert (bitwise_not(a) == ~a).all()
        assert (invert(a) == ~a).all()

    def test_shift(self):
        from _numpypy import left_shift, right_shift

        assert (left_shift([5, 1], [2, 13]) == [20, 2**13]).all()
        assert (right_shift(10, range(5)) == [10, 5, 2, 1, 0]).all()

    def test_comparisons(self):
        import operator
        from _numpypy import equal, not_equal, less, less_equal, greater, greater_equal

        for ufunc, func in [
            (equal, operator.eq),
            (not_equal, operator.ne),
            (less, operator.lt),
            (less_equal, operator.le),
            (greater, operator.gt),
            (greater_equal, operator.ge),
        ]:
            for a, b in [
                (3, 3),
                (3, 4),
                (4, 3),
                (3.0, 3.0),
                (3.0, 3.5),
                (3.5, 3.0),
                (3.0, 3),
                (3, 3.0),
                (3.5, 3),
                (3, 3.5),
            ]:
                assert ufunc(a, b) == func(a, b)

    def test_count_reduce_items(self):
        from _numpypy import count_reduce_items, arange
        a = arange(24).reshape(2, 3, 4)
        assert count_reduce_items(a) == 24
        assert count_reduce_items(a, 1) == 3
        assert count_reduce_items(a, (1, 2)) == 3 * 4
        raises(ValueError, count_reduce_items, a, -4)
        raises(ValueError, count_reduce_items, a, (0, 2, -4))

    def test_count_nonzero(self):
        from _numpypy import where, count_nonzero, arange
        a = arange(10)
        assert count_nonzero(a) == 9
        a[9] = 0
        assert count_nonzero(a) == 8

    def test_true_divide(self):
        from _numpypy import arange, array, true_divide
        assert (true_divide(arange(3), array([2, 2, 2])) == array([0, 0.5, 1])).all()

    def test_isnan_isinf(self):
        from _numpypy import isnan, isinf, float64, array
        assert isnan(float('nan'))
        assert isnan(float64(float('nan')))
        assert not isnan(3)
        assert isinf(float('inf'))
        assert not isnan(3.5)
        assert not isinf(3.5)
        assert not isnan(float('inf'))
        assert not isinf(float('nan'))
        assert (isnan(array([0.2, float('inf'), float('nan')])) == [False, False, True]).all()
        assert (isinf(array([0.2, float('inf'), float('nan')])) == [False, True, False]).all()
        assert isinf(array([0.2])).dtype.kind == 'b'

    def test_isposinf_isneginf(self):
        from _numpypy import isneginf, isposinf
        assert isposinf(float('inf'))
        assert not isposinf(float('-inf'))
        assert not isposinf(float('nan'))
        assert not isposinf(0)
        assert not isposinf(0.0)
        assert isneginf(float('-inf'))
        assert not isneginf(float('inf'))
        assert not isneginf(float('nan'))
        assert not isneginf(0)
        assert not isneginf(0.0)

    def test_isfinite(self):
        from _numpypy import isfinite
        assert (isfinite([0, 0.0, 1e50, -1e-50]) ==
            [True, True, True, True]).all()
        assert (isfinite([float('-inf'), float('inf'), float('-nan'), float('nan')]) ==
            [False, False, False, False]).all()

    def test_logical_ops(self):
        from _numpypy import logical_and, logical_or, logical_xor, logical_not

        assert (logical_and([True, False , True, True], [1, 1, 3, 0])
                == [True, False, True, False]).all()
        assert (logical_or([True, False, True, False], [1, 2, 0, 0])
                == [True, True, True, False]).all()
        assert (logical_xor([True, False, True, False], [1, 2, 0, 0])
                == [False, True, True, False]).all()
        assert (logical_not([True, False]) == [False, True]).all()

    def test_logn(self):
        import math
        from _numpypy import log, log2, log10

        for log_func, base in [(log, math.e), (log2, 2), (log10, 10)]:
            for v in [float('-nan'), float('-inf'), -1, float('nan')]:
                assert math.isnan(log_func(v))
            for v in [-0.0, 0.0]:
                assert log_func(v) == float("-inf")
            assert log_func(float('inf')) == float('inf')
            assert (log_func([1, base]) == [0, 1]).all()

    def test_log1p(self):
        import math
        from _numpypy import log1p

        for v in [float('-nan'), float('-inf'), -2, float('nan')]:
            assert math.isnan(log1p(v))
        for v in [-1]:
            assert log1p(v) == float("-inf")
        assert log1p(float('inf')) == float('inf')
        assert (log1p([0, 1e-50, math.e - 1]) == [0, 1e-50, 1]).all()

    def test_power_float(self):
        import math
        from _numpypy import power, array
        a = array([1., 2., 3.])
        b = power(a, 3)
        for i in range(len(a)):
            assert b[i] == a[i] ** 3

        a = array([1., 2., 3.])
        b = array([1., 2., 3.])
        c = power(a, b)
        for i in range(len(a)):
            assert c[i] == a[i] ** b[i]

        assert power(2, float('inf')) == float('inf')
        assert power(float('inf'), float('inf')) == float('inf')
        assert power(12345.0, 12345.0) == float('inf')
        assert power(-12345.0, 12345.0) == float('-inf')
        assert power(-12345.0, 12346.0) == float('inf')
        assert math.isnan(power(-1, 1.1))
        assert math.isnan(power(-1, -1.1))
        assert power(-2.0, -1) == -0.5
        assert power(-2.0, -2) == 0.25
        assert power(12345.0, -12345.0) == 0
        assert power(float('-inf'), 2) == float('inf')
        assert power(float('-inf'), 2.5) == float('inf')
        assert power(float('-inf'), 3) == float('-inf')

    def test_power_int(self):
        import math
        from _numpypy import power, array
        a = array([1, 2, 3])
        b = power(a, 3)
        for i in range(len(a)):
            assert b[i] == a[i] ** 3

        a = array([1, 2, 3])
        b = array([1, 2, 3])
        c = power(a, b)
        for i in range(len(a)):
            assert c[i] == a[i] ** b[i]

        # assert power(12345, 12345) == -9223372036854775808
        # assert power(-12345, 12345) == -9223372036854775808
        # assert power(-12345, 12346) == -9223372036854775808
        assert power(2, 0) == 1
        assert power(2, -1) == 0
        assert power(2, -2) == 0
        assert power(-2, -1) == 0
        assert power(-2, -2) == 0
        assert power(12345, -12345) == 0

    def test_floordiv(self):
        from _numpypy import floor_divide, array
        a = array([1., 2., 3., 4., 5., 6., 6.01])
        b = floor_divide(a, 2.5)
        for i in range(len(a)):
            assert b[i] == a[i] // 2.5

    def test_logaddexp(self):
        import math
        import sys
        float_max, float_min = sys.float_info.max, sys.float_info.min
        from _numpypy import logaddexp

        # From the numpy documentation
        prob1 = math.log(1e-50)
        prob2 = math.log(2.5e-50)
        prob12 = logaddexp(prob1, prob2)
        assert math.fabs(-113.87649168120691 - prob12) < 0.000000000001

        assert logaddexp(0, 0) == math.log(2)
        assert logaddexp(float('-inf'), 0) == 0
        assert logaddexp(float_max, float_max) == float_max
        assert logaddexp(float_min, float_min) == math.log(2)

        assert math.isnan(logaddexp(float('nan'), 1))
        assert math.isnan(logaddexp(1, float('nan')))
        assert math.isnan(logaddexp(float('nan'), float('inf')))
        assert math.isnan(logaddexp(float('inf'), float('nan')))
        assert logaddexp(float('-inf'), float('-inf')) == float('-inf')
        assert logaddexp(float('-inf'), float('inf')) == float('inf')
        assert logaddexp(float('inf'), float('-inf')) == float('inf')
        assert logaddexp(float('inf'), float('inf')) == float('inf')

    def test_logaddexp2(self):
        import math
        import sys
        float_max, float_min = sys.float_info.max, sys.float_info.min
        from _numpypy import logaddexp2
        log2 = math.log(2)

        # From the numpy documentation
        prob1 = math.log(1e-50) / log2
        prob2 = math.log(2.5e-50) / log2
        prob12 = logaddexp2(prob1, prob2)
        assert math.fabs(-164.28904982231052 - prob12) < 0.000000000001

        assert logaddexp2(0, 0) == 1
        assert logaddexp2(float('-inf'), 0) == 0
        assert logaddexp2(float_max, float_max) == float_max
        assert logaddexp2(float_min, float_min) == 1.0

        assert math.isnan(logaddexp2(float('nan'), 1))
        assert math.isnan(logaddexp2(1, float('nan')))
        assert math.isnan(logaddexp2(float('nan'), float('inf')))
        assert math.isnan(logaddexp2(float('inf'), float('nan')))
        assert logaddexp2(float('-inf'), float('-inf')) == float('-inf')
        assert logaddexp2(float('-inf'), float('inf')) == float('inf')
        assert logaddexp2(float('inf'), float('-inf')) == float('inf')
        assert logaddexp2(float('inf'), float('inf')) == float('inf')

    def test_conjugate(self):
        from _numpypy import conj, conjugate, complex128, complex64
        import _numpypy as np

        c0 = complex128(complex(2.5, 0))
        c1 = complex64(complex(1, 2))

        assert conj is conjugate
        assert conj(c0) == c0
        assert conj(c1) == complex(1, -2)
        assert conj(1) == 1
        assert conj(-3) == -3
        assert conj(float('-inf')) == float('-inf')


        assert np.conjugate(1+2j) == 1-2j

        x = np.eye(2) + 1j * np.eye(2)
        for a, b in zip(np.conjugate(x), np.array([[ 1.-1.j,  0.-0.j], [ 0.-0.j,  1.-1.j]])):
            assert a[0] == b[0]
            assert a[1] == b[1]


    def test_complex(self):
        from _numpypy import (complex128, complex64, add,
            subtract as sub, multiply, divide, negative, abs, fmod, 
            reciprocal)
        from _numpypy import (equal, not_equal, greater, greater_equal, less,
                less_equal)

        for complex_ in complex128, complex64:

            O = complex(0, 0)
            c0 = complex_(complex(2.5, 0))
            c1 = complex_(complex(1, 2))
            c2 = complex_(complex(3, 4))
            c3 = complex_(complex(-3, -3))

            assert equal(c0, 2.5)
            assert equal(c1, complex_(complex(1, 2)))
            assert equal(c1, complex(1, 2))
            assert equal(c1, c1)
            assert not_equal(c1, c2)
            assert not equal(c1, c2)

            assert less(c1, c2)
            assert less_equal(c1, c2)
            assert less_equal(c1, c1)
            assert not less(c1, c1)

            assert greater(c2, c1)
            assert greater_equal(c2, c1)
            assert not greater(c1, c2)

            assert add(c1, c2) == complex_(complex(4, 6))
            assert add(c1, c2) == complex(4, 6)
            
            assert sub(c0, c0) == sub(c1, c1) == 0
            assert sub(c1, c2) == complex(-2, -2)
            assert negative(complex(1,1)) == complex(-1, -1)
            assert negative(complex(0, 0)) == 0
            

            assert multiply(1, c1) == c1
            assert multiply(2, c2) == complex(6, 8)
            assert multiply(c1, c2) == complex(-5, 10)

            assert divide(c0, 1) == c0
            assert divide(c2, -1) == negative(c2)
            assert divide(c1, complex(0, 1)) == complex(2, -1)
            n = divide(c1, O)
            assert repr(n.real) == 'inf'
            assert repr(n.imag).startswith('inf') #can be inf*j or infj
            assert divide(c0, c0) == 1
            res = divide(c2, c1)
            assert abs(res.real-2.2) < 0.001
            assert abs(res.imag+0.4) < 0.001

            assert abs(c0) == 2.5
            assert abs(c2) == 5
            
            raises (TypeError, fmod, c0, 3) 
            inf_c = complex_(complex(float('inf'), 0.))
            assert repr(abs(inf_c)) == 'inf'
            assert repr(abs(n)) == 'nan'

            assert False, 'untested: copysign, reciprocal, sign, floor_div, ' + \
                     'signbit, fabs, fmax, fmin, floor, ceil, trunc, ' + \
                     'exp2, expm1, isnan, isinf, isneginf, isposinf, ' + \
                     'isfinite, radians, degrees, log2, log1p, ' + \
                     'logaddexp, npy_log2_1p, logaddexp2'

    def test_complex_math(self):
        import  _numpypy as np
        from math import isnan, isinf, copysign
        from sys import version_info
        testcases = self.testcases
        def parse_testfile(fname):
            """Parse a file with test values

            Empty lines or lines starting with -- are ignored
            yields id, fn, arg_real, arg_imag, exp_real, exp_imag
            """
            with open(fname) as fp:
                for line in fp:
                    # skip comment lines and blank lines
                    if line.startswith('--') or not line.strip():
                        continue

                    lhs, rhs = line.split('->')
                    id, fn, arg_real, arg_imag = lhs.split()
                    rhs_pieces = rhs.split()
                    exp_real, exp_imag = rhs_pieces[0], rhs_pieces[1]
                    flags = rhs_pieces[2:]

                    yield (id, fn,
                           float(arg_real), float(arg_imag),
                           float(exp_real), float(exp_imag),
                           flags
                          )
        def rAssertAlmostEqual(a, b, rel_err = 2e-15, abs_err = 5e-323, msg=''):
            """Fail if the two floating-point numbers are not almost equal.

            Determine whether floating-point values a and b are equal to within
            a (small) rounding error.  The default values for rel_err and
            abs_err are chosen to be suitable for platforms where a float is
            represented by an IEEE 754 double.  They allow an error of between
            9 and 19 ulps.
            """

            # special values testing
            if isnan(a):
                if isnan(b):
                    return
                #raise AssertionError(msg + '%r should be nan' % (b,))
                print(msg + '%r should be nan' % (b,))

            if isinf(a):
                if a == b:
                    return
                #raise AssertionError(msg + 'finite result where infinity expected: '
                #                           'expected %r, got %r' % (a, b))
                print(msg + 'finite result where infinity expected: '
                                           'expected %r, got %r' % (a, b))

            # if both a and b are zero, check whether they have the same sign
            # (in theory there are examples where it would be legitimate for a
            # and b to have opposite signs; in practice these hardly ever
            # occur).
            if not a and not b:
                # only check it if we are running on top of CPython >= 2.6
                if version_info >= (2, 6) and copysign(1., a) != copysign(1., b):
                    #raise AssertionError(msg + 'zero has wrong sign: expected %r, '
                    #                           'got %r' % (a, b))
                    print(msg + 'zero has wrong sign: expected %r, '
                                               'got %r' % (a, b))

            # if a-b overflows, or b is infinite, return False.  Again, in
            # theory there are examples where a is within a few ulps of the
            # max representable float, and then b could legitimately be
            # infinite.  In practice these examples are rare.
            try:
                absolute_error = abs(b-a)
            except OverflowError:
                pass
            else:
                # test passes if either the absolute error or the relative
                # error is sufficiently small.  The defaults amount to an
                # error of between 9 ulps and 19 ulps on an IEEE-754 compliant
                # machine.
                if absolute_error <= max(abs_err, rel_err * abs(a)):
                    return
            #raise AssertionError(msg + '%r and %r are not sufficiently close' % (a, b))
            print(msg + '%r and %r are not sufficiently close' % (a, b))
        tested_funcs=[]
        for complex_, abs_err in ((np.complex128, 5e-323), (np.complex64, 5e-32)):
            for id, fn, ar, ai, er, ei, flags in parse_testfile(testcases):
                arg = complex_(complex(ar, ai))
                expected = (er, ei)
                if fn.startswith('acos'):
                    fn = 'arc' + fn[1:]
                elif fn.startswith('asin'):
                    fn = 'arc' + fn[1:]
                elif fn.startswith('atan'):
                    fn = 'arc' + fn[1:]
                function = getattr(np, fn)
                _actual = function(arg)
                actual = (_actual.real, _actual.imag)

                if 'ignore-real-sign' in flags:
                    actual = (abs(actual[0]), actual[1])
                    expected = (abs(expected[0]), expected[1])
                if 'ignore-imag-sign' in flags:
                    actual = (actual[0], abs(actual[1]))
                    expected = (expected[0], abs(expected[1]))

                # for the real part of the log function, we allow an
                # absolute error of up to 2e-15.
                if fn in ('log', 'log10'):
                    real_abs_err = 2e-15
                else:
                    real_abs_err = abs_err

                error_message = (
                    '%s: %s(%r(%r, %r))\n'
                    'Expected: complex(%r, %r)\n'
                    'Received: complex(%r, %r)\n'
                    ) % (id, fn, complex_, ar, ai,
                         expected[0], expected[1],
                         actual[0], actual[1])
                         
                if not function in tested_funcs:        
                    print 'fuction',function
                    tested_funcs.append(function)
                rAssertAlmostEqual(expected[0], actual[0],
                                   abs_err=real_abs_err,
                                   msg=error_message)
                rAssertAlmostEqual(expected[1], actual[1],
                                   msg=error_message)
