from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply function"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                # raw_vals.append(v.data)
                raw_vals.append(float(v.data))
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                # raw_vals.append(v)
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Addition function $f(x, y) = x + y$"""
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Addition function $f(x, y) = x + y$"""
        return float(d_output), float(d_output)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log function $f(x) = log(x)$"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Log function $f(x) = log(x)$"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Mul function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Mul function $f(x, y) = x * y$"""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Mul function $f(x, y) = x * y$
        Since df/da = b, df/db = a, the backward of mul with respect to d_output for a is df/da * d_output = b * d_output
        """
        (a, b) = ctx.saved_values
        return float(b * d_output), float(a * d_output)


class Inv(ScalarFunction):
    """Inv function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inv function"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Inv function"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Neg function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Neg function"""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Neg function"""
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid function"""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Sigmoid function"""
        (a,) = ctx.saved_values
        return operators.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    """Relu function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Relu function"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Relu function"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exp function"""
        exp_a = float(operators.exp(a))
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Exp function y$"""
        (a,) = ctx.saved_values
        return a * float(d_output)


class LT(ScalarFunction):
    """Lt function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Lt function"""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Lt function - there is no smooth change to propagate a gradient$"""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Eq function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Eq function"""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Eq function - there is no smooth change to propagate a gradient$"""
        return 0.0, 0.0
