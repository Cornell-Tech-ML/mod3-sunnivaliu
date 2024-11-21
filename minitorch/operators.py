"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Compute mul

    Args:x (float): The input number.

    Returns:float: The output of mul.
    """
    return float(x * y)


def id(x: float) -> float:
    """Compute id

    Args:x (float): The input number.

    Returns:float: The output of id.
    """
    return float(x)


def add(x: float, y: float) -> float:
    """Compute add

    Args:x (float): The input number.

    Returns:float: The output of add.
    """
    return float(x + y)


def neg(x: float) -> float:
    """Compute neg

    Args:x (float): The input number.

    Returns:float: The output of neg.
    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """Compute lt

    Args:x (float): The input number.

    Returns:float: The output of lt.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compute eq

    Args:x (float): The input number.

    Returns:float: The output of eq.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Compute max

    Args:x (float): The input number.

    Returns:float: The output of max.
    """
    return float(x) if x > y else float(y)


def is_close(x: float, y: float) -> float:
    """Compute is_close

    Args:x (float): The input number.

    Returns:float: The output of is_close.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute sigmoid

    Args:x (float): The input number.

    Returns:float: The output of sigmoid.
    """
    return 1.0 / (1.0 + math.exp(-x))  # if x >= 0 else 1.0 / (1.0+math.exp(x))


def relu(x: float) -> float:
    """Compute relu

    Args:x (float): The input number.

    Returns:float: The output of relu.
    """
    if x > 0:
        return x
    else:
        return 0.0


def log(x: float) -> float:
    """Compute log

    Args:x (float): The input number.

    Returns:float: The output of log.
    """
    eps = 1e-6
    return float(math.log(x + eps))


def exp(x: float) -> float:
    """Compute exp

    Args:x (float): The input number.

    Returns:float: The output of exp.
    """
    return float(math.exp(float(x)))


def inv(x: float) -> float:
    """Compute inv

    Args:x (float): The input number.

    Returns:float: The output of inv.
    """
    return float(1 / x)


def log_back(x: float, d: float) -> float:
    """Compute log_back

    Args:x (float): The input number.

    Returns:float: The output of log_back.
    """
    return float((1 / x) * d)


def inv_back(x: float, d: float) -> float:
    """Compute inv_back

    Args:x (float): The input number.

    Returns:float: The output of inv_back.
    """
    return float(-d / x**2)


def relu_back(x: float, d: float) -> float:
    """Compute relu_back

    Args:x (float): The input number.

    Returns:float: The output of relu_back.
    """
    return float(d) if x > 0 else 0.0


def sigmoid_back(x: float, d: float) -> float:
    """Compute sigmoid back"""
    return float(d * exp(-x) / ((1 + exp(-x)) ** 2))


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# TODO: Implement for Task 0.3.


def map(ls: list, fn: Callable) -> list:
    """Compute map

    Args:ls (list): The input.

    Returns:list: The output of map.
    """
    return [fn(x) for x in ls]


def zipWith(ls1: list, ls2: list, fn: Callable) -> list:
    """Compute zipWith

    Args:ls (list): The input.

    Returns:list: The output of zipWith.
    """
    return [fn(ls1[i], ls2[i]) for i in range(len(ls1))]


def reduce(ls: Iterable[float], fn: Callable, start: float) -> float:
    """Compute reduce

    Args:ls (list): The input.

    Returns:float: The output of reduce.
    """
    for n in ls:
        start = fn(start, n)
    return start


def negList(ls: list) -> list:
    """Compute negList

    Args:ls (list): The input.

    Returns:list: The output of negLIst.
    """
    return map(ls, neg)


def addLists(ls1: list, ls2: list) -> list:
    """Compute addLists

    Args:ls (list): The input.

    Returns:list: The output of addLists.
    """
    return zipWith(ls1, ls2, add)


def sum(ls: list) -> float:
    """Compute sum

    Args:ls (list): The input.

    Returns:list: The output of sum.
    """
    return reduce(ls, add, 0)


def prod(ls: Iterable[float]) -> float:
    """Compute sum

    Args:ls (list): The input.

    Returns:list: The output of sum.
    """
    return reduce(ls, mul, 1)
