from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# from __future__ import annotations
# from dataclasses import dataclass
# from msvcrt import LK_LOCK
# from re import L
# from typing import Any, Iterable, List, Tuple, Protocol
# from networkx import is_empty

# ## Task 1.1
# Central Difference calculations


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.

    vals_plus = [v + epsilon if i == arg else v for i, v in enumerate(vals)]
    vals_minus = [v - epsilon if i == arg else v for i, v in enumerate(vals)]
    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)  # add pointer


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate_derivative"""
        ...

    @property
    def unique_id(self) -> int:
        """Unique_id"""
        ...

    def is_leaf(self) -> bool:
        """Is_leaf"""
        ...

    def is_constant(self) -> bool:
        """Is_constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Parents"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain_rule"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    topo_order = []
    visited = set()

    def visit(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for parent in v.parents:
                if not parent.is_constant():
                    visit(parent)
        visited.add(v.unique_id)
        topo_order.append(v)

    visit(variable)
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
    None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    derivs = {variable.unique_id: deriv}
    for v in topological_sort(variable):
        if v.is_leaf():
            v.accumulate_derivative(derivs[v.unique_id])
        else:
            for parent, d in v.chain_rule(derivs[v.unique_id]):
                if parent.is_constant():
                    continue
                derivs.setdefault(parent.unique_id, 0.0)
                derivs[parent.unique_id] += d


### only leaf Scalars should ever have non-None .derivative value. All intermediate Scalars should only keep their current derivative values in the dictionary.


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Saved_tensors"""
        return self.saved_values
