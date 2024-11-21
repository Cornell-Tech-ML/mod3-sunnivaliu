import graph_builder # type: ignore
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from interface.streamlit_utils import render_function # type: ignore # type: ignore

import minitorch
from minitorch import MathTest, MathTestVariable

MyModule = None
minitorch


def render_math_sandbox(use_scalar=False, use_tensor=False): # type: ignore # type: ignore
    st.write("## Sandbox for Math Functions")
    st.write("Visualization of the mathematical tests run on the underlying code.")

    if use_scalar:
        one, two, red = MathTestVariable._comp_testing()
    else:
        one, two, red = MathTest._comp_testing()
    f_type = st.selectbox("Function Type", ["One Arg", "Two Arg", "Reduce"])
    select = {"One Arg": one, "Two Arg": two, "Reduce": red}

    fn = st.selectbox("Function", select[f_type], format_func=lambda a: a[0])
    name, _, scalar = fn
    if f_type == "One Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                ys = [scalar(minitorch.tensor([p]))[0] for p in xs] # type: ignore # type: ignore
            else:
                ys = [scalar(minitorch.Scalar(p)).data for p in xs]
        else:
            ys = [scalar(p) for p in xs]
        scatter = go.Scatter(mode="lines", x=xs, y=ys) # type: ignore
        fig = go.Figure(scatter) # type: ignore
        st.write(fig)

        if use_scalar:
            st.write("Derivative f'(x)")
            if use_tensor:
                x_var = [minitorch.tensor(x, requires_grad=True) for x in xs] # type: ignore # type: ignore
            else:
                x_var = [minitorch.Scalar(x) for x in xs]
            for x in x_var:
                out = scalar(x)
                if use_tensor:
                    out.backward(minitorch.tensor([1.0])) # type: ignore
                else:
                    out.backward()
            if use_tensor:
                scatter = go.Scatter(mode="lines", x=xs, y=[x.grad[0] for x in x_var]) # type: ignore # type: ignore
            else:
                scatter = go.Scatter(
                    mode="lines", x=xs, y=[x.derivative for x in x_var]
                ) # type: ignore
            fig = go.Figure(scatter) # type: ignore
            st.write(fig)
            G = graph_builder.GraphBuilder().run(out) # type: ignore
            G.graph["graph"] = {"rankdir": "LR"}
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    if f_type == "Two Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x, y)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                zs = [
                    [
                        scalar(minitorch.tensor([x]), minitorch.tensor([y]))[0] # type: ignore # type: ignore
                        for x in xs
                    ]
                    for y in ys
                ]
            else:
                zs = [
                    [scalar(minitorch.Scalar(x), minitorch.Scalar(y)).data for x in xs]
                    for y in ys
                ]
        else:
            zs = [[scalar(x, y) for x in xs] for y in ys]

        scatter = go.Surface(x=xs, y=ys, z=zs) # type: ignore

        fig = go.Figure(scatter) # type: ignore # type: ignore
        st.write(fig)
        if use_scalar:
            a, b = [], []
            for x in xs:
                oa, ob = [], []

                if use_tensor:
                    for y in ys:
                        x1 = minitorch.tensor([x]) # type: ignore # type: ignore
                        y1 = minitorch.tensor([y]) # type: ignore
                        out = scalar(x1, y1)
                        out.backward(minitorch.tensor([1])) # type: ignore
                        oa.append((x, y, x1.derivative[0]))
                        ob.append((x, y, y1.derivative[0]))
                else:
                    for y in ys:
                        x1 = minitorch.Scalar(x)
                        y1 = minitorch.Scalar(y)
                        out = scalar(x1, y1)
                        out.backward()
                        oa.append((x, y, x1.derivative))
                        ob.append((x, y, y1.derivative))
                a.append(oa)
                b.append(ob)
            st.write("Derivative f'_x(x, y)")

            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in a],
                y=[[c[1] for c in a2] for a2 in a],
                z=[[c[2] for c in a2] for a2 in a],
            ) # type: ignore
            fig = go.Figure(scatter) # type: ignore # type: ignore
            st.write(fig)
            st.write("Derivative f'_y(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in b],
                y=[[c[1] for c in a2] for a2 in b],
                z=[[c[2] for c in a2] for a2 in b],
            ) # type: ignore # type: ignore
            fig = go.Figure(scatter) # type: ignore
            st.write(fig)
    if f_type == "Reduce":
        st.write("### " + name)
        render_function(scalar)
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]

        if use_tensor:
            scatter = go.Surface(
                x=xs,
                y=ys,
                z=[[scalar(minitorch.tensor([x, y]))[0] for x in xs] for y in ys], # type: ignore # type: ignore
            ) # type: ignore
        else:
            scatter = go.Surface(
                x=xs, y=ys, z=[[scalar([x, y]) for x in xs] for y in ys]
            ) # type: ignore
        fig = go.Figure(scatter) # type: ignore
        st.write(fig)
