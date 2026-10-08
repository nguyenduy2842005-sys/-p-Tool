
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimization_core import (
    OptimizationSettings,
    optimize_single_member,
    variational_axial_solution,
)


st.set_page_config(
    page_title="Đớp Tool — Structural Optimization",
    page_icon="🧠",
    layout="wide",
)


st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.2rem;}
.hero {
    padding: 18px 22px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(30,80,150,.14), rgba(120,40,140,.08));
}
.hero h1 {margin:0; font-size: 2.25rem;}
.hero p {margin:.35rem 0 0; opacity:.78;}
.kpi {
    border:1px solid rgba(128,128,128,.25);
    border-radius:10px; padding:12px;
}
.small {font-size:.85rem; opacity:.72;}
</style>
""", unsafe_allow_html=True)


def plot_distribution(x, y, title, ytitle):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="x (m)",
        yaxis_title=ytitle,
        height=360,
        margin=dict(l=45, r=20, t=55, b=45),
        template="plotly_white",
    )
    return fig


def main():
    st.markdown("""
    <div class="hero">
      <h1>🧠 Đớp Tool</h1>
      <p>Structural Optimization Lab — FEM · Calculus of Variations · KKT</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Mục tiêu: dùng FEM để đánh giá cấu trúc, dùng Biến phân để tìm dạng thiết kế liên tục, "
        "sau đó dùng KKT để kiểm chứng điều kiện tối ưu."
    )

    tabs = st.tabs([
        "🏠 Tổng quan",
        "1️⃣ Cấu kiện riêng lẻ",
        "2️⃣ Biến phân A(x)",
        "3️⃣ KKT Verification",
        "4️⃣ Tối ưu toàn khung",
    ])

    with tabs[0]:
        st.subheader("Kiến trúc tính toán")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 01 — FEM")
            st.write("Tính chuyển vị, phản lực và nội lực cho cấu kiện/khung.")
        with c2:
            st.markdown("### 02 — Variational")
            st.write("Thiết lập phiếm hàm và Euler–Lagrange để tìm dạng A*(x), I*(x).")
        with c3:
            st.markdown("### 03 — KKT")
            st.write("Kiểm tra khả thi nguyên thủy, đối ngẫu, bổ sung và stationarity.")

        st.info(
            "Lưu ý: dàn thanh thuần túy tối ưu chủ yếu theo A(x). I(x) phù hợp với dầm/khung "
            "khi độ cứng uốn là một phần của bài toán."
        )

        st.latex(
            r"\min_{A(x),I(x)} J[A,I]"
            r"\quad\text{s.t.}\quad"
            r"g_i[A,I]\le0,\; h_j[A,I]=0"
        )
        st.markdown(
            "**Luồng của Đớp Tool:**  "
            "`Geometry → FEM → Objective/Constraints → Variational candidate → Numerical optimization → KKT check`"
        )

    with tabs[1]:
        st.subheader("Tối ưu một cấu kiện riêng lẻ")

        left, right = st.columns([0.34, 0.66])
        with left:
            length = st.number_input("Chiều dài L (m)", 1.0, 100.0, 10.0, .5)
            nseg = st.slider("Số đoạn FEM / số DOF thiết kế", 2, 30, 10)
            load_type = st.selectbox("Trạng thái tải", ["axial", "bending", "combined"])
            load = st.number_input("Tải tại đầu (kN)", -10000.0, 10000.0, 100.0, 10.0)

            st.markdown("**Biến thiết kế**")
            opt_A = st.checkbox("Tối ưu A(x)", True)
            opt_I = st.checkbox("Tối ưu I(x)", False)

            A0 = st.number_input("A ban đầu (m²)", 1e-4, .1, .01, 1e-4, format="%.6f")
            I0 = st.number_input("I ban đầu (m⁴)", 1e-8, .01, 1e-4, 1e-6, format="%.8f")

            sigma = st.number_input("σ cho phép (kN/m²)", 1e3, 1e6, 160e3, 1e3)
            disp = st.number_input("u cho phép (m)", 1e-5, 1.0, .02, .001)
            c = st.number_input("c — khoảng cách biên (m)", .001, 2.0, .10, .01)

            run = st.button("🚀 Chạy tối ưu", type="primary", use_container_width=True)

        with right:
            if run:
                settings = OptimizationSettings(
                    sigma_allow=sigma,
                    disp_allow=disp,
                    c=c,
                    optimize_A=opt_A,
                    optimize_I=opt_I,
                    I_min=max(I0*0.01, 1e-10),
                    I_max=max(I0*100, I0*1.01),
                    A_min=max(A0*0.01, 1e-6),
                    A_max=max(A0*100, A0*1.01),
                    volume_weight_I=0.0,
                )
                with st.spinner("FEM + SLSQP + KKT..."):
                    result = optimize_single_member(
                        length, nseg, settings, load_type, load, A0, I0
                    )
                st.session_state["last_opt"] = result

            result = st.session_state.get("last_opt")
            if result is None:
                st.info("Thiết lập bài toán bên trái rồi nhấn Chạy tối ưu.")
            else:
                cols = st.columns(4)
                vals = [
                    ("Trạng thái", "OK" if result.success else "FAIL"),
                    ("Objective", f"{result.objective:.5g} m²"),
                    ("u_max", f"{result.max_displacement:.5g} m"),
                    ("KKT", "PASS" if result.kkt["passed"] else "CHECK"),
                ]
                for col, (lab, val) in zip(cols, vals):
                    with col:
                        st.markdown(f'<div class="kpi"><b>{lab}</b><br>{val}</div>', unsafe_allow_html=True)

                x = np.linspace(0, length, nseg)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_distribution(x, result.A, "A*(x)", "A (m²)"),
                                    use_container_width=True)
                with c2:
                    st.plotly_chart(plot_distribution(x, result.I, "I*(x)", "I (m⁴)"),
                                    use_container_width=True)

                st.write(result.message)
                if result.history:
                    st.dataframe(pd.DataFrame(result.history), use_container_width=True)

    with tabs[2]:
        st.subheader("Biến phân bậc nhất — Euler–Lagrange cho A(x)")
        st.markdown(
            "Bài toán mẫu: **tối thiểu thể tích** của thanh chịu lực dọc, với ràng buộc chuyển vị và ứng suất."
        )
        st.latex(
            r"\min_A\;J[A]=\int_0^L A(x)\,dx"
        )
        st.latex(
            r"\text{s.t.}\quad"
            r"\delta[A]=\int_0^L\frac{N(x)^2}{E\,A(x)}\,dx\le\delta_{allow},\qquad"
            r"\frac{|N(x)|}{A(x)}\le\sigma_{allow}"
        )
        st.markdown("**Phiếm hàm Lagrange:**")
        st.latex(
            r"\mathcal{L}=\int_0^L\left[A(x)+\lambda\frac{N(x)^2}{E A(x)}\right]dx"
        )
        st.markdown("Vì F không chứa A'(x), phương trình Euler–Lagrange suy biến thành điều kiện đại số:")
        st.latex(
            r"\frac{\partial F}{\partial A}=0"
            r"\;\Rightarrow\;"
            r"1-\lambda\frac{N(x)^2}{E A(x)^2}=0"
            r"\;\Rightarrow\;"
            r"A^*(x)=|N(x)|\sqrt{\frac{\lambda}{E}}"
        )
        st.markdown(
            "Khi đồng thời có ràng buộc ứng suất, nghiệm KKT có dạng:"
        )
        st.latex(
            r"A^*(x)=\max\left("
            r"\frac{|N(x)|}{\sigma_{allow}},\;"
            r"|N(x)|\sqrt{\frac{\lambda}{E}}"
            r"\right)"
        )

        st.divider()
        st.markdown("### Thí nghiệm số")
        L = st.number_input("L (m)", 1.0, 100.0, 10.0, .5, key="varL")
        P = st.number_input("|N| (kN)", 1.0, 10000.0, 100.0, 10.0, key="varP")
        E = st.number_input("E (kN/m²)", 1e5, 1e9, 200e6, 1e6, key="varE")
        sig = st.number_input("σallow (kN/m²)", 1e3, 1e6, 160e3, 1e3, key="varSig")
        du = st.number_input("δallow (m)", 1e-5, 1.0, .02, .001, key="varDu")

        if st.button("Tính nghiệm Euler–Lagrange + KKT", key="run_var"):
            out = variational_axial_solution(
                lambda x: np.full_like(x, P), L, E, sig, du
            )
            st.session_state["var_out"] = out

        out = st.session_state.get("var_out")
        if out:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_distribution(out["x"], out["A"], "Tiết diện tối ưu A*(x)", "A (m²)"),
                    use_container_width=True
                )
            with c2:
                st.metric("λ", f"{out['lambda']:.6g}")
                st.metric("δ(A*)", f"{out['displacement']:.6g} m")
                st.metric("V(A*)", f"{out['volume']:.6g} m²")
            st.caption("Đây là nghiệm biến phân liên tục; bước FEM ở tab cấu kiện dùng nghiệm số để kiểm chứng.")

    with tabs[3]:
        st.subheader("Kiểm chứng KKT")
        result = st.session_state.get("last_opt")
        if result is None:
            st.info("Hãy chạy một bài toán tối ưu ở tab Cấu kiện riêng lẻ trước.")
        else:
            k = result.kkt
            st.write("### 1. Primal feasibility")
            st.write(f"Max violation: `{k['primal_violation']:.4e}`")
            st.write("### 2. Dual feasibility")
            st.write(f"Max negative multiplier: `{k['dual_violation']:.4e}`")
            st.write("### 3. Complementary slackness")
            st.write(f"Max |λᵢgᵢ|: `{k['complementarity']:.4e}`")
            st.write("### 4. Stationarity")
            st.write(f"‖∇f + Σλᵢ∇gᵢ‖ = `{k['stationarity_norm']:.4e}`")
            st.write("Active constraints:", k["active"])
            st.dataframe(pd.DataFrame({
                "constraint": np.arange(len(k["constraint_values"])),
                "g(x)": k["constraint_values"],
                "lambda": k["lambda"],
            }), use_container_width=True)
            st.success("KKT PASS" if k["passed"] else "KKT cần kiểm tra thêm")

    with tabs[4]:
        st.subheader("Tối ưu toàn bộ khung — kiến trúc sẽ dùng chung FEM core")
        st.warning(
            "MVP này mới hoàn thiện bộ máy tối ưu cấu kiện. Tab này được dành cho optimizer cấp hệ: "
            "A_e(x), I_e(x) → FEM toàn khung → constraints toàn hệ → KKT."
        )
        st.markdown("""
        **Kiến trúc dự kiến:**

        1. Người dùng vẽ/nhập khung.
        2. `fem_core.solve_plane_frame()` phân tích hệ.
        3. Mỗi phần tử có trường thiết kế `A_e(x), I_e(x)`.
        4. Objective: tổng thể tích/khối lượng hoặc chi phí.
        5. Constraints: chuyển vị nút, ứng suất, ổn định, giới hạn tiết diện.
        6. Variational module sinh nghiệm liên tục ban đầu.
        7. Numerical optimizer hiệu chỉnh nghiệm trên toàn hệ.
        8. KKT module xác nhận nghiệm cuối.
        """)
        st.latex(
            r"\min_{\{A_e(x),I_e(x)\}}\;"
            r"\sum_e\int_0^{L_e}\rho A_e(x)\,dx"
            r"\quad\text{s.t.}\quad"
            r"\mathbf K(\mathbf A,\mathbf I)\mathbf U=\mathbf F"
        )


if __name__ == "__main__":
    main()
