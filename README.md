# Đớp Tool

Structural Optimization Tool — FEM + Calculus of Variations + KKT.

## Kiến trúc

- `fem_core.py`: giữ nguyên FEM core hiện tại.
- `optimization_core.py`: lớp tối ưu mới.
- `dop_tool_app.py`: giao diện Streamlit.

## Chạy

```bash
pip install -r requirements.txt
streamlit run dop_tool_app.py
```

## Nguyên tắc

### FEM
FEM là bộ phân tích: từ geometry + loads + A/I → U, reactions, N/V/M.

### Variational
Với bài toán thanh chịu lực dọc:

\[
\min_A \int_0^L A(x)dx
\]

\[
\delta=\int_0^L \frac{N(x)^2}{EA(x)}dx\le\delta_{allow}
\]

Euler–Lagrange cho nghiệm trong miền:

\[
A^*(x)=|N(x)|\sqrt{\lambda/E}.
\]

Nếu có ràng buộc ứng suất:

\[
A^*(x)=\max\left(|N|/\sigma_{allow}, |N|\sqrt{\lambda/E}\right).
\]

### KKT
Nghiệm số được kiểm tra:

- primal feasibility
- dual feasibility
- complementary slackness
- stationarity

## Giai đoạn tiếp theo

1. Truss solver riêng với DOF `[ux, uy]`.
2. Frame optimizer toàn hệ.
3. Buckling constraint.
4. Stress interaction.
5. Section library: thép hình, thép hộp, tiết diện tổ hợp.
6. Continuous-to-discrete mapping: A*(x), I*(x) → tiết diện chế tạo.
7. Multi-load cases.
8. Robust / manufacturability constraints.
