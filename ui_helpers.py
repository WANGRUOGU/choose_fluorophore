# ui_helpers.py
import numpy as np
import streamlit as st

# 颜色表，和你原来 DEFAULT_COLORS 完全一样
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25], [0.25, 0.65, 0.95],
    [0.25, 0.85, 0.35], [0.90, 0.70, 0.20],
    [0.80, 0.40, 0.80], [0.25, 0.80, 0.80],
    [0.85, 0.50, 0.35], [0.60, 0.60, 0.60],
], dtype=float)


def _ensure_colors(R):
    if R <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:R]
    hs = np.linspace(0, 1, R, endpoint=False)
    extra = np.stack([
        np.abs(np.sin(2*np.pi*hs))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.33)))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.66)))*0.7+0.3
    ], axis=1)
    return extra[:R]


def _rgb01_to_plotly(col):
    r, g, b = (int(255*x) for x in col)
    return f"rgb({r},{g},{b})"


def _pair_only_fluor(a, b):
    fa = a.split(" – ", 1)[1] if " – " in a else a
    fb = b.split(" – ", 1)[1] if " – " in b else b
    return f"{fa} vs {fb}"


def _html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                        color_second_row=False, color_thresh=0.9, fmt2=False):
    def esc(x): 
        return (str(x).replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))
    def fmtv(v):
        if fmt2:
            try:
                return f"{float(v):.3f}"
            except Exception:
                return esc(v)
        return esc(v)

    cells0 = "".join(
        f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>"
        for v in row0_vals
    )
    tds0 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>"
        f"{esc(row0_label)}</td>{cells0}"
    )

    tds1_list = []
    for v in row1_vals:
        style = "padding:6px 10px;border:1px solid #ddd;"
        if color_second_row:
            try:
                vv = float(v)
                style += f"color:{'red' if vv > color_thresh else 'green'};"
            except Exception:
                pass
        tds1_list.append(f"<td style='{style}'>{fmtv(v)}</td>")
    tds1 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>"
        f"{esc(row1_label)}</td>{''.join(tds1_list)}"
    )

    st.markdown(f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;table-layout:auto;">
        <tbody><tr>{tds0}</tr><tr>{tds1}</tr></tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


def _html_table(headers, rows, num_cols=None):
    num_cols = num_cols or set()

    def esc(x):
        return (str(x).replace("&","&amp;")
                      .replace("<","&lt;")
                      .replace(">","&gt;"))

    thead = "".join(
        f"<th style='padding:6px 10px;border:1px solid #ddd;text-align:left'>{esc(h)}</th>"
        for h in headers
    )
    trs = []
    for r in rows:
        tds = []
        for j, v in enumerate(r):
            text = f"{float(v):.4f}" if j in num_cols else esc(v)
            align = "right" if j in num_cols else "left"
            tds.append(
                f"<td style='padding:6px 10px;border:1px solid #ddd;"
                f"text-align:{align}'>{text}</td>"
            )
        trs.append(f"<tr>{''.join(tds)}</tr>")

    st.markdown(
        f"""
        <div style="overflow-x:auto;">
          <table style="border-collapse:collapse;width:100%;table-layout:auto;">
            <thead><tr>{thead}</tr></thead>
            <tbody>{''.join(trs)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True
    )


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _show_bw_grid(title, imgs_uint8, labels, cols_per_row=6):
    st.markdown(f"**{title}**")
    n = len(imgs_uint8)
    for i in range(0, n, cols_per_row):
        chunk_imgs = imgs_uint8[i:i+cols_per_row]
        chunk_labels = labels[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if j < len(chunk_imgs):
                cols[j].image(chunk_imgs[j], use_container_width=True, clamp=True)
                cols[j].caption(chunk_labels[j])
            else:
                cols[j].markdown("&nbsp;")


def _prettify_name(label: str) -> str:
    """Map 'Probe – AF405' -> 'AF 405'; leave other names as-is."""
    name = label.split(" – ", 1)[1] if " – " in label else label
    up = name.upper()
    if up.startswith("AF") and name[2:].isdigit():
        return f"AF {name[2:]}"
    return name
