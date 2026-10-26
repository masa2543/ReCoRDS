"""RocketCEA 液体ロケットエンジン性能計算 Web アプリ (Streamlit + Plotly)

Qiita「RocketCEAでロケットエンジン性能計算」(ina111) のサンプル 01〜07, 11 を
ブラウザから手軽に実行できるようにしたもの。単位系は SI (MPa, m/s, K, sec)。
"""

import hashlib
import re
import threading

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
from rocketcea.biprop_utils.density_at_mr import add_prop_sg, bulkDensity
from rocketcea.biprop_utils.mr_t_limits import MR_Temperature_Limits
from rocketcea import cea_obj as cea_mod
from rocketcea.cea_obj import CEA_Obj as CEA_Obj_raw
from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer, add_new_propellant
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.input_cards import fuelCards, oxCards

G0 = 9.80665


@st.cache_resource
def _cea_lock():
    # RocketCEA は Fortran の共有状態 (COMMON ブロックや結果キャッシュ) を使うため、複数セッションが同時に
    # 計算すると互いの値を上書きして誤った値や nan になる。全セッションで共有するロックで 1 件ずつ実行する
    return threading.RLock()


CEA_LOCK = _cea_lock()


def fnum(v):
    """表示用の数値。余分な桁は省くが、整数でも '2.0' のように小数点以下を 1 桁残す"""
    s = f"{v:g}"
    return s if any(ch in s for ch in ".eEn") else s + ".0"

st.set_page_config(page_title="ReCoRDS", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      div[data-testid="stVerticalBlockBorderWrapper"] {background: rgba(127,127,127,0.04);}
      h1 {font-size: 1.6rem !important; margin-bottom: 0 !important;}
      .sec {display: flex; align-items: baseline; gap: .6rem; margin: -.2rem 0 .5rem;
            padding-left: .6rem; border-left: 4px solid var(--accent, #ff4b4b);}
      .sec b {font-size: 1.05rem; white-space: nowrap;}
      /* 横に並んだ入力パネル (枠付きコンテナ) の高さを揃える */
      [data-testid="stColumn"] > [data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"]:only-child),
      [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child,
      [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child
        > [data-testid="stVerticalBlock"] {height: 100%;}
      .sec span {font-size: .8rem; opacity: .65;}
      .sub {font-size: .85rem; font-weight: 600; margin-bottom: .2rem; opacity: .85;}
      .res {font-size: 1.15rem; font-weight: 700; margin: 1.2rem 0 .3rem;}
      /* サイドメニューを開くボタンをハンバーガー (☰) 表示に */
      [data-testid="stExpandSidebarButton"] {width: 40px; height: 40px; border-radius: 8px;
            border: 1px solid rgba(127,127,127,.35); background: rgba(127,127,127,.06);}
      [data-testid="stExpandSidebarButton"] > span {display: none;}
      [data-testid="stExpandSidebarButton"]::before {content: "☰"; font-size: 1.4rem; line-height: 1;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 ReCoRDS")
st.caption("NASA CEA (RocketCEA) による化学平衡計算 ― 単位: 圧力 MPa / c* m/s / 温度 K / Isp sec")

# グラフ操作:
#   - グラフをクリックするとそのグラフが「操作中」になり、ホイールでズーム・ドラッグで移動できる
#     (操作中でないグラフ上のホイールは普通のページスクロール。グラフ外をクリックで解除)
#   - 線または凡例をクリックするとその系列に注目 (他を薄く。クリックごとに追加/解除で複数可)、
#     ダブルクリックで注目とズームをリセット
PLOT_CFG = {
    "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
}
st.html(
    """
    <style>
      .js-plotly-plot {border-radius: 6px; outline: 2px solid transparent; transition: outline-color .15s;}
      .js-plotly-plot.rec-active {outline-color: #1c7ed6;}
    </style>
    <script>
    (() => {
      if (window.__recordsPlots) return;
      window.__recordsPlots = true;
      const DIM = 0.15, NEAR_PX = 15;
      let active = null;

      // 注目の単位: meta (実線と破線で別) → なければ凡例グループ → なければトレースごと
      const keyOf = (gd, i) => { const t = gd._fullData[i]; return (t.meta != null ? String(t.meta) : "") || t.legendgroup || "i" + i; };
      function applyFocus(gd) {                   // 注目中の系列 (複数可) 以外を薄くする
        const f = gd.__recFocus || new Set();
        const op = gd._fullData.map((t, i) =>
          !f.size || f.has(keyOf(gd, i)) || (!t.legendgroup && t.hoverinfo === "skip") ? 1 : DIM);
        if (window.Plotly) window.Plotly.restyle(gd, {opacity: op});
      }
      function toggleFocus(gd, i) {               // クリックするたびに注目に追加/解除
        const f = gd.__recFocus || (gd.__recFocus = new Set()), k = keyOf(gd, i);
        f.has(k) ? f.delete(k) : f.add(k);
        applyFocus(gd);
      }
      function setActive(gd) {
        if (active === gd) return;
        if (active) { active.classList.remove("rec-active"); window.Plotly.relayout(active, {dragmode: false}); }
        active = gd;
        if (gd) { gd.classList.add("rec-active"); window.Plotly.relayout(gd, {dragmode: "pan"}); }
      }
      function fixTicks(gd) {                     // 目盛りの整数 "2" を "2.0" と表示する
        gd.querySelectorAll(".xtick text, .ytick text").forEach((t) => {
          if (/^[−-]?\\d+$/.test(t.textContent)) t.textContent += ".0";
        });
      }
      function bind(gd) {
        gd.__recBound = true;
        gd.on("plotly_afterplot", () => fixTicks(gd));
        fixTicks(gd);
        gd.on("plotly_click", (d) => {           // カーソルに最も近い線に注目
          const top = gd.getBoundingClientRect().top, my = d.event.clientY - top;
          let best = null, bestDist = NEAR_PX;
          for (const p of d.points) {
            const py = p.yaxis.l2p(p.y) + p.yaxis._offset, dist = Math.abs(py - my);
            if (dist < bestDist) { bestDist = dist; best = p; }
          }
          if (best) toggleFocus(gd, best.curveNumber);
        });
        gd.on("plotly_legendclick", (d) => { toggleFocus(gd, d.curveNumber); return false; });
        gd.on("plotly_legenddoubleclick", () => { gd.__recFocus = null; applyFocus(gd); return false; });
        gd.on("plotly_doubleclick", () => { gd.__recFocus = null; applyFocus(gd); });
      }
      setInterval(() => {                         // 再描画で作り直されたグラフにも付け直す
        document.querySelectorAll(".js-plotly-plot").forEach((gd) => { if (!gd.__recBound && gd.on) bind(gd); });
        if (active && !document.contains(active)) active = null;
        if (active && active._fullLayout && active._fullLayout.dragmode !== "pan") window.Plotly.relayout(active, {dragmode: "pan"});
      }, 700);

      window.addEventListener("mousedown", (e) => {
        setActive(e.target.closest ? e.target.closest(".js-plotly-plot") : null);
        // サイドメニューが開いているとき、メニューの外をクリックしたら閉じる
        // (選択肢のドロップダウンやツールチップはメニュー外に描かれるので除外)
        const sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb && sb.dataset.collapsed === "false" && !sb.contains(e.target) &&
            !e.target.closest('[data-testid="stExpandSidebarButton"], [data-baseweb="popover"], [role="listbox"], [role="tooltip"]')) {
          const btn = sb.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                      sb.querySelector('[data-testid="stSidebarCollapseButton"]');
          if (btn) btn.click();
        }
      }, {capture: true});
      // 数値・テキスト欄をクリック (フォーカス) したら現在の値を全選択し、そのまま入力で上書きできるようにする
      const EDIT_SEL = '[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input';
      let justFocused = null;
      document.addEventListener("focusin", (e) => {
        if (e.target.matches && e.target.matches(EDIT_SEL)) {
          justFocused = e.target;
          setTimeout(() => e.target.select(), 0);
        }
      });
      document.addEventListener("mouseup", (e) => {   // クリック直後のマウスアップで選択が解除されないように
        if (justFocused && e.target === justFocused) { e.preventDefault(); justFocused.select(); }
        justFocused = null;
      }, true);
      window.addEventListener("wheel", (e) => {   // 操作中でないグラフ → Plotly に渡さず通常スクロール
        const gd = e.target.closest ? e.target.closest(".js-plotly-plot") : null;
        if (gd && gd !== active) e.stopImmediatePropagation();
      }, {capture: true, passive: true});
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# 追加燃料: ABS 樹脂 (ハイブリッドロケット用)。組成 C3.85 H4.85 N0.43,
# 生成エンタルピー 62.63 kJ/mol, 密度 1.05 g/cc
ABS_CARD = """fuel=ABS  C 3.85 H 4.85 N 0.43  wt%=100.00
h,kj/mol=62.63     t(k)=298.15  rho.g/cc=1.05"""


@st.cache_resource(show_spinner=False)
def register_extra_fuels():
    with CEA_LOCK:
        add_new_fuel("ABS", ABS_CARD)
        add_prop_sg("ABS", 1.05)  # 推進剤比較 (かさ密度) 用
    return True


register_extra_fuels()

OX_LIST = sorted(oxCards, key=str.upper)
FUEL_LIST = sorted(fuelCards, key=str.upper)

# カスタム推進剤カードの項目別入力。初期値は液体酸素・液体メタン・過酸化水素相当
CARD_DEFAULTS = {
    "ox": dict(name="MyOx", formula="O 2", wt=100.0, h=-12.979, t=90.18, rho=1.1409),
    "fuel": dict(name="MyFuel", formula="C 1 H 4", wt=100.0, h=-89.496, t=111.66, rho=0.4226),
    "mono": dict(name="H2O2(L)", formula="H 2 O 2", wt=100.0, h=-187.78, t=298.15, rho=1.407),
}
CARD_HEAD = {"ox": "oxid={}", "fuel": "fuel={}", "mono": "name {}"}  # NASA CEA カードの先頭の書式


def norm_formula(text):
    """'C 1 H 4' / 'CH4' / 'C3.85H4.85N0.43' のどれでも 'C 1 H 4' 形式にする"""
    s = text.strip()
    if not re.fullmatch(r"(\s*[A-Z][a-z]?\s*[0-9.]*\s*)+", s):
        raise ValueError(f"組成を解釈できません: '{text}' (例: C 1 H 4 または CH4)")
    return " ".join(f"{el} {n or 1}" for el, n in re.findall(r"([A-Z][a-z]?)\s*([0-9.]*)", s))


def card_fields(p, kind, title, where=None):
    d = CARD_DEFAULTS[kind]
    with (where or st).container(border=True):
        st.markdown(f'<div class="sub">{title}</div>', unsafe_allow_html=True)
        c = p.cols(3)
        p.w("text_input", "物質名", f"{kind}_name", d["name"], where=c[0], help="空白なしの名前")
        p.w("text_input", "組成", f"{kind}_formula", d["formula"], where=c[1], help="例: C 1 H 4 または CH4")
        p.w("number_input", "質量パーセント濃度 [wt%]", f"{kind}_wt", d["wt"], where=c[2],
            min_value=0.0, max_value=100.0, step=1.0)
        c = p.cols(3)
        p.w("number_input", "標準生成エンタルピー [kJ/mol]", f"{kind}_h", d["h"], where=c[0], step=1.0, format="%.3f")
        p.w("number_input", "基準とする温度 [K]", f"{kind}_t", d["t"], where=c[1],
            min_value=1.0, max_value=6000.0, step=1.0, format="%.2f")
        p.w("number_input", "密度 [g/cm³]", f"{kind}_rho", d["rho"], where=c[2],
            min_value=0.0, max_value=30.0, step=0.01, format="%.4f")


def build_card(kind):
    S = st.session_state
    name = S[f"{kind}_name"].strip().replace(" ", "_") or CARD_DEFAULTS[kind]["name"]
    return (f"{CARD_HEAD[kind].format(name)} {norm_formula(S[f'{kind}_formula'])}  wt%={S[f'{kind}_wt']:.2f}\n"
            f"h,kj/mol={S[f'{kind}_h']:.4f}     t(k)={S[f'{kind}_t']:.2f}  rho.g/cc={S[f'{kind}_rho']:.4f}")

FLOW_MODES = {
    "平衡流 (equilibrium)": (0, 0),
    "スロートで凍結 (frozen at throat)": (1, 1),
    "燃焼室で凍結 (frozen in chamber)": (1, 0),
}


def section(title, note="", color="#ff4b4b"):
    st.markdown(f'<div class="sec" style="--accent:{color}"><b>{title}</b><span>{note}</span></div>',
                unsafe_allow_html=True)


class Panel:
    """入力パネルの描画先。同じパネルをページ上部 (wide) とサイドメニュー (narrow) に描き、
    両者のウィジェットは共通の session_state の値 (キー名そのまま) を介して同期する。"""

    def __init__(self, prefix, wide):
        self.prefix, self.wide = prefix, wide

    def cols(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return st.columns(spec) if self.wide else [st.container() for _ in range(n)]

    def w(self, kind, label, key, default, *args, where=None, **kw):
        ss = st.session_state
        if key not in ss:
            ss[key] = default
        wk = f"{self.prefix}:{key}"
        ss[wk] = ss[key]  # もう一方で変更された値を反映
        getattr(where or st, kind)(label, *args, key=wk, on_change=_sync, args=(key, wk), **kw)
        return ss[key]


def _sync(key, wk):
    st.session_state[key] = st.session_state[wk]


def range_input(p, col_lo, col_hi, label, vmin, vmax, lo, hi, step, disabled=False):
    kw = dict(min_value=vmin, max_value=vmax, step=step, disabled=disabled)
    p.w("number_input", "最小", f"{label}_lo", lo, where=col_lo, **kw)
    p.w("number_input", "最大", f"{label}_hi", hi, where=col_hi, **kw)


def get_range(label, step):
    lo, hi = st.session_state[f"{label}_lo"], st.session_state[f"{label}_hi"]
    if lo >= hi:
        st.warning(f"{label} の範囲は 最小 < 最大 にしてください。入れ替えて計算します。")
        lo, hi = min(lo, hi), max(lo, hi) + (step if lo == hi else 0)
    return lo, hi


# 推進剤比較タブでオン/オフできる組 (酸化剤 / 燃料)
CMP_PAIRS = ["N2O / ABS", "LOX / ABS", "LOX / LH2", "LOX / CH4", "LOX / RP1", "LOX / Ethanol",
             "H2O2 / RP1", "N2O4 / UDMH", "N2O4 / MMH", "IRFNA / UDMH"]


def card_name(prefix, card):
    return prefix + hashlib.md5(card.encode()).hexdigest()[:8]


# ============================================================
# パラメータ入力 (ページ上部 + 左のサイドメニューに同じものを表示)
# ============================================================
CUSTOM = "(カスタムカード)"


def render_inputs(p):
    num = lambda lo, hi, step, **kw: dict(min_value=lo, max_value=hi, step=step, **kw)  # noqa: E731
    top = p.cols([1.15, 1])

    # --- ① 推進剤 ---
    with top[0].container(border=True):
        section("① 推進剤", "推進剤の種類と燃焼モデル", "#e8590c")
        c = p.cols([1.1, 1, 1])
        prop_mode = p.w("radio", "推進剤タイプ", "prop_mode", "二液 (酸化剤 / 燃料)",
                        ["二液 (酸化剤 / 燃料)", "一液 (カスタムカード)"], where=c[0])
        mono = prop_mode.startswith("一液")
        if not mono:
            ox_sel = p.w("selectbox", "酸化剤", "ox_sel", "N2O", OX_LIST + [CUSTOM], where=c[1])
            fuel_sel = p.w("selectbox", "燃料", "fuel_sel", "ABS", FUEL_LIST + [CUSTOM], where=c[2])
        else:
            c[1].caption("一液推進剤ではカードで組成を指定します (O/F は無視されます)。")

        c = p.cols([1.1, 1, 1])
        p.w("selectbox", "流れモデル", "flow_label", list(FLOW_MODES)[0], list(FLOW_MODES), where=c[0])
        use_fac = p.w("toggle", "有限断面積燃焼室", "use_fac", False, where=c[1],
                      help="fac_CR (燃焼室断面積 / スロート断面積) を指定")
        if use_fac:
            p.w("number_input", "縮流比 fac_CR", "fac_CR", 2.5, where=c[2], **num(1.0, 1000.0, 0.1))

        if mono:
            card_fields(p, "mono", "一液推進剤カード")
        else:
            if ox_sel == CUSTOM:
                card_fields(p, "ox", "酸化剤カード")
            if fuel_sel == CUSTOM:
                card_fields(p, "fuel", "燃料カード")

    # --- ② 作動点 ---
    with top[1].container(border=True):
        section("② 作動点", "上部の数値と CEA 出力・ΔV に使う条件", "#1c7ed6")
        c = p.cols(3)
        p.w("number_input", "燃焼室圧力 Pc [MPa]", "Pc", 2.0, where=c[0], **num(0.05, 100.0, 0.5))
        p.w("number_input", "混合比 O/F", "MR", 5.0, where=c[1], **num(0.1, 50.0, 0.05, disabled=mono))
        p.w("number_input", "ノズル開口比 ε", "eps", 5.0, where=c[2], **num(1.0, 500.0, 1.0))
        c = p.cols(3)
        p.w("number_input", "外気圧 Pamb [MPa]", "Pamb", 0.1013, where=c[0],
            **num(0.0, 1.0, 0.001, format="%.4f", help="比推力 Isp_SL はこの外気圧で計算します"))
        p.w("number_input", "Isp 効率", "eff_isp", 0.90, where=c[1], **num(0.5, 1.0, 0.01))

    # --- ③ スイープ範囲 ---
    with st.container(border=True):
        section("③ グラフの範囲", "横軸の範囲 (最小〜最大)。ε の系列は「② 性能特性」で横軸 Pc のときに使用", "#2f9e44")
        g = p.cols(3)
        with g[0].container(border=True):
            st.markdown('<div class="sub">O/F</div>', unsafe_allow_html=True)
            c = p.cols(2)
            range_input(p, c[0], c[1], "O/F", 0.1, 50.0, 0.1, 10.0, 0.5, disabled=mono)
        with g[1].container(border=True):
            st.markdown('<div class="sub">燃焼室圧力 Pc [MPa]</div>', unsafe_allow_html=True)
            c = p.cols(2)
            range_input(p, c[0], c[1], "Pc", 0.05, 100.0, 1.0, 5.0, 0.5)
        with g[2].container(border=True):
            st.markdown('<div class="sub">ノズル開口比 ε</div>', unsafe_allow_html=True)
            c = p.cols([1, 1, 1.4])
            range_input(p, c[0], c[1], "ε", 1.0, 500.0, 1.0, 10.0, 1.0)
            p.w("text_input", "系列", "eps_list", "3, 5, 7, 9", where=c[2],
                help="「② 性能特性」で横軸を Pc にしたとき、この ε ごとに線を引き分けます (カンマ区切り)")

    bottom = p.cols([1, 1.6])

    # --- ④ ΔV 計算 ---
    with bottom[0].container(border=True):
        section("④ ΔV 計算", "結果は「② 性能特性」タブの表", "#7048e8")
        if p.w("toggle", "ΔV 計算を行う", "use_dv", False):
            c = p.cols(3)
            p.w("number_input", "ペイロード [kg]", "w_payload", 30.0, where=c[0], **num(0.0, 1e7, 1.0))
            p.w("number_input", "構造 [kg]", "w_stage", 350.0, where=c[1], **num(0.0, 1e7, 1.0))
            p.w("number_input", "推進剤 [kg]", "w_prop", 800.0, where=c[2], **num(0.0, 1e7, 1.0))

    # --- ⑤ 推進剤比較 ---
    with bottom[1].container(border=True):
        section("⑤ 推進剤比較", "結果は「④ 推進剤比較」タブ", "#c2255c")
        c = p.cols([1.2, 1, 1])
        if p.w("toggle", "推進剤比較を行う", "use_cmp", False, where=c[0]):
            p.w("number_input", "比較用 Pc [MPa]", "cmp_pc", 5.0, where=c[1], **num(0.1, 50.0, 0.5))
            p.w("number_input", "比較用 ε", "cmp_eps", 15.0, where=c[2], **num(1.0, 300.0, 1.0))
            cc = p.cols(5)
            for i, pair in enumerate(CMP_PAIRS):
                p.w("checkbox", pair, f"cmp_{pair}", True, where=cc[i % 5])


with st.sidebar:
    st.markdown("### ☰ 設定")
    st.caption("ページ上部の入力パネルと同じ内容です。どちらで変更しても両方に反映されます。")
    render_inputs(Panel("side", wide=False))
render_inputs(Panel("top", wide=True))

# 入力値 (上部とサイドメニューで共通)
S = st.session_state
mono = S["prop_mode"].startswith("一液")
ox_sel = fuel_sel = ox_card = fuel_card = mono_card = None
try:
    if mono:
        mono_card = build_card("mono")
    else:
        ox_sel, fuel_sel = S["ox_sel"], S["fuel_sel"]
        ox_card = build_card("ox") if ox_sel == CUSTOM else None
        fuel_card = build_card("fuel") if fuel_sel == CUSTOM else None
except ValueError as e:
    st.error(str(e))
    st.stop()
flow_label = S["flow_label"]
fac_CR = S["fac_CR"] if S["use_fac"] else None
Pc, MR, eps, Pamb, eff_isp = S["Pc"], S["MR"], S["eps"], S["Pamb"], S["eff_isp"]
mr_rng, pc_rng, eps_rng = get_range("O/F", 0.5), get_range("Pc", 0.5), get_range("ε", 1.0)
try:
    eps_list = [float(v) for v in S["eps_list"].replace("、", ",").split(",") if v.strip()] or [3, 5, 7, 9]
except ValueError:
    st.warning(f"ε 系列を解釈できません: '{S['eps_list']}'。3, 5, 7, 9 を使用します。")
    eps_list = [3, 5, 7, 9]
use_dv, use_cmp = S["use_dv"], S["use_cmp"]
if use_dv:
    w_payload, w_stage, w_prop = S["w_payload"], S["w_stage"], S["w_prop"]
cmp_pairs = []
if use_cmp:
    cmp_pc, cmp_eps = S["cmp_pc"], S["cmp_eps"]
    cmp_pairs = [pair for pair in CMP_PAIRS if S[f"cmp_{pair}"]]

st.markdown('<div class="res">計算結果</div>', unsafe_allow_html=True)

frozen, frozenAtThroat = FLOW_MODES[flow_label]
FZ = dict(frozen=frozen, frozenAtThroat=frozenAtThroat)

# ============================================================
# CEA オブジェクト生成
# ============================================================
UNITS = dict(cstar_units="m/s", pressure_units="MPa", temperature_units="K",
             sonic_velocity_units="m/s", enthalpy_units="kJ/kg", density_units="kg/m^3",
             specific_heat_units="kJ/kg-K")


@st.cache_resource(show_spinner=False)
def get_cea(propName, oxName, fuelName, fac_CR, cards):
    with CEA_LOCK:
        for kind, name, card in cards:
            {"ox": add_new_oxidizer, "fuel": add_new_fuel, "prop": add_new_propellant}[kind](name, card)
        kw = dict(propName=propName, oxName=oxName, fuelName=fuelName, fac_CR=fac_CR)
        return CEA_Obj(**kw, **UNITS), CEA_Obj_raw(**kw)


def _purge_nan_cache():
    """RocketCEA の結果キャッシュから nan の項目だけを消す。
    (clearCache() はキャッシュ表そのものを消してしまい、以後 get_Isp / get_Cstar が全部失敗するので使わない)"""
    for cache in cea_mod.getCacheDict().values():
        for d in (cache.ispDict, cache.cstarDict, cache.tcDict):
            for key in [k for k, v in d.items() if isinstance(v, float) and v != v]:
                del d[key]


def cea_call(f, *a, **k):
    """RocketCEA の呼び出しは必ずこれを通す (全セッション共通のロックで 1 件ずつ実行)"""
    with CEA_LOCK:
        return f(*a, **k)


cards = []
if mono:
    propName, oxName, fuelName = card_name("Mono_", mono_card), "", ""
    cards.append(("prop", propName, mono_card))
else:
    propName = ""
    oxName, fuelName = ox_sel, fuel_sel
    if ox_card:
        oxName = card_name("Ox_", ox_card)
        cards.append(("ox", oxName, ox_card))
    if fuel_card:
        fuelName = card_name("Fuel_", fuel_card)
        cards.append(("fuel", fuelName, fuel_card))

try:
    cea, cea_raw = get_cea(propName, oxName, fuelName, fac_CR, tuple(cards))
except Exception as e:  # カード記述ミスなど
    st.error(f"CEA オブジェクトの生成に失敗しました: {e}")
    st.stop()

MRv = 1.0 if mono else MR

# グラフ用: 無限断面積燃焼室 (実線) と、有限断面積燃焼室オン時はそれを同色の破線で重ねる
cea_inf = get_cea(propName, oxName, fuelName, None, tuple(cards))[0]
base_desc = cea_inf.desc
desc = base_desc + (f", fac_CR={fnum(fac_CR)}" if fac_CR else "")
VARIANTS = [(cea_inf, "solid", "")] + ([(cea, "dash", f" (fac_CR={fnum(fac_CR)})")] if fac_CR else [])
COLORS = pc.qualitative.Plotly


def safe(f, *a, **k):
    for attempt in range(2):
        try:
            v = cea_call(f, *a, **k)
            v = v[0] if isinstance(v, tuple) else v
        except Exception:
            return np.nan
        # RocketCEA は一度 nan になった結果をキャッシュして返し続けるので、nan の記録だけ消して 1 回だけ再計算する
        if attempt == 0 and isinstance(v, float) and np.isnan(v):
            cea_call(_purge_nan_cache)
            continue
        return v
    return v


def isp_amb(Pc_, MR_, eps_, obj=None, **fz):
    return safe((obj or cea).estimate_Ambient_Isp, Pc_, MR_, eps_, Pamb=Pamb, **(fz or FZ))


def add_series(fig, x, y_of, name, color, **line_kw):
    """y_of(obj) の値を VARIANTS ごとに描く。同じ系列の実線と破線は同色・同じ凡例グループにまとめる
    (凡例の並び用)。クリックで注目する単位は meta で、実線と破線を別々に選べる。"""
    for obj, dash, suffix in VARIANTS:
        fig.add_trace(go.Scatter(x=x, y=np.asarray(y_of(obj), dtype=float), mode="lines", name=name + suffix,
                                 legendgroup=name, meta=name + suffix, line=dict(color=color, dash=dash, **line_kw)))


def fig_layout(fig, title, xt, yt, **kw):
    fig.update_layout(title=dict(text=title, font=dict(size=14)), xaxis_title=xt, yaxis_title=yt,
                      hovermode="x unified", height=480, margin=dict(l=60, r=20, t=50, b=50),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), dragmode=False, **kw)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


mr_arr = np.arange(mr_rng[0], mr_rng[1] + 1e-9, 0.05)
pc_arr = np.linspace(pc_rng[0], pc_rng[1], 60)

# ============================================================
# 作動点サマリー
# ============================================================
m = st.columns(7)
isp_vac = safe(cea.get_Isp, Pc, MRv, eps, **FZ)
isp_sl = isp_amb(Pc, MRv, eps)
try:
    _, _, Tc, mw, gam = cea_call(cea.get_IvacCstrTc_ChmMwGam, Pc=Pc, MR=MRv, eps=eps)
except Exception:
    Tc = mw = gam = np.nan
cstar = safe(cea.get_Cstar, Pc, MRv)
pe = Pc / safe(cea.get_PcOvPe, Pc, MRv, eps, **FZ)
m[0].metric("比推力 Isp_vac [s]", f"{isp_vac:.1f}")
m[1].metric("比推力 Isp_SL [s]", f"{isp_sl:.1f}")
m[2].metric("c* [m/s]", f"{cstar:.1f}")
m[3].metric("Tc [K]", f"{Tc:.1f}")
m[4].metric("燃焼ガス平均分子量 [g/mol]", f"{mw:.2f}",
            help="燃焼室内の燃焼ガス (生成物の混合気体) の平均分子量。小さいほど比推力が高くなりやすい")
m[5].metric("γ (燃焼室)", f"{gam:.4f}")
m[6].metric("出口圧 Pe [MPa]", f"{pe:.4f}")
st.caption(f"{desc} ― Pc={fnum(Pc)} MPa, O/F={fnum(MRv)}, ε={fnum(eps)}, {flow_label}")

tabs = st.tabs(["① CEA 出力", "② 性能特性", "③ 平衡流 / 凍結流", "④ 推進剤比較"])

# ① sample01 / sample03 : NASA CEA フル出力
with tabs[0]:
    short = st.toggle("短縮出力 (short_output)", value=True)
    with st.spinner("CEA 計算中..."):
        out = cea_call(cea_raw.get_full_cea_output, Pc=Pc * 10.0, MR=MRv, eps=eps, short_output=int(short),
                                          pc_units="bar", output="siunits", fac_CR=fac_CR, **FZ)
    st.code(out, language=None)
    st.download_button("テキストを保存", out, file_name="cea_output.txt")

# ② sample02 / 04 / 05 : 性能特性 (横軸 O/F または ε)
QTY = {  # 表示名: (列名, 書式)
    "比推力 Isp_vac [s]": ("Isp_vac", "{:.1f}"),
    "比推力 Isp_SL [s]": ("Isp_SL", "{:.1f}"),
    "c* [m/s]": ("cstar", "{:.1f}"),
    "燃焼温度 Tc [K]": ("Tc", "{:.1f}"),
    "出口圧 Pe [MPa]": ("Pe", "{:.4f}"),
    "燃焼ガス平均分子量 [g/mol]": ("MolWt", "{:.2f}"),
    "γ (燃焼室)": ("gamma", "{:.4f}"),
}
ISP_BOTH = "比推力 Isp_vac / Isp_SL [s]"
CHARTS = {ISP_BOTH: ["Isp_vac", "Isp_SL"],  # グラフ名: 描く列 (1 グラフに重ねる)
          **{k: [v[0]] for k, v in QTY.items()}}
PEAK_COLS = {"Isp_vac", "Isp_SL", "cstar"}  # グラフに頂点マークを付ける列


def perf_point(Pc_, MR_, eps_, obj=None):
    obj = obj or cea
    try:
        _, cs, tc, mw_, g = cea_call(obj.get_IvacCstrTc_ChmMwGam, Pc=Pc_, MR=MR_, eps=eps_)
    except Exception:
        cs = tc = mw_ = g = np.nan
    return dict(Isp_vac=safe(obj.get_Isp, Pc_, MR_, eps_, **FZ), Isp_SL=isp_amb(Pc_, MR_, eps_, obj),
                cstar=cs, Tc=tc, Pe=Pc_ / safe(obj.get_PcOvPe, Pc_, MR_, eps_, **FZ),
                MolWt=mw_, gamma=g)


# 横軸: 表示名 → (短い名前, 範囲, 作動点 (Pc, O/F, ε) のどれを置き換えるか)
X_AXES = {"混合比 O/F": ("O/F", mr_rng, 1), "ノズル開口比 ε": ("ε", eps_rng, 2),
          "燃焼室圧力 Pc [MPa]": ("Pc", pc_rng, 0)}


def point_at(x_idx, x, s=None):
    """横軸以外は作動点の値に固定した (Pc, O/F, ε)。s は ε 系列の値"""
    p = [Pc, MRv, eps]
    p[x_idx] = x
    if s is not None:
        p[2] = s
    return p


_curve_cache = {}


def get_curves(x_axis, use_series):
    """横軸ごとの計算結果。同じ横軸のグラフが複数あっても 1 回だけ計算する"""
    key = (x_axis, use_series)
    if key not in _curve_cache:
        _, x_rng, x_idx = X_AXES[x_axis]
        xs = np.linspace(*x_rng, 100)
        series = eps_list if use_series else [None]
        _curve_cache[key] = xs, {(s, obj): pd.DataFrame([perf_point(*point_at(x_idx, x, s), obj) for x in xs])
                                  for s in series for obj, _, _ in VARIANTS}
    return _curve_cache[key]


def fixed_text(x_idx, use_series):
    return ", ".join(s for i, s in enumerate([f"Pc={fnum(Pc)} MPa", f"O/F={fnum(MRv)}", f"ε={fnum(eps)}"])
                     if i != x_idx and not (mono and i == 1) and not (use_series and i == 2))


def perf_chart(x_axis, q, use_series):
    x_lab, _, x_idx = X_AXES[x_axis]
    series = eps_list if use_series else [None]
    xs, curves = get_curves(x_axis, use_series)
    fixed = fixed_text(x_idx, use_series)
    fig = go.Figure()
    lines = [(col, s) for col in CHARTS[q] for s in series]
    for n, (col, s) in enumerate(lines):
        name, color = (col if s is None else f"ε={fnum(s)}"), COLORS[n % len(COLORS)]
        add_series(fig, xs, lambda obj: curves[s, obj][col], name, color, width=2.5)
        for obj, dash, suffix in VARIANTS:
            y = curves[s, obj][col].to_numpy(dtype=float)
            k = int(np.nanargmax(y)) if col in PEAK_COLS and np.isfinite(y).any() else -1
            if 0 < k < len(xs) - 1:  # 範囲の端ではなく山になっているときだけ頂点を示す
                fig.add_trace(go.Scatter(
                    x=[xs[k]], y=[y[k]], mode="markers+text", showlegend=False, legendgroup=name, meta=name + suffix,
                    marker=dict(color=color if dash == "solid" else "white", size=11, symbol="circle",
                                line=dict(color="white" if dash == "solid" else color, width=2)),
                    text=[f"{y[k]:.1f} ({x_lab}={xs[k]:.2f})"],
                    textposition="top center" if dash == "solid" else "bottom center",
                    textfont=dict(color=color), hoverinfo="skip"))
    if q == "出口圧 Pe [MPa]":
        fig.add_hline(y=Pamb, line_dash="dash", line_color="gray",
                      annotation_text=f"Pamb={fnum(Pamb)} MPa", annotation_position="top right")
        # Pe = Pamb (適正膨張) の点: 符号が変わる区間を線形補間
        for s in series:
            for obj, dash, suffix in VARIANTS:
                d = curves[s, obj]["Pe"].to_numpy(dtype=float) - Pamb
                for k in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
                    xc = xs[k] - d[k] * (xs[k + 1] - xs[k]) / (d[k + 1] - d[k])
                    head = "Pe=Pamb" if s is None else f"ε={fnum(s)}"
                    fig.add_trace(go.Scatter(
                        x=[xc], y=[Pamb], mode="markers+text", showlegend=False, hoverinfo="skip",
                        marker=dict(color="#d6336c", size=11, symbol="diamond", line=dict(color="white", width=2)),
                        text=[f"{head}{suffix} ({x_lab}={xc:.2f})"],
                        textposition="top right" if dash == "solid" else "bottom right",
                        textfont=dict(color="#d6336c")))
    both = len(CHARTS[q]) > 1
    title = "比推力 Isp_vac / Isp_SL" if both else q.split(" [")[0]
    y_title = "比推力 [s]" if both else q
    fig = fig_layout(fig, f"{title} ({base_desc}, {fixed})", x_axis, y_title)
    fig.update_layout(showlegend=len(lines) > 1 or len(VARIANTS) > 1)
    return fig


def _generate_charts(x_axis, use_series):
    """「生成」: 選んだ量のグラフを追加し、表示する量の選択を空に戻す (ボタンのコールバックなので
    ウィジェット描画前に実行され、q_sel を書き換えられる)"""
    ss = st.session_state
    for q in ss.get("q_sel", []):
        if not any(g["x"] == x_axis and g["q"] == q and g["series"] == use_series for g in ss.perf_charts):
            ss.perf_charts.append(dict(id=f"g{len(ss.perf_charts)}_{abs(hash((x_axis, q, use_series)))}",
                                       x=x_axis, q=q, series=use_series))
    ss.q_sel = []


def _remove_chart(cid):
    st.session_state.perf_charts = [c for c in st.session_state.perf_charts if c["id"] != cid]


with tabs[1]:
    S.setdefault("perf_charts", [])  # 生成済みグラフ: {"id", "x", "q", "series"} のリスト
    c = st.columns([1.4, 3, 0.9])
    x_opts = [k for k in X_AXES if not (mono and k == "混合比 O/F")]
    x_axis = c[0].radio("横軸", x_opts, horizontal=True, key="perf_x")
    x_lab, x_rng, x_idx = X_AXES[x_axis]
    x_is_mr = x_lab == "O/F"
    q_sel = c[1].multiselect("表示する量", list(CHARTS), key="q_sel",
                             placeholder="グラフにする量を選んでください")
    # 例外: 横軸 Pc のときだけ ε 系列で線を引き分けられる
    use_series = x_lab == "Pc" and c[0].toggle("ε 系列で線を引き分ける", value=True, key="perf_series",
                                               help="ε の値は上部「③ グラフの範囲」の ε 系列で指定")
    c[2].button("生成", type="primary", disabled=not q_sel, width="stretch", on_click=_generate_charts,
                args=(x_axis, use_series))
    if S.perf_charts and c[2].button("すべて削除", width="stretch"):
        S.perf_charts = []

    if not S.perf_charts:
        st.info("横軸と表示する量を選んで「生成」を押すとグラフが追加されます。"
                "横軸を変えて続けて生成すると、O/F と ε のグラフなどを並べて比較できます。")
    else:
        st.caption("グラフは常に「② 作動点」「③ グラフの範囲」の現在の値で再計算されます。各グラフの × で削除できます。")
    items = []  # (グラフ id, 横軸, 表示する量, ε 系列)
    for g in S.perf_charts:
        if mono and g["x"] == "混合比 O/F":
            continue
        # ε 系列表示のときは Isp_vac / Isp_SL を別々のグラフに分ける (線が多くなりすぎるため)
        qs = ["比推力 Isp_vac [s]", "比推力 Isp_SL [s]"] if g["q"] == ISP_BOTH and g["series"] else [g["q"]]
        items += [(g["id"], g["x"], q, g["series"]) for q in qs]
    cols = st.columns(2)
    with st.spinner("計算中..."):
        for i, (cid, gx, q, gs) in enumerate(items):
            with cols[i % 2]:
                h = st.columns([6, 1])
                h[0].caption(f"横軸: {gx}" + (" / ε 系列" if gs else ""))
                h[1].button("×", key=f"del_{cid}_{i}", on_click=_remove_chart, args=(cid,), help="このグラフを削除")
                st.plotly_chart(perf_chart(gx, q, gs), width="stretch", config=PLOT_CFG, key=f"chart_{cid}_{i}")

    # 作動点 (Pc, O/F, ε) を基準に横軸だけ振った表
    step_xs = np.round(np.arange(mr_rng[0], mr_rng[1] + 1e-9, 0.1), 3) if x_is_mr else \
        np.round(np.linspace(*x_rng, 19), 3)
    df = pd.DataFrame([perf_point(*point_at(x_idx, x)) for x in step_xs])
    x_col = "Pc [MPa]" if x_lab == "Pc" else x_lab
    df.insert(0, x_col, step_xs)
    fmt = {x_col: fnum, **{v[0]: v[1] for v in QTY.values()}}
    op_fixed = ", ".join(s for i, s in enumerate([f"Pc={fnum(Pc)} MPa", f"O/F={fnum(MRv)}", f"ε={fnum(eps)}"])
                         if i != x_idx and not (mono and i == 1))
    caption = f"**作動点 ({op_fixed}) で {x_lab} を変えたときの値**"
    if use_dv:
        w_init = w_stage + w_prop + w_payload
        w_final = w_init - w_prop
        df["Isp_vac実測"] = eff_isp * df["Isp_vac"]
        df["Isp_SL実測"] = eff_isp * df["Isp_SL"]
        df["ΔV [m/s]"] = G0 * df["Isp_vac実測"] * np.log(w_init / w_final) if w_final > 0 else np.nan
        fmt.update({"Isp_vac実測": "{:.1f}", "Isp_SL実測": "{:.1f}", "ΔV [m/s]": "{:.1f}"})
        caption += (f" ― ΔV: 初期質量 {fnum(w_init)} kg / 最終質量 {fnum(w_final)} kg, Isp 効率 {fnum(eff_isp)} "
                    "(真空比推力ベース)")
    st.markdown(caption)
    st.dataframe(df.style.format(fmt), width="stretch", hide_index=True)


# ⑤ sample06 : 平衡流 / 凍結流の比較
with tabs[2]:
    # 一液は Pc を、二液は O/F を横軸にする (それ以外は作動点に固定)
    xt, xv = ("燃焼室圧力 Pc [MPa]", pc_arr) if mono else ("混合比 O/F", mr_arr)
    args = (lambda x: (x, MRv, eps)) if mono else (lambda x: (Pc, x, eps))
    res = {}  # (流れモデル, obj) → (Isp_vac, Isp_SL)
    for obj, _, _ in VARIANTS:
        for lab, fz in FLOW_MODES.items():
            kw = dict(frozen=fz[0], frozenAtThroat=fz[1])
            vac = np.array([safe(obj.get_Isp, *args(x), **kw) for x in xv], dtype=float)
            sl = np.array([isp_amb(*args(x), obj, **kw) for x in xv], dtype=float)
            bad = ~(vac > 0)  # 凍結流の計算が収束しないと CEA は 0 を返すので空欄にする
            vac[bad] = sl[bad] = np.nan
            res[lab, obj] = (vac, sl)
    c1, c2 = st.columns(2)
    for i, (col_, y_title) in enumerate([(c1, "比推力 Isp_vac [s]"), (c2, "比推力 Isp_SL [s]")]):
        fig = go.Figure()
        for j, lab in enumerate(FLOW_MODES):
            add_series(fig, xv, lambda obj: res[lab, obj][i], lab.split(" (")[0], COLORS[j])
        cond = f"ε={fnum(eps)}" if mono else f"Pc={fnum(Pc)} MPa, ε={fnum(eps)}"
        col_.plotly_chart(fig_layout(fig, f"{y_title.split(' [')[0]} ({base_desc}, {cond})", xt, y_title),
                          width="stretch", config=PLOT_CFG)
    st.caption("実線: 無限断面積燃焼室 / 破線: 有限断面積燃焼室 (上部「① 推進剤」の「有限断面積燃焼室」をオンにすると表示)")

# ⑦ sample11 : 推進剤比較 (かさ密度 vs Isp)
def show_comparison():
    fig = go.Figure()
    skipped = []
    with st.spinner("推進剤ごとに計算中..."):
        for j, pair in enumerate(cmp_pairs):
            o, f = [s.strip() for s in pair.split("/")]
            try:
                lim = cea_call(MR_Temperature_Limits, oxName=o, fuelName=f, MR_MAX=20.0)
                mr_min = max(lim.min_MR, 1.0)
                mrs = np.arange(mr_min, lim.max_MR, 0.1)
                rho = np.asarray(bulkDensity(o, f, mrs)) * 1000.0
                lines = []
                for fac, dash, suffix in [(None, "solid", "")] + ([(fac_CR, "dash", f" (fac_CR={fnum(fac_CR)})")]
                                                                  if fac_CR else []):
                    obj, _ = get_cea("", o, f, fac, ())
                    lines.append(([safe(obj.get_Isp, Pc=cmp_pc, MR=mr, eps=cmp_eps) for mr in mrs], dash, suffix))
            except Exception:
                skipped.append(pair)
                continue
            for isp, dash, suffix in lines:
                fig.add_trace(go.Scatter(x=rho, y=isp, mode="lines", name=pair + suffix, legendgroup=pair, meta=pair + suffix,
                                         line=dict(color=COLORS[j % len(COLORS)], dash=dash),
                                         customdata=mrs, hovertemplate="ρ=%{x:.1f} kg/m³<br>Isp_vac=%{y:.1f} s"
                                                                       "<br>O/F=%{customdata:.2f}"))
    fig = fig_layout(fig, f"Pc={fnum(cmp_pc)} MPa, ε={fnum(cmp_eps)} (平衡流)", "推進剤かさ密度 [kg/m³]",
                     "比推力 Isp_vac [s]")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    if skipped:
        st.caption("密度データがない等で計算できなかった組: " + ", ".join(skipped))


with tabs[3]:
    if not use_cmp:
        st.caption("推進剤比較はオフです (上部の「推進剤比較」でオン)。")
    elif not cmp_pairs:
        st.caption("比較する推進剤の組を上部で 1 つ以上オンにしてください。")
    else:
        show_comparison()
