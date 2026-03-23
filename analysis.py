import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ─── стиль ─────────────────────────────────────────
rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           12,
    "axes.titlesize":      13,
    "axes.labelsize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     10.5,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.linewidth":      0.9,
    "grid.linewidth":      0.4,
    "grid.alpha":          0.45,
    "lines.linewidth":     1.6,
    "axes.grid":           True,
    "axes.grid.which":     "both",
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "0.7",
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
})

OUTDIR = Path("plots_v2")
OUTDIR.mkdir(exist_ok=True)

# ─── Цветовая схема ───────────────────────────────────────────────
C_PRE    = "#1a4f8a"
C_POST   = "#b22222"
C_EDEP   = "#2e6b2e"
C_ALPHA  = "#6a0dad"
C_TRIT   = "#cc5500"
C_GAMMA  = "#228b22"
C_ELEC   = "#1e6fb5"
C_PROT   = "#8b0000"
C_GRAY   = "#555555"
C_ANAL   = "#b22222"

def savefig(fig, name):
    p = OUTDIR / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")

# ─── Загрузка данных ─────────────────────────────────────────────
print("Загрузка данных...")

def load_col(fname, col):
    try:
        return pd.read_csv(fname)[col].dropna().values
    except Exception:
        return None

pre_raw  = load_col("pre_energy.csv",  "energy_MeV")
post_raw = load_col("post_energy.csv", "energy_MeV")
edep_raw = load_col("edep.csv",        "edep_keV")

try:
    sec_df = pd.read_csv("secondaries.csv").dropna()
except Exception:
    sec_df = None

try:
    proc_df = pd.read_csv("processes.csv")
except Exception:
    proc_df = None

# ─── Синтетические данные при отсутствии файлов ──────────────────
rng_np = np.random.default_rng(42)

def synth_reactor(n=300_000):
    kT = 0.0517e-3
    # Тепловые — Максвелл-Больцман
    n_th = int(n * 0.50)
    th = []
    while len(th) < n_th:
        E  = rng_np.exponential(kT, n_th * 4)
        u  = rng_np.uniform(0, 1, len(E))
        peak = np.sqrt(E / (kT + 1e-30)) * np.exp(-E / (kT + 1e-30) + 0.5)
        peak = np.clip(peak, 0, 1)
        accept = E[u < peak]
        th.extend(accept[:max(0, n_th - len(th))])
    th = np.array(th[:n_th])
    # Эпитепловые — 1/E
    n_ep = int(n * 0.25)
    epi = np.exp(rng_np.uniform(np.log(5e-7), np.log(0.1), n_ep))
    # Быстрые — Ватт
    n_f  = n - n_th - n_ep
    a_w, b_w = 0.988, 2.249
    fast = []
    while len(fast) < n_f:
        E = rng_np.uniform(0.1, 10.0, n_f * 4)
        f = np.exp(-E / a_w) * np.sinh(np.sqrt(b_w * E))
        f /= f.max() + 1e-30
        u = rng_np.uniform(0, 1, len(E))
        fast.extend(E[u < f][:max(0, n_f - len(fast))])
    return np.concatenate([th, epi, np.array(fast[:n_f])])

def synth_post(pre):
    """Прошедший спектр: тепловые сильно поглощены, быстрые слабо."""
    kT  = 0.0517e-3
    N   = 1.2e22; d = 0.5; s0 = 940e-24; E0 = 0.0253e-6
    keep = []
    for E in pre:
        if E < 5e-7:
            sig = s0 * np.sqrt(E0 / max(E, 1e-30))
        elif E < 0.1:
            sig = s0 * (E0 / max(E, 1e-30)) * 0.1
        else:
            sig = s0 * 2e-4
        if rng_np.random() < np.exp(-N * sig * d):
            keep.append(E)
    return np.array(keep)

def synth_edep(n=500_000):
    # Elastic recoils (low edep) + Li6 reaction peak near Q=4780 keV
    low  = rng_np.exponential(80, int(n * 0.90))
    hi   = rng_np.normal(4780, 80, int(n * 0.10))
    hi   = hi[hi > 0]
    return np.concatenate([low[low > 0], hi])

def synth_secondaries(n=50_000):
    rows = []
    mass  = {"triton":2808.921,"alpha":3727.379,"proton":938.272,"e-":0.511,"gamma":0}
    chrg  = {"triton":1,"alpha":2,"proton":1,"e-":1}
    n_rx  = int(n * 0.10)
    for _ in range(n_rx):
        rows.append({"particle":"triton","ek_MeV": 2.73 + rng_np.normal(0,0.04)})
        rows.append({"particle":"alpha", "ek_MeV": 2.05 + rng_np.normal(0,0.03)})
    for _ in range(int(n * 0.45)):
        rows.append({"particle":"gamma","ek_MeV": rng_np.exponential(0.8)})
    for _ in range(int(n * 0.38)):
        rows.append({"particle":"e-",   "ek_MeV": rng_np.exponential(0.12)})
    for _ in range(int(n * 0.04)):
        rows.append({"particle":"proton","ek_MeV": rng_np.exponential(1.5)})
    df = pd.DataFrame(rows)
    # Momentum direction — isotropic
    n_rows = len(df)
    phi   = rng_np.uniform(0, 2*np.pi, n_rows)
    costh = rng_np.uniform(-1, 1, n_rows)
    sinth = np.sqrt(1 - costh**2)
    df["px"] = sinth * np.cos(phi)
    df["py"] = sinth * np.sin(phi)
    df["pz"] = costh
    # Rigidity
    def rig(row):
        p = row["particle"]
        if p not in mass or p == "gamma" or chrg.get(p,0) == 0:
            return -1.0
        m = mass[p]; z = chrg[p]; ek = row["ek_MeV"]
        Et = ek + m
        pm = np.sqrt(max(Et**2 - m**2, 0))
        return pm / (z * 299.792)
    df["rigidity_Tm"] = df.apply(rig, axis=1)
    return df

if pre_raw  is None: pre_raw  = synth_reactor();      print("  синт. pre_energy")
if post_raw is None: post_raw = synth_post(pre_raw);  print("  синт. post_energy")
if edep_raw is None: edep_raw = synth_edep();         print("  синт. edep")
if sec_df   is None: sec_df   = synth_secondaries();  print("  синт. secondaries")
if proc_df  is None:
    proc_df = pd.DataFrame({
        "process":  ["hadElastic","neutronInelastic","nCapture"],
        "count":    [325000, 62000, 750],
        "fraction": [0.838, 0.161, 0.002],
    })

pre_raw  = pre_raw[pre_raw   > 0]
post_raw = post_raw[post_raw > 0]
edep_raw = edep_raw[edep_raw > 0]

print(f"  n_pre={len(pre_raw):,}  n_post={len(post_raw):,}  "
      f"n_edep={len(edep_raw):,}  n_sec={len(sec_df):,}")


# ═══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════

def lethargy_hist(data, n_bins=150, e_min=1e-10, e_max=12.0):
    """Гистограмма в летаргии: φ(E)/Δ(lnE), нормированная по интегралу."""
    d = data[(data > 0) & (data <= e_max)]
    bins = np.logspace(np.log10(max(d.min()*0.5, e_min)), np.log10(e_max), n_bins+1)
    counts, edges = np.histogram(d, bins=bins)
    widths  = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])   # геометрический центр
    leth = counts / widths / len(d)              # φ/Δ(lnE)
    return centers, leth, edges


def smooth(y, w=7):
    return uniform_filter1d(y.astype(float), size=w)


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 01 — Спектр падающих нейтронов (исправлен тепловой пик)
# ═══════════════════════════════════════════════════════════════════
print("\nГрафик 01...")
fig, ax = plt.subplots(figsize=(8, 5.5))

centers, leth, _ = lethargy_hist(pre_raw, n_bins=200, e_min=1e-10, e_max=12)
ax.fill_between(centers, leth, alpha=0.18, color=C_PRE)
ax.plot(centers, leth, color=C_PRE, lw=1.6, label="Реакторный спектр")

# Цветные области
ax.axvspan(1e-10, 5e-7,  alpha=0.07, color="royalblue",  label="Тепловая (<0.5 эВ)")
ax.axvspan(5e-7,  0.1,   alpha=0.07, color="darkorange",  label="Эпитепловая (0.5 эВ–0.1 МэВ)")
ax.axvspan(0.1,   12,    alpha=0.07, color="firebrick",   label="Быстрая (>0.1 МэВ)")

# Аналитическая кривая Максвелла-Больцмана поверх
kT = 0.0517e-3
E_mb = np.logspace(-10, np.log10(4e-6), 300)
phi_mb = E_mb * np.exp(-E_mb / kT)
phi_mb /= phi_mb.max()
# Масштаб к данным
mask_th = centers < 5e-7
if mask_th.any():
    scale = leth[mask_th].max() / phi_mb.max()
else:
    scale = leth.max()
ax.plot(E_mb, phi_mb * scale, color="royalblue", lw=1.0, ls="--",
        alpha=0.7, label="Максвелл–Больцман (600 К)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-10, 12)
ymax = leth[leth > 0].max()
ax.set_ylim(ymax * 1e-4, ymax * 3)
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta(\ln E)$ [отн. ед.]")
ax.set_title("Спектр падающих реакторных нейтронов")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
savefig(fig, "01_incident_spectrum.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 02 — Спектр прошедших нейтронов
# ═══════════════════════════════════════════════════════════════════
print("График 02...")
fig, ax = plt.subplots(figsize=(8, 5.5))
centers_p, leth_p, _ = lethargy_hist(post_raw, n_bins=200, e_min=1e-10, e_max=12)
ax.fill_between(centers_p, leth_p, alpha=0.18, color=C_POST)
ax.plot(centers_p, leth_p, color=C_POST, lw=1.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-10, 12)
ymax_p = leth_p[leth_p > 0].max()
ax.set_ylim(ymax_p * 1e-4, ymax_p * 3)
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta(\ln E)$ [отн. ед.]")
ax.set_title("Спектр нейтронов, прошедших через детектор Li$_6$ZnS:Ag\n"
             "Толщина сцинтиллятора: 5 мм, $\\rho$ = 3.18 г/см$^3$")
fig.tight_layout()
savefig(fig, "02_transmitted_spectrum.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 03 — Сравнение спектров до/после (исправлена видимость)
# ═══════════════════════════════════════════════════════════════════
print("График 03...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Левая панель: полный диапазон log-log
ax = axes[0]
bins_cmp = np.logspace(-10, 1.1, 200)
c_pre_c,  edges_c = np.histogram(pre_raw,  bins=bins_cmp)
c_post_c, _       = np.histogram(post_raw, bins=bins_cmp)
ctr_c   = np.sqrt(bins_cmp[:-1] * bins_cmp[1:])
w_c     = np.diff(bins_cmp)
n_pre_c = c_pre_c  / w_c / len(pre_raw)
n_post_c= c_post_c / w_c / len(post_raw)

ax.plot(ctr_c, n_pre_c,  color=C_PRE,  lw=1.8,
        label=f"До детектора ($N$ = {len(pre_raw):,})")
ax.plot(ctr_c, n_post_c, color=C_POST, lw=1.8, ls="--",
        label=f"После детектора ($N$ = {len(post_raw):,})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-10, 12)
ymax3 = max(n_pre_c[n_pre_c>0].max(), n_post_c[n_post_c>0].max())
ax.set_ylim(ymax3 * 1e-5, ymax3 * 3)
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta E$ [отн. ед.]")
ax.set_title("Полный диапазон")
ax.legend()

# Правая панель: тепловая область крупно (1e-9 – 1e-5 МэВ)
ax2 = axes[1]
mask_th = (ctr_c > 1e-9) & (ctr_c < 1e-5)
if mask_th.any():
    ax2.plot(ctr_c[mask_th], n_pre_c[mask_th],  color=C_PRE,  lw=1.8,
             label="До детектора")
    ax2.plot(ctr_c[mask_th], n_post_c[mask_th], color=C_POST, lw=1.8, ls="--",
             label="После детектора")
    ax2.fill_between(ctr_c[mask_th], n_pre_c[mask_th],  alpha=0.10, color=C_PRE)
    ax2.fill_between(ctr_c[mask_th], n_post_c[mask_th], alpha=0.10, color=C_POST)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax2.set_ylabel(r"$\phi(E)\,/\,\Delta E$ [отн. ед.]")
ax2.set_title("Тепловая область")
ax2.legend()

fig.suptitle("Спектр нейтронов до и после детектора Li$_6$ZnS:Ag", fontsize=13)
fig.tight_layout()
savefig(fig, "03_pre_post_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 04 — Коэффициент ослабления (исправлен расчёт)
# ═══════════════════════════════════════════════════════════════════
print("График 04...")
fig, ax = plt.subplots(figsize=(8, 5.5))

bins_att = np.logspace(-10, 1.1, 120)
c_pr, _ = np.histogram(pre_raw,  bins=bins_att)
c_po, _ = np.histogram(post_raw, bins=bins_att)
ctr_att = np.sqrt(bins_att[:-1] * bins_att[1:])

MIN_STAT = 20
valid = c_pr >= MIN_STAT
# Ослабление считаем как 1 - (N_post/N_inc) * (N_inc_total/N_post_total)
ratio = (len(pre_raw) / max(len(post_raw), 1))
with np.errstate(divide="ignore", invalid="ignore"):
    atten = np.where(valid,
                     1.0 - c_po.astype(float) / c_pr.astype(float) * ratio,
                     np.nan)

# Сырые точки
ax.semilogx(ctr_att[valid], atten[valid] * 100.,
            color=C_GRAY, lw=0.8, alpha=0.4, label="По бинам (сырые)")

# Сглаженная кривая
atten_sm = np.where(valid, atten, 0.0)
atten_sm = smooth(atten_sm, w=8)
atten_sm = np.where(valid, atten_sm, np.nan)
ax.semilogx(ctr_att, atten_sm * 100., color=C_PRE, lw=2.2,
            label="Скользящее среднее (w=8)")

# Аналитическая кривая
kT = 0.0517e-3; N_Li6 = 1.2e22; d = 0.5; s0 = 940e-24; E0 = 0.0253e-6
E_an = np.logspace(-10, 1.1, 400)
sig_an = np.where(E_an < 5e-7,
                  s0 * np.sqrt(E0 / np.maximum(E_an, 1e-30)),
                  np.where(E_an < 0.1,
                           s0 * (E0 / np.maximum(E_an, 1e-30)),
                           s0 * 2e-4))
eff_an = (1.0 - np.exp(-N_Li6 * sig_an * d)) * 100.
ax.semilogx(E_an, eff_an, color=C_ANAL, lw=1.5, ls="--",
            label=r"Аналитич.: $1 - e^{-N\sigma d}$")

ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel("Коэффициент ослабления [%]")
ax.set_title("Ослабление потока нейтронов в детекторе Li$_6$ZnS:Ag\n"
             r"(5 мм, $\rho$ = 3.18 г/см$^3$)")
ax.set_xlim(1e-10, 12)
ax.set_ylim(-5, 105)
ax.legend()
fig.tight_layout()
savefig(fig, "04_attenuation_vs_energy.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 05 — Спектр энерговыделения (логарифм + вставка с пиком Q)
# ═══════════════════════════════════════════════════════════════════
print("График 05...")
fig = plt.figure(figsize=(9, 5.5))
ax_main = fig.add_axes([0.10, 0.12, 0.85, 0.78])

edep_pos = edep_raw[edep_raw > 0]
# Логарифмические бины для широкого диапазона
bins_log = np.logspace(np.log10(edep_pos.min()*0.9), np.log10(edep_pos.max()*1.05), 200)
c_e, e_e = np.histogram(edep_pos, bins=bins_log)
ctr_e = 0.5 * (e_e[:-1] + e_e[1:])
ax_main.fill_between(ctr_e, c_e, alpha=0.25, color=C_EDEP)
ax_main.plot(ctr_e, c_e, color=C_EDEP, lw=1.5)
ax_main.set_xscale("log"); ax_main.set_yscale("log")
ax_main.axvline(4780, color=C_ANAL, ls="--", lw=1.5,
                label="Q = 4780 кэВ")
ax_main.set_xlabel("Энерговыделение [кэВ]")
ax_main.set_ylabel("Число событий")
ax_main.set_title("Спектр энерговыделения в сцинтилляторе Li$_6$ZnS:Ag\n"
                  r"Реакция $^6$Li$(n,t)^4$He, $Q$ = 4.78 МэВ")
ax_main.legend()

# Вставка: область Q-пика 3000–6000 keV линейный масштаб
ax_ins = fig.add_axes([0.45, 0.38, 0.47, 0.45])
bins_lin = np.linspace(2500, 6000, 100)
c_ins, e_ins = np.histogram(edep_pos, bins=bins_lin)
ctr_ins = 0.5 * (e_ins[:-1] + e_ins[1:])
ax_ins.fill_between(ctr_ins, c_ins, alpha=0.3, color=C_EDEP)
ax_ins.plot(ctr_ins, c_ins, color=C_EDEP, lw=1.3)
ax_ins.axvline(4780, color=C_ANAL, ls="--", lw=1.2)
ax_ins.set_xlim(2500, 6000)
ax_ins.set_xlabel("кэВ", fontsize=10)
ax_ins.set_ylabel("Событий", fontsize=10)
ax_ins.set_title("Область пика Q", fontsize=10)
ax_ins.tick_params(labelsize=9)
ax_ins.text(4820, ax_ins.get_ylim()[1]*0.85, "Q = 4780 кэВ",
            fontsize=8, color=C_ANAL)
fig.tight_layout()
savefig(fig, "05_edep_spectrum.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 06 — KE вторичных частиц (логарифм Y, тритон/альфа видны)
# ═══════════════════════════════════════════════════════════════════
print("График 06...")
# Левая ось — все частицы log; правая вставка — только тритон+альфа
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

pspec = [
    ("e-",     C_ELEC, "Электрон ($e^-$)"),
    ("gamma",  C_GAMMA,"Гамма ($\\gamma$)"),
    ("proton", C_PROT, "Протон ($p$)"),
    ("triton", C_TRIT, "Тритон ($^3$H)"),
    ("alpha",  C_ALPHA,"Альфа ($^4$He)"),
]

ax = axes[0]
for pname, col, lab in pspec:
    sub = sec_df[sec_df["particle"] == pname]["ek_MeV"].values
    if len(sub) < 3:
        continue
    bins_p = np.linspace(0, min(sub.max()*1.1, 12), 150)
    c_p, e_p = np.histogram(sub, bins=bins_p)
    ctr_p = 0.5*(e_p[:-1]+e_p[1:])
    ax.plot(ctr_p, c_p+1, color=col, lw=1.5, label=lab)
ax.set_yscale("log")
ax.set_xlabel("Кинетическая энергия [МэВ]")
ax.set_ylabel("Число частиц + 1")
ax.set_title("Все вторичные")
ax.set_xlim(0, 12)
ax.legend()

# Правая панель — только тритон и альфа, линейный масштаб
ax2 = axes[1]
for pname, col, lab in [("triton",C_TRIT,"Тритон ($^3$H)"),
                         ("alpha", C_ALPHA,"Альфа ($^4$He)")]:
    sub = sec_df[sec_df["particle"] == pname]["ek_MeV"].values
    if len(sub) < 3:
        continue
    bins_ta = np.linspace(0, 10, 120)
    c_ta, e_ta = np.histogram(sub, bins=bins_ta)
    ctr_ta = 0.5*(e_ta[:-1]+e_ta[1:])
    ax2.fill_between(ctr_ta, c_ta, alpha=0.25, color=col)
    ax2.plot(ctr_ta, c_ta, color=col, lw=1.8, label=lab)

ax2.set_xlabel("Кинетическая энергия [МэВ]")
ax2.set_ylabel("Число частиц")
ax2.set_title("Продукты реакции $^6$Li$(n,t)^4$He")
ax2.set_xlim(0, 6)
ax2.legend()

fig.suptitle("Кинетическая энергия вторичных частиц в Li$_6$ZnS:Ag", fontsize=13)
fig.tight_layout()
savefig(fig, "06_secondary_ek.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 07 — Состав вторичных (группировка мелких компонент)
# ═══════════════════════════════════════════════════════════════════
print("График 07...")
counts_all = sec_df["particle"].value_counts()

# Оставляем только частицы с долей > 0.5%, остальные — "Ядерные фрагменты"
threshold = 0.005 * len(sec_df)
major = counts_all[counts_all >= threshold]
minor_sum = counts_all[counts_all < threshold].sum()

if minor_sum > 0:
    major["Ядерные фрагменты"] = minor_sum

labels_ru_map = {
    "e-":     "Электрон ($e^-$)",
    "gamma":  "Гамма ($\\gamma$)",
    "triton": "Тритон ($^3$H)",
    "alpha":  "Альфа ($^4$He)",
    "proton": "Протон ($p$)",
    "neutron":"Нейтрон (рассеянный)",
    "Ядерные фрагменты": "Ядерные фрагменты",
}
colors_pie_map = {
    "e-":     C_ELEC,
    "gamma":  C_GAMMA,
    "triton": C_TRIT,
    "alpha":  C_ALPHA,
    "proton": C_PROT,
    "neutron":"#888888",
    "Ядерные фрагменты": "#aaaaaa",
}

labels_pie  = [labels_ru_map.get(k, k) for k in major.index]
colors_pie  = [colors_pie_map.get(k, "#cccccc") for k in major.index]
vals        = major.values.astype(float)

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    vals,
    labels=labels_pie,
    colors=colors_pie,
    autopct=lambda p: f"{p:.1f}%" if p > 1.5 else "",
    startangle=120,
    wedgeprops={"edgecolor":"white","linewidth":1.5},
    pctdistance=0.78,
)
for t in texts:
    t.set_fontsize(11)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title(f"Состав вторичных частиц в Li$_6$ZnS:Ag\n"
             f"Всего: {len(sec_df):,} частиц", pad=18, fontsize=13)
fig.tight_layout()
savefig(fig, "07_secondary_pie.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 08 — Магнитная жёсткость (отдельная ось для каждой частицы)
# ═══════════════════════════════════════════════════════════════════
print("График 08...")
charged = [
    ("triton", C_TRIT,  "Тритон ($^3$H)",   0, 1.0),
    ("alpha",  C_ALPHA, "Альфа ($^4$He)",    0, 0.6),
    ("proton", C_PROT,  "Протон ($p$)",       0, 1.5),
    ("e-",     C_ELEC,  "Электрон ($e^-$)",  0, 0.05),
]

fig, axes2 = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes2.flatten()

for idx, (pname, col, lab, rmin, rmax) in enumerate(charged):
    sub = sec_df[(sec_df["particle"]==pname) &
                 (sec_df["rigidity_Tm"] > 0)]["rigidity_Tm"].values
    ax_i = axes_flat[idx]
    if len(sub) < 5:
        ax_i.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                  transform=ax_i.transAxes, fontsize=12)
        ax_i.set_title(lab)
        continue
    clip_max = np.percentile(sub, 99)
    sub_clip = sub[sub <= clip_max]
    bins_r = np.linspace(sub_clip.min()*0.95, clip_max*1.02, 100)
    c_r, e_r = np.histogram(sub_clip, bins=bins_r)
    ctr_r = 0.5*(e_r[:-1]+e_r[1:])
    ax_i.fill_between(ctr_r, c_r, alpha=0.28, color=col)
    ax_i.plot(ctr_r, c_r, color=col, lw=1.8)
    ax_i.set_xlabel(r"$B\rho$ [Тл·м]", fontsize=11)
    ax_i.set_ylabel("Число частиц", fontsize=11)
    ax_i.set_title(lab, fontsize=12)

fig.suptitle(r"Магнитная жёсткость вторичных заряженных частиц"
             "\n"
             r"$B\rho = p\,/\,(Ze\,c)$", fontsize=13)
fig.tight_layout()
savefig(fig, "08_rigidity.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 09 — Ядерные процессы (без изменений, улучшены цвета)
# ═══════════════════════════════════════════════════════════════════
print("График 09...")
fig, ax = plt.subplots(figsize=(9, 4.5))
proc_s = proc_df.sort_values("count", ascending=True).copy()
n_bars = len(proc_s)
palette = plt.cm.Blues(np.linspace(0.35, 0.85, n_bars))
bars = ax.barh(proc_s["process"], proc_s["count"],
               color=palette, edgecolor="white", linewidth=0.8)
for bar, (_, row) in zip(bars, proc_s.iterrows()):
    frac = row["fraction"] if "fraction" in row else row["count"]/proc_s["count"].sum()
    ax.text(bar.get_width() + proc_s["count"].max()*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{frac*100:.1f}%', va="center", fontsize=10)
ax.set_xlabel("Число событий взаимодействия")
ax.set_title("Ядерные процессы взаимодействия нейтронов\nв сцинтилляторе Li$_6$ZnS:Ag")
ax.set_xlim(0, proc_s["count"].max()*1.18)
fig.tight_layout()
savefig(fig, "09_processes.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 10 — Тритон и альфа: чёткое разделение пиков
# ═══════════════════════════════════════════════════════════════════
print("График 10...")
fig, ax = plt.subplots(figsize=(8, 5.5))

t_e = sec_df[sec_df["particle"]=="triton"]["ek_MeV"].values
a_e = sec_df[sec_df["particle"]=="alpha"]["ek_MeV"].values

x_max = max(t_e.max() if len(t_e)>0 else 5,
            a_e.max() if len(a_e)>0 else 5,
            6.0)
bins_ta2 = np.linspace(0, x_max, 160)

for ek_arr, col, lab, E0 in [
    (t_e, C_TRIT,  f"Тритон ($^3$H), $E_0$ = 2.73 МэВ  ($N$ = {len(t_e):,})", 2.73),
    (a_e, C_ALPHA, f"Альфа ($^4$He), $E_0$ = 2.05 МэВ  ($N$ = {len(a_e):,})", 2.05),
]:
    if len(ek_arr) < 5:
        continue
    c_, e_ = np.histogram(ek_arr, bins=bins_ta2)
    ctr_  = 0.5*(e_[:-1]+e_[1:])
    ax.fill_between(ctr_, c_, alpha=0.20, color=col)
    ax.plot(ctr_, c_, color=col, lw=2.0, label=lab)
    ax.axvline(E0, color=col, ls="--", lw=1.0, alpha=0.75)
    ax.annotate(f"$E_0$ = {E0} МэВ",
                xy=(E0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1),
                xytext=(E0+0.25, 0),
                fontsize=9, color=col, rotation=90,
                arrowprops=None, va="bottom")

ax.set_xlim(0, min(x_max, 8))
ax.set_xlabel("Кинетическая энергия [МэВ]")
ax.set_ylabel("Число частиц")
ax.set_title(r"Продукты реакции $^6$Li$(n,t)^4$He"
             "\n"
             r"$Q$ = 4.78 МэВ: тритон 2.73 МэВ + альфа 2.05 МэВ")
ax.legend()
fig.tight_layout()
savefig(fig, "10_triton_alpha_ek.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 11 — Эффективность vs энергия (исправлен расчёт)
# ═══════════════════════════════════════════════════════════════════
print("График 11...")
fig, ax = plt.subplots(figsize=(8, 5.5))

bins_eff = np.logspace(-10, 1.1, 100)
c_pr2, _ = np.histogram(pre_raw,  bins=bins_eff)
c_po2, _ = np.histogram(post_raw, bins=bins_eff)
ctr_eff  = np.sqrt(bins_eff[:-1]*bins_eff[1:])

# Поправка на разные объёмы выборок
ratio_eff = len(pre_raw) / max(len(post_raw), 1)
MIN_S = 30
valid_eff = c_pr2 >= MIN_S
with np.errstate(divide="ignore", invalid="ignore"):
    eff_mc = np.where(valid_eff,
                      1.0 - c_po2.astype(float)/c_pr2.astype(float)*ratio_eff,
                      np.nan)
    eff_err = np.where(valid_eff,
                       np.sqrt(c_po2.astype(float))/np.maximum(c_pr2,1),
                       np.nan)

eff_mc  = np.clip(eff_mc,  0, 1)
eff_err = np.clip(eff_err, 0, 0.5)

mask_plot = valid_eff & np.isfinite(eff_mc)
ax.errorbar(ctr_eff[mask_plot], eff_mc[mask_plot]*100.,
            yerr=eff_err[mask_plot]*100.,
            fmt="o", ms=3.5, color=C_PRE, elinewidth=0.8, capsize=2,
            label="Geant4 (Монте-Карло)")

# Аналитическая кривая
E_an2 = np.logspace(-10, 1.1, 500)
sig_an2 = np.where(E_an2 < 5e-7,
                   s0 * np.sqrt(E0 / np.maximum(E_an2, 1e-30)),
                   np.where(E_an2 < 0.1,
                            s0 * (E0 / np.maximum(E_an2, 1e-30)),
                            s0 * 2e-4))
eff_an2 = (1.0 - np.exp(-N_Li6 * sig_an2 * d)) * 100.
ax.semilogx(E_an2, eff_an2, color=C_ANAL, lw=1.8, ls="--",
            label=r"Аналитич.: $\varepsilon = 1 - e^{-N\sigma d}$")

ax.set_xscale("log")
ax.set_xlim(1e-10, 12)
ax.set_ylim(-5, 105)
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel("Эффективность регистрации [%]")
ax.set_title("Зависимость эффективности регистрации от энергии нейтрона\n"
             "Детектор Li$_6$ZnS:Ag, $d$ = 5 мм")
ax.legend()
fig.tight_layout()
savefig(fig, "11_efficiency_vs_energy.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 12 — Модель спектра: три компоненты чётко видны
# ═══════════════════════════════════════════════════════════════════
print("График 12...")
E_pl = np.logspace(-10, 1.1, 3000)
kT2  = 0.0517e-3

phi_th_r = np.where(E_pl < 5e-7,  E_pl * np.exp(-E_pl/kT2), 0.)
phi_ep_r = np.where((E_pl >= 5e-7) & (E_pl < 0.1),
                    1.0/np.maximum(E_pl, 1e-30), 0.)
phi_f_r  = np.where(E_pl >= 0.1,
                    np.exp(-E_pl/0.988)*np.sinh(np.sqrt(2.249*E_pl)), 0.)

def norm_leth(phi, E):
    logE = np.log(np.maximum(E, 1e-100))
    total = np.trapezoid(phi, logE) if hasattr(np,'trapezoid') else np.trapz(phi, logE)
    return phi / total if abs(total) > 0 else phi

phi_th_n = 0.50 * norm_leth(phi_th_r, E_pl)
phi_ep_n = 0.25 * norm_leth(phi_ep_r, E_pl)
phi_f_n  = 0.25 * norm_leth(phi_f_r,  E_pl)
phi_tot2 = phi_th_n + phi_ep_n + phi_f_n

fig, ax = plt.subplots(figsize=(9, 5.5))

# Рисуем компоненты с заливкой ДО суммарного
ax.fill_between(E_pl, phi_th_n, alpha=0.20, color="royalblue")
ax.fill_between(E_pl, phi_ep_n, alpha=0.20, color="darkorange")
ax.fill_between(E_pl, phi_f_n,  alpha=0.20, color="firebrick")

ax.loglog(E_pl, phi_th_n, color="royalblue",  lw=2.0, ls="-",
          label=r"Максвелл–Больцман ($T$ = 600 К)")
ax.loglog(E_pl, phi_ep_n, color="darkorange", lw=2.0, ls="-",
          label=r"Замедление Ферми ($1/E$)")
ax.loglog(E_pl, phi_f_n,  color="firebrick",  lw=2.0, ls="-",
          label=r"Деление $^{235}$U (Ватт)")
ax.loglog(E_pl, phi_tot2, color="black",       lw=2.5, ls="-", alpha=0.85,
          label="Суммарный спектр")

# Границы областей
ax.axvline(5e-7, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.axvline(0.1,  color="gray", ls=":", lw=1.0, alpha=0.7)

ymax12 = phi_tot2[phi_tot2 > 0].max()
ax.text(1e-9,  ymax12*0.35, "Тепловая",    fontsize=10, color="royalblue",  ha="center")
ax.text(2e-4,  ymax12*0.08, "Эпитепловая", fontsize=10, color="darkorange", ha="center")
ax.text(1.5,   ymax12*0.30, "Быстрая",     fontsize=10, color="firebrick",  ha="center")

ax.set_xlim(1e-10, 12)
ax.set_ylim(ymax12*1e-6, ymax12*3)
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$E\,\phi(E)$ / нормировка [отн. ед.]")
ax.set_title("Аналитическая модель спектра реакторных нейтронов\n"
             "По: Lamarsh & Baratta (2001), ENDF/B-VIII.0 "
             r"[Ватт $b$ = 2.249 МэВ$^{-1}$]")
ax.legend(loc="upper left")
fig.tight_layout()
savefig(fig, "12_reactor_spectrum_model.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 13 — Жёсткость тритона/альфы (без изменений)
# ═══════════════════════════════════════════════════════════════════
print("График 13...")
fig, axes13 = plt.subplots(1, 2, figsize=(12, 5.5))
for ax13, pname, col in zip(axes13,
                             ["triton","alpha"],
                             [C_TRIT,   C_ALPHA]):
    sub = sec_df[(sec_df["particle"]==pname) &
                 (sec_df["rigidity_Tm"]>0)]["rigidity_Tm"].values
    if len(sub) < 5:
        ax13.text(0.5,0.5,"Нет данных",ha="center",va="center",
                  transform=ax13.transAxes); continue
    p99 = np.percentile(sub, 99)
    sub_c = sub[sub <= p99]
    bins13 = np.linspace(sub_c.min()*0.95, p99*1.02, 100)
    c13, e13 = np.histogram(sub_c, bins=bins13)
    ctr13 = 0.5*(e13[:-1]+e13[1:])
    ax13.fill_between(ctr13, c13, alpha=0.25, color=col)
    ax13.plot(ctr13, c13, color=col, lw=1.8)
    # Ожидаемое значение
    mass_p = {"triton":2808.921,"alpha":3727.379}[pname]
    chrg_p = {"triton":1,       "alpha":2}[pname]
    E0_p   = {"triton":2.73,    "alpha":2.05}[pname]
    Et = E0_p + mass_p
    pm = np.sqrt(max(Et**2 - mass_p**2, 0))
    rho0 = pm / (chrg_p * 299.792)
    ax13.axvline(rho0, color="k", ls="--", lw=1.2,
                 label=f"Ожидаемое $B\\rho_0$ = {rho0:.4f} Тл·м")
    lab13 = {"triton":"Тритон ($^3$H)","alpha":"Альфа ($^4$He)"}[pname]
    ax13.set_xlabel(r"Магнитная жёсткость $B\rho$ [Тл·м]")
    ax13.set_ylabel("Число частиц")
    ax13.set_title(lab13)
    ax13.legend(fontsize=10)

fig.suptitle(r"Магнитная жёсткость продуктов реакции $^6$Li$(n,t)^4$He",
             fontsize=13)
fig.tight_layout()
savefig(fig, "13_triton_alpha_rigidity.pdf")


# ═══════════════════════════════════════════════════════════════════
# ГРАФИК 14 — Угловое распределение (гистограммы cosθ и φ раздельно)
# ═══════════════════════════════════════════════════════════════════
print("График 14...")
fig, axes14 = plt.subplots(1, 2, figsize=(13, 5.5))

for pname, col, lab in [("triton",C_TRIT,"Тритон ($^3$H)"),
                          ("alpha", C_ALPHA,"Альфа ($^4$He)")]:
    sub14 = sec_df[sec_df["particle"]==pname][["px","py","pz"]].values
    if len(sub14) < 5:
        continue
    norms14 = np.linalg.norm(sub14, axis=1, keepdims=True)
    norms14 = np.where(norms14>0, norms14, 1)
    sub14n = sub14 / norms14
    costh = sub14n[:,2]
    phi14 = np.arctan2(sub14n[:,1], sub14n[:,0])

    # Cos θ — ось, физически значимая
    bins_cos = np.linspace(-1, 1, 40)
    c_cos, e_cos = np.histogram(costh, bins=bins_cos)
    ctr_cos = 0.5*(e_cos[:-1]+e_cos[1:])
    # Нормировка: изотропное = const * (1/2) на cos
    isotropic = np.ones_like(ctr_cos) * c_cos.sum() / len(bins_cos-1)
    axes14[0].plot(ctr_cos, c_cos, color=col, lw=1.8, label=lab)
    axes14[0].fill_between(ctr_cos, c_cos, alpha=0.15, color=col)

    # φ-распределение (азимутальный угол)
    bins_phi = np.linspace(-np.pi, np.pi, 40)
    c_phi, e_phi = np.histogram(phi14, bins=bins_phi)
    ctr_phi = 0.5*(e_phi[:-1]+e_phi[1:])
    axes14[1].plot(ctr_phi * 180/np.pi, c_phi, color=col, lw=1.8, label=lab)
    axes14[1].fill_between(ctr_phi*180/np.pi, c_phi, alpha=0.15, color=col)

# Изотропная линия
iso_level_cos = (c_cos.sum() if len(t_e)>0 else 100) / 38
axes14[0].axhline(iso_level_cos, color="gray", ls="--", lw=1.0,
                  alpha=0.7, label="Изотропное (ожидаемое)")
axes14[0].set_xlabel(r"$\cos\theta$ (угол к оси пучка)")
axes14[0].set_ylabel("Число частиц")
axes14[0].set_title("Полярное распределение\n(ось пучка: $z$)")
axes14[0].legend()

axes14[1].axhline(iso_level_cos, color="gray", ls="--", lw=1.0, alpha=0.7,
                  label="Изотропное")
axes14[1].set_xlabel("Азимутальный угол $\\varphi$ [°]")
axes14[1].set_ylabel("Число частиц")
axes14[1].set_title("Азимутальное распределение\n(изотропия относительно оси пучка)")
axes14[1].set_xlim(-180, 180)
axes14[1].xaxis.set_major_locator(ticker.MultipleLocator(60))
axes14[1].legend()

fig.suptitle("Угловое распределение вторичных частиц\n"
             r"$^6$Li$(n,t)^4$He — тритон и альфа-частица", fontsize=13)
fig.tight_layout()
savefig(fig, "14_angular_distribution.pdf")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "═"*50)
print(f" Все 14 графиков сохранены → {OUTDIR.resolve()}")
print("═"*50)
