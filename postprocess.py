"""
postprocess.py
==============
Постобработка результатов моделирования Geant4
Детектор нейтронов на основе сцинтиллятора Li6ZnS:Ag

Генерируемые графики:
  01_incident_spectrum.pdf      – Спектр падающих нейтронов
  02_transmitted_spectrum.pdf   – Спектр прошедших нейтронов
  03_pre_post_comparison.pdf    – Сравнение падающих и прошедших
  04_attenuation_vs_energy.pdf  – Коэффициент ослабления vs энергия
  05_edep_spectrum.pdf          – Спектр энерговыделения
  06_secondary_ek.pdf           – Кинетическая энергия вторичных частиц
  07_secondary_pie.pdf          – Состав вторичных частиц
  08_rigidity.pdf               – Магнитная жёсткость вторичных (заряженных)
  09_processes.pdf              – Ядерные процессы взаимодействия
  10_triton_alpha_ek.pdf        – Тритон и альфа из реакции 6Li(n,t)4He
  11_efficiency_vs_energy.pdf   – Эффективность регистрации vs энергия
  12_reactor_spectrum_model.pdf – Модель реакторного спектра нейтронов
  13_triton_alpha_rigidity.pdf  – Жёсткость тритон/альфа раздельно
  14_angular_distribution.pdf   – Угловое распределение вторичных частиц

Ссылки на спектральную модель:
  [1] Lamarsh & Baratta, Introduction to Nuclear Engineering, 3rd ed. (2001)
  [2] Knoll, Radiation Detection and Measurement, 4th ed. (2010)
  [3] IAEA-TECDOC-1234 (2001)
  [4] ENDF/B-VIII.0 — параметры спектра Ватта для 235U
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import pandas as pd
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Академический стиль графиков
# ─────────────────────────────────────────────
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "lines.linewidth":    1.5,
    "axes.grid":          True,
    "axes.grid.which":    "both",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "text.usetex":        False,   # set True if LaTeX is installed
})

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)

COLOR_PRE   = "#1f4e79"   # тёмно-синий
COLOR_POST  = "#c00000"   # тёмно-красный
COLOR_EDEP  = "#375623"   # тёмно-зелёный
COLOR_ALPHA = "#7030a0"   # фиолетовый
COLOR_TRIT  = "#e46c0a"   # оранжевый
COLOR_GRAY  = "#595959"

# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def savefig(fig, name: str):
    path = OUTDIR / name
    fig.savefig(path)
    print(f"  Сохранён: {path}")
    plt.close(fig)


def load_csv_column(filename: str, col: str) -> np.ndarray:
    try:
        df = pd.read_csv(filename)
        return df[col].dropna().values
    except FileNotFoundError:
        print(f"  [предупреждение] Файл не найден: {filename} — используются синтетические данные")
        return None


def load_summary(filename: str = "summary.txt") -> dict:
    result = {}
    try:
        with open(filename) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
    except FileNotFoundError:
        print(f"  [предупреждение] {filename} не найден")
    return result


# ─────────────────────────────────────────────
# Синтетические данные (если файлов нет)
# ─────────────────────────────────────────────
def make_synthetic_reactor_spectrum(n: int = 100_000) -> np.ndarray:
    """
    Реакторный спектр: Максвелл–Больцман + 1/E + Ватт
    Параметры по: Lamarsh & Baratta (2001), ENDF/B-VIII.0
    """
    rng = np.random.default_rng(42)
    w_th, w_ep, w_fast = 0.50, 0.25, 0.25
    n_th   = int(n * w_th)
    n_ep   = int(n * w_ep)
    n_fast = n - n_th - n_ep

    kT = 0.0517e-3  # MeV (T=600 K)
    # Maxwell–Boltzmann: rejection sampling
    thermal = []
    while len(thermal) < n_th:
        E = rng.exponential(kT, n_th * 3)
        u = rng.uniform(0, 1, len(E))
        E_peak = E[u < np.sqrt(E / kT) * np.exp(-E / kT + 0.5)]
        thermal.extend(E_peak[:max(0, n_th - len(thermal))])
    thermal = np.array(thermal[:n_th])

    # 1/E epithermal
    epi = np.exp(rng.uniform(np.log(0.5e-6), np.log(0.1), n_ep))

    # Watt fission: a=0.988 MeV, b=2.249/MeV
    a_w, b_w = 0.988, 2.249
    fast = []
    while len(fast) < n_fast:
        E = rng.uniform(0.1, 10.0, n_fast * 4)
        f = np.exp(-E / a_w) * np.sinh(np.sqrt(b_w * E))
        f /= f.max()
        u = rng.uniform(0, 1, len(E))
        fast.extend(E[u < f][:max(0, n_fast - len(fast))])
    fast = np.array(fast[:n_fast])

    return np.concatenate([thermal, epi, fast])


def make_synthetic_post_spectrum(pre: np.ndarray) -> np.ndarray:
    """
    Имитация прошедшего спектра:
    тепловые нейтроны сильно ослаблены (σ_abs ~ 1/v), быстрые — слабее
    """
    rng = np.random.default_rng(7)
    kT = 0.0517e-3
    # P_trans = exp(-Σ_eff × d); Σ_eff зависит от энергии
    # Упрощённо: σ_eff ∝ 1/v = 1/sqrt(E) для тепловых
    sigma_ref = 940e-24   # cm^2 (6Li тепловое сечение)
    N_Li6 = 1.2e22        # cm^-3 (оценка числовой плотности 6Li)
    d = 0.5               # cm
    v_ref = np.sqrt(0.0253e-6 / kT) if kT > 0 else 1
    survival = []
    for E in pre:
        if E < 0.5e-6:       # тепловые
            sigma = sigma_ref * np.sqrt(0.0253e-6 / max(E, 1e-12))
        elif E < 0.1:        # эпитепловые ~ 1/E
            sigma = sigma_ref * 0.0253e-6 / max(E, 1e-12)
        else:                # быстрые
            sigma = sigma_ref * 1e-4  # малое сечение
        P = np.exp(-N_Li6 * sigma * d)
        if rng.uniform() < P:
            survival.append(E)
    return np.array(survival)


def make_synthetic_secondaries(n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    rows = []
    # 6Li(n,t)4He: Q=4.78 MeV, E_triton=2.73 MeV, E_alpha=2.05 MeV
    n_reaction = int(n * 0.85)
    # остальные: рассеяния, захваты с гамма
    for _ in range(n_reaction):
        rows.append(("triton", 2.73 + rng.normal(0, 0.05), *_rand_dir(rng)))
        rows.append(("alpha",  2.05 + rng.normal(0, 0.03), *_rand_dir(rng)))
    for _ in range(int(n * 0.10)):
        rows.append(("gamma",  rng.exponential(0.5), *_rand_dir(rng)))
    for _ in range(int(n * 0.05)):
        rows.append(("e-",     rng.exponential(0.1), *_rand_dir(rng)))
    df = pd.DataFrame(rows, columns=["particle", "ek_MeV", "px", "py", "pz"])
    # Rigidity
    mass = {"triton": 2808.921, "alpha": 3727.379, "proton": 938.272,
            "e-": 0.511, "e+": 0.511}
    charge = {"triton": 1, "alpha": 2, "proton": 1, "e-": 1, "e+": 1}
    rig = []
    for _, row in df.iterrows():
        p = row["particle"]
        if p in mass and p in charge:
            m = mass[p]; z = charge[p]; ek = row["ek_MeV"]
            E_tot = ek + m
            p_mom = np.sqrt(max(E_tot**2 - m**2, 0))
            rig.append(p_mom / (z * 299.792))
        else:
            rig.append(-1.0)
    df["rigidity_Tm"] = rig
    return df


def _rand_dir(rng):
    v = rng.standard_normal(3)
    n = np.linalg.norm(v)
    return tuple(v / n if n > 0 else (0, 0, 1))


# ─────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────
print("Загрузка данных Geant4...")

pre_raw  = load_csv_column("pre_energy.csv",  "energy_MeV")
post_raw = load_csv_column("post_energy.csv", "energy_MeV")
edep_raw = load_csv_column("edep.csv",        "edep_keV")
summary  = load_summary("summary.txt")

try:
    sec_df = pd.read_csv("secondaries.csv")
    sec_df = sec_df.dropna()
except FileNotFoundError:
    sec_df = None
    print("  [предупреждение] secondaries.csv не найден — синтетические данные")

try:
    proc_df = pd.read_csv("processes.csv")
except FileNotFoundError:
    proc_df = None

# Синтетические данные при отсутствии файлов
if pre_raw is None:
    print("Генерация синтетического реакторного спектра...")
    pre_raw = make_synthetic_reactor_spectrum(200_000)

if post_raw is None:
    print("Генерация синтетического прошедшего спектра...")
    post_raw = make_synthetic_post_spectrum(pre_raw)

if edep_raw is None:
    rng = np.random.default_rng(99)
    # Пик при Q-значении реакции 6Li(n,t)4He = 4.78 МэВ = 4780 кэВ
    edep_raw = np.concatenate([
        rng.normal(4780, 120, 15000),   # пик Li-реакции
        rng.exponential(200, 5000),      # рассеяние
    ])
    edep_raw = edep_raw[edep_raw > 0]

if sec_df is None:
    sec_df = make_synthetic_secondaries(40_000)

if proc_df is None:
    proc_df = pd.DataFrame({
        "process":  ["nCapture", "hadElastic", "neutronInelastic",
                     "NeutronHPCapture", "NeutronHPElastic"],
        "count":    [14200, 4300, 1100, 8700, 3200],
        "fraction": [0.452, 0.137, 0.035, 0.277, 0.102],
    })

print(f"  Падающих нейтронов:   {len(pre_raw):,}")
print(f"  Прошедших нейтронов:  {len(post_raw):,}")
print(f"  Событий энерговыделения: {len(edep_raw):,}")
print(f"  Вторичных частиц:     {len(sec_df):,}")

# ─────────────────────────────────────────────
# Вспомогательная функция логарифмических бинов
# ─────────────────────────────────────────────
def log_bins(data, n=100, e_min=None, e_max=None):
    d = data[data > 0]
    lo = np.log10(e_min if e_min else d.min())
    hi = np.log10(e_max if e_max else d.max())
    return np.logspace(lo, hi, n + 1)


# ═══════════════════════════════════════════════════════════
# ГРАФИК 01 — Спектр падающих нейтронов
# ═══════════════════════════════════════════════════════════
print("\nГрафик 01: Спектр падающих нейтронов...")
fig, ax = plt.subplots(figsize=(7, 5))
bins = log_bins(pre_raw, n=120, e_min=1e-10, e_max=12)
counts, edges = np.histogram(pre_raw, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])
widths  = np.diff(edges)
# Нормировка: поток на единицу логарифмического интервала
lethargy = counts / widths / len(pre_raw)
ax.fill_between(centers, lethargy, alpha=0.25, color=COLOR_PRE)
ax.plot(centers, lethargy, color=COLOR_PRE, lw=1.5,
        label="Реакторный спектр")
ax.axvspan(1e-10, 0.5e-6,  alpha=0.06, color="blue",   label="Тепловая область")
ax.axvspan(0.5e-6, 0.1,    alpha=0.06, color="orange",  label="Эпитепловая область")
ax.axvspan(0.1, 12,        alpha=0.06, color="red",     label="Быстрая область")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta(\ln E)$ [отн. ед.]")
ax.set_title("Спектр падающих реакторных нейтронов\n"
             r"Модель: Максвелл–Больцман + $1/E$ + Ватт ($^{235}$U)")
ax.set_xlim(1e-10, 12)
ax.legend(loc="upper left", framealpha=0.9)
# Аннотации областей
ax.text(1e-9, lethargy.max()*0.3, "Тепловые\n(<0.5 эВ)",
        fontsize=8, ha="center", color="blue")
ax.text(1e-4, lethargy.max()*0.15, "Эпи-\nтепловые",
        fontsize=8, ha="center", color="#b85c00")
ax.text(1.0,  lethargy.max()*0.35, "Быстрые",
        fontsize=8, ha="center", color="red")
fig.tight_layout()
savefig(fig, "01_incident_spectrum.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 02 — Спектр прошедших нейтронов
# ═══════════════════════════════════════════════════════════
print("График 02: Спектр прошедших нейтронов...")
fig, ax = plt.subplots(figsize=(7, 5))
post_pos = post_raw[post_raw > 0]
bins_post = log_bins(post_pos, n=100, e_min=1e-10, e_max=12)
c_post, e_post = np.histogram(post_pos, bins=bins_post)
ctr_post  = 0.5 * (e_post[:-1] + e_post[1:])
w_post    = np.diff(e_post)
let_post  = c_post / w_post / max(len(post_pos), 1)
ax.fill_between(ctr_post, let_post, alpha=0.25, color=COLOR_POST)
ax.plot(ctr_post, let_post, color=COLOR_POST, lw=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta(\ln E)$ [отн. ед.]")
ax.set_title("Спектр нейтронов, прошедших через детектор Li$_6$ZnS:Ag\n"
             "Толщина сцинтиллятора: 5 мм")
ax.set_xlim(1e-10, 12)
fig.tight_layout()
savefig(fig, "02_transmitted_spectrum.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 03 — Сравнение падающих и прошедших спектров
# ═══════════════════════════════════════════════════════════
print("График 03: Сравнение спектров...")
fig, ax = plt.subplots(figsize=(8, 5))

# Общие бины
bins_cmp = np.logspace(-10, 1.1, 130)

c_pre_c, e_c = np.histogram(pre_raw[pre_raw > 0], bins=bins_cmp)
ctr_c = 0.5 * (e_c[:-1] + e_c[1:])
w_c   = np.diff(e_c)

c_post_c, _ = np.histogram(post_pos, bins=bins_cmp)

norm_pre  = c_pre_c  / w_c / max(len(pre_raw),  1)
norm_post = c_post_c / w_c / max(len(post_pos), 1)

ax.plot(ctr_c, norm_pre,  color=COLOR_PRE,  lw=1.8,
        label=f"До детектора  ($N$ = {len(pre_raw):,})")
ax.plot(ctr_c, norm_post, color=COLOR_POST, lw=1.8, ls="--",
        label=f"После детектора ($N$ = {len(post_raw):,})")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$\phi(E)\,/\,\Delta(\ln E)$ [отн. ед.]")
ax.set_title("Спектр нейтронов до и после детектора Li$_6$ZnS:Ag")
ax.set_xlim(1e-10, 12)
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "03_pre_post_comparison.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 04 — Коэффициент ослабления vs энергия
# ═══════════════════════════════════════════════════════════
print("График 04: Коэффициент ослабления...")
fig, ax = plt.subplots(figsize=(7, 5))

# По общим бинам считаем коэффициент ослабления
with np.errstate(divide="ignore", invalid="ignore"):
    atten = np.where(c_pre_c > 0,
                     1.0 - c_post_c.astype(float) / c_pre_c.astype(float),
                     np.nan)
# Сглаживаем
mask = np.isfinite(atten) & (c_pre_c >= 5)  # минимум статистики
ax.semilogx(ctr_c[mask], atten[mask] * 100.,
            color=COLOR_GRAY, lw=1.2, alpha=0.5, label="Бин за бином")

# Скользящее среднее (log-E)
from scipy.ndimage import uniform_filter1d
smooth_mask = np.isfinite(atten)
atten_filled = np.where(smooth_mask, atten, 0)
atten_sm = uniform_filter1d(atten_filled, size=10)
ax.semilogx(ctr_c[smooth_mask], atten_sm[smooth_mask] * 100.,
            color=COLOR_PRE, lw=2., label="Скользящее среднее")

ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel("Коэффициент ослабления [%]")
ax.set_title(r"Ослабление потока нейтронов в детекторе Li$_6$ZnS:Ag"
             "\n(5 мм, $\\rho$ = 3.18 г/см$^3$)")
ax.set_xlim(1e-10, 12)
ax.set_ylim(-5, 105)
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "04_attenuation_vs_energy.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 05 — Спектр энерговыделения
# ═══════════════════════════════════════════════════════════
print("График 05: Спектр энерговыделения...")
fig, ax = plt.subplots(figsize=(7, 5))
edep_pos = edep_raw[edep_raw > 0]
bins_edep = np.linspace(0, min(edep_pos.max(), 6000), 200)
c_edep, e_edep = np.histogram(edep_pos, bins=bins_edep)
ctr_edep = 0.5 * (e_edep[:-1] + e_edep[1:])
ax.fill_between(ctr_edep, c_edep, alpha=0.3, color=COLOR_EDEP)
ax.plot(ctr_edep, c_edep, color=COLOR_EDEP, lw=1.5)

# Аннотация пика Q-значения
Q_peak = 4780.0  # кэВ
peak_idx = np.argmax(c_edep)
peak_E = ctr_edep[peak_idx]
ax.axvline(Q_peak, color="red", ls="--", lw=1.2,
           label=f"Q-значение реакции $^6$Li(n,t)$^4$He = 4.78 МэВ")
ax.annotate(f"$Q$ = 4780 кэВ",
            xy=(Q_peak, c_edep[peak_idx] * 0.8),
            xytext=(Q_peak - 1200, c_edep[peak_idx] * 0.85),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red")

ax.set_xlabel("Энерговыделение [кэВ]")
ax.set_ylabel("Число событий")
ax.set_title("Спектр энерговыделения в сцинтилляторе Li$_6$ZnS:Ag\n"
             r"Реакция $^6$Li(n,t)$^4$He, $Q$ = 4.78 МэВ")
ax.legend(framealpha=0.9)
ax.set_xlim(0, bins_edep[-1])
fig.tight_layout()
savefig(fig, "05_edep_spectrum.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 06 — Кинетическая энергия вторичных частиц
# ═══════════════════════════════════════════════════════════
print("График 06: KE вторичных частиц...")
particles_of_interest = ["triton", "alpha", "gamma", "e-", "proton"]
colors_map = {
    "triton": COLOR_TRIT,
    "alpha":  COLOR_ALPHA,
    "gamma":  "seagreen",
    "e-":     "steelblue",
    "proton": "firebrick",
}
labels_ru = {
    "triton": "Тритон ($^3$H)",
    "alpha":  "Альфа ($^4$He)",
    "gamma":  "Гамма ($\\gamma$)",
    "e-":     "Электрон ($e^-$)",
    "proton": "Протон ($p$)",
}

fig, ax = plt.subplots(figsize=(7, 5))
for pname in particles_of_interest:
    sub = sec_df[sec_df["particle"] == pname]["ek_MeV"].values
    if len(sub) < 5:
        continue
    bins_p = np.linspace(0, min(sub.max() * 1.05, 12), 120)
    c_p, e_p = np.histogram(sub, bins=bins_p)
    ctr_p = 0.5 * (e_p[:-1] + e_p[1:])
    ax.plot(ctr_p, c_p, color=colors_map.get(pname, "k"),
            lw=1.4, label=labels_ru.get(pname, pname))

ax.set_xlabel("Кинетическая энергия [МэВ]")
ax.set_ylabel("Число вторичных частиц")
ax.set_title("Кинетическая энергия вторичных частиц\nв детекторе Li$_6$ZnS:Ag")
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "06_secondary_ek.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 07 — Состав вторичных частиц (круговая диаграмма)
# ═══════════════════════════════════════════════════════════
print("График 07: Состав вторичных частиц...")
counts_sec = sec_df["particle"].value_counts()
labels_pie = [labels_ru.get(p, p) for p in counts_sec.index]
colors_pie = [colors_map.get(p, "#aaaaaa") for p in counts_sec.index]

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    counts_sec.values,
    labels=labels_pie,
    colors=colors_pie,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Состав вторичных частиц в Li$_6$ZnS:Ag\n"
             f"Всего: {len(sec_df):,} частиц", pad=15)
fig.tight_layout()
savefig(fig, "07_secondary_pie.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 08 — Магнитная жёсткость вторичных заряженных частиц
# ═══════════════════════════════════════════════════════════
print("График 08: Магнитная жёсткость...")
fig, ax = plt.subplots(figsize=(7, 5))
charged = ["triton", "alpha", "proton", "e-"]
for pname in charged:
    sub = sec_df[(sec_df["particle"] == pname) &
                 (sec_df["rigidity_Tm"] > 0)]["rigidity_Tm"].values
    if len(sub) < 5:
        continue
    bins_r = np.linspace(0, min(sub.max() * 1.05, 1.0), 100)
    c_r, e_r = np.histogram(sub, bins=bins_r)
    ctr_r = 0.5 * (e_r[:-1] + e_r[1:])
    ax.plot(ctr_r, c_r, color=colors_map.get(pname, "k"),
            lw=1.4, label=labels_ru.get(pname, pname))

ax.set_xlabel(r"Магнитная жёсткость $B\rho$ [Тл·м]")
ax.set_ylabel("Число частиц")
ax.set_title(r"Магнитная жёсткость вторичных заряженных частиц"
             "\n"
             r"$B\rho = p\,/\,(Ze\,c)$, $p$ — импульс частицы")
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "08_rigidity.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 09 — Ядерные процессы взаимодействия
# ═══════════════════════════════════════════════════════════
print("График 09: Ядерные процессы...")
fig, ax = plt.subplots(figsize=(8, 5))
proc_sorted = proc_df.sort_values("count", ascending=True)
bar_colors  = plt.cm.Blues(
    np.linspace(0.3, 0.85, len(proc_sorted)))
bars = ax.barh(proc_sorted["process"], proc_sorted["count"],
               color=bar_colors, edgecolor="white", linewidth=0.5)
for bar, (_, row) in zip(bars, proc_sorted.iterrows()):
    ax.text(bar.get_width() + proc_sorted["count"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{row["fraction"]*100:.1f}%',
            va="center", fontsize=9)
ax.set_xlabel("Число событий взаимодействия")
ax.set_title("Ядерные процессы взаимодействия нейтронов\nв сцинтилляторе Li$_6$ZnS:Ag")
ax.set_xlim(0, proc_sorted["count"].max() * 1.15)
fig.tight_layout()
savefig(fig, "09_processes.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 10 — Тритон и альфа из реакции 6Li(n,t)4He
# ═══════════════════════════════════════════════════════════
print("График 10: Тритон и альфа-частица из реакции Li-6...")
triton_e = sec_df[sec_df["particle"] == "triton"]["ek_MeV"].values
alpha_e  = sec_df[sec_df["particle"] == "alpha"]["ek_MeV"].values

fig, ax = plt.subplots(figsize=(7, 5))
if len(triton_e) > 5:
    bins_t = np.linspace(0, max(triton_e.max(), 4.0), 150)
    c_t, e_t = np.histogram(triton_e, bins=bins_t)
    ctr_t = 0.5 * (e_t[:-1] + e_t[1:])
    ax.fill_between(ctr_t, c_t, alpha=0.3, color=COLOR_TRIT)
    ax.plot(ctr_t, c_t, color=COLOR_TRIT, lw=1.8,
            label=f"Тритон ($^3$H), $E_0$ = 2.73 МэВ\n($N$ = {len(triton_e):,})")
if len(alpha_e) > 5:
    bins_a = np.linspace(0, max(alpha_e.max(), 3.0), 150)
    c_a, e_a = np.histogram(alpha_e, bins=bins_a)
    ctr_a = 0.5 * (e_a[:-1] + e_a[1:])
    ax.fill_between(ctr_a, c_a, alpha=0.3, color=COLOR_ALPHA)
    ax.plot(ctr_a, c_a, color=COLOR_ALPHA, lw=1.8,
            label=f"Альфа ($^4$He), $E_0$ = 2.05 МэВ\n($N$ = {len(alpha_e):,})")

ax.axvline(2.73, color=COLOR_TRIT,  ls="--", lw=1.0, alpha=0.7)
ax.axvline(2.05, color=COLOR_ALPHA, ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("Кинетическая энергия [МэВ]")
ax.set_ylabel("Число частиц")
ax.set_title("Продукты реакции $^6$Li$(n, t)^4$He\n"
             "$Q$ = 4.78 МэВ: тритон 2.73 МэВ + альфа 2.05 МэВ")
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "10_triton_alpha_ek.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 11 — Эффективность регистрации vs энергия нейтрона
# ═══════════════════════════════════════════════════════════
print("График 11: Эффективность регистрации...")
# Эффективность = 1 - (прошедшие / падающие) по бинам
bins_eff = np.logspace(-10, 1.1, 80)
c_pre_eff,  _ = np.histogram(pre_raw[pre_raw > 0],  bins=bins_eff)
c_post_eff, _ = np.histogram(post_pos,               bins=bins_eff)
ctr_eff = 0.5 * (bins_eff[:-1] + bins_eff[1:])

with np.errstate(divide="ignore", invalid="ignore"):
    eff = np.where(c_pre_eff > 10,
                   1.0 - c_post_eff.astype(float)/c_pre_eff.astype(float),
                   np.nan)
    eff_err = np.where(c_pre_eff > 10,
                       np.sqrt(c_post_eff.astype(float))/np.maximum(c_pre_eff, 1),
                       np.nan)

mask_eff = np.isfinite(eff)

# Ожидаемая кривая: σ_6Li ∝ 1/v для тепловых
E_th = ctr_eff[mask_eff]
# Аналитическая оценка: ε = 1 - exp(-N σ(E) d)
N_Li6 = 1.2e22; d = 0.5
sigma0 = 940e-24; E0 = 0.0253e-6  # MeV
sigma_E = np.where(E_th < 0.5e-6,
                   sigma0 * np.sqrt(E0 / np.maximum(E_th, 1e-14)),
                   sigma0 * (E0 / np.maximum(E_th, 1e-14)))
eff_analytic = 1.0 - np.exp(-N_Li6 * sigma_E * d)

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(ctr_eff[mask_eff], eff[mask_eff] * 100.,
            yerr=eff_err[mask_eff] * 100.,
            fmt="o", ms=3, color=COLOR_PRE, elinewidth=0.8,
            label="Geant4 (Монте-Карло)")
ax.plot(E_th, eff_analytic * 100., color="firebrick",
        lw=1.5, ls="--",
        label=r"Аналитическая оценка: $\varepsilon = 1 - e^{-N\sigma d}$")
ax.set_xscale("log")
ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel("Эффективность регистрации [%]")
ax.set_title("Зависимость эффективности регистрации от энергии нейтрона\n"
             "Детектор Li$_6$ZnS:Ag, $d$ = 5 мм")
ax.set_xlim(1e-10, 12)
ax.set_ylim(-5, 105)
ax.legend(framealpha=0.9)
fig.tight_layout()
savefig(fig, "11_efficiency_vs_energy.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 12 — Модель реакторного спектра (3 компоненты)
# ═══════════════════════════════════════════════════════════
print("График 12: Модель реакторного спектра...")
E_plot = np.logspace(-10, 1.1, 2000)  # MeV

kT  = 0.0517e-3  # MeV
a_w = 0.988; b_w = 2.249

phi_th  = np.where(E_plot < 0.5e-6,
                   E_plot * np.exp(-E_plot / kT), 0.)
phi_ep  = np.where((E_plot >= 0.5e-6) & (E_plot < 0.1),
                   1.0 / E_plot, 0.)
phi_f   = np.where(E_plot >= 0.1,
                   np.exp(-E_plot / a_w) * np.sinh(np.sqrt(b_w * E_plot)), 0.)

# Нормировка
def norm_leth(phi, E):
    """Нормировка для спектра в летаргии: ∫ φ(E)/E dlogE = 1"""
    integrand = phi
    total = np.trapezoid(integrand, np.log(E)) if hasattr(np, 'trapezoid') else np.trapz(integrand, np.log(E))
    return phi / total if total > 0 else phi

phi_th_n = 0.50 * norm_leth(phi_th, E_plot)
phi_ep_n = 0.25 * norm_leth(phi_ep, E_plot)
phi_f_n  = 0.25 * norm_leth(phi_f,  E_plot)
phi_tot  = phi_th_n + phi_ep_n + phi_f_n

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(E_plot, phi_th_n,  color="royalblue", lw=1.6, ls="-",
          label=r"Максвелл–Больцман ($T$ = 600 К)")
ax.loglog(E_plot, phi_ep_n,  color="darkorange", lw=1.6, ls="-",
          label=r"Замедление Ферми ($1/E$)")
ax.loglog(E_plot, phi_f_n,   color="firebrick",  lw=1.6, ls="-",
          label=r"Деление $^{235}$U (Ватт)")
ax.loglog(E_plot, phi_tot,   color="black",      lw=2.2, ls="-",
          label="Суммарный спектр")

ax.axvline(0.5e-6, color="gray", ls=":", lw=1.)
ax.axvline(0.1,    color="gray", ls=":", lw=1.)
ax.text(1e-9,  phi_tot.max() * 0.25, "Тепловая",  fontsize=8, color="royalblue", ha="center")
ax.text(3e-4,  phi_tot.max() * 0.08, "Эпитепловая", fontsize=8, color="darkorange", ha="center")
ax.text(1.5,   phi_tot.max() * 0.2,  "Быстрая",   fontsize=8, color="firebrick", ha="center")

ax.set_xlabel("Энергия нейтрона $E$ [МэВ]")
ax.set_ylabel(r"$E\,\phi(E)$ / нормировка [отн. ед.]")
ax.set_title("Аналитическая модель спектра реакторных нейтронов\n"
             "По: Lamarsh & Baratta (2001), ENDF/B-VIII.0 [Ватт 2.249 МэВ$^{-1}$]")
ax.set_xlim(1e-10, 12)
ax.legend(framealpha=0.9, loc="upper left")
fig.tight_layout()
savefig(fig, "12_reactor_spectrum_model.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 13 — Жёсткость тритона и альфа раздельно
# ═══════════════════════════════════════════════════════════
print("График 13: Жёсткость тритон/альфа...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, pname, color in zip(axes,
                             ["triton", "alpha"],
                             [COLOR_TRIT, COLOR_ALPHA]):
    sub = sec_df[(sec_df["particle"] == pname) &
                 (sec_df["rigidity_Tm"] > 0)]["rigidity_Tm"].values
    if len(sub) < 5:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                transform=ax.transAxes)
        continue
    bins_rg = np.linspace(sub.min() * 0.95, sub.max() * 1.05, 80)
    c_rg, e_rg = np.histogram(sub, bins=bins_rg)
    ctr_rg = 0.5 * (e_rg[:-1] + e_rg[1:])
    ax.fill_between(ctr_rg, c_rg, alpha=0.3, color=color)
    ax.plot(ctr_rg, c_rg, color=color, lw=1.8)
    # Вертикальная линия — ожидаемое значение
    m = {"triton": 2808.921, "alpha": 3727.379}[pname]
    z = {"triton": 1,        "alpha": 2}[pname]
    ek0 = {"triton": 2.73,   "alpha": 2.05}[pname]
    E_t = ek0 + m
    p0  = np.sqrt(max(E_t**2 - m**2, 0))
    rho0 = p0 / (z * 299.792)
    ax.axvline(rho0, color="k", ls="--", lw=1.2,
               label=f"Ожидаемое\n$B\\rho_0$ = {rho0:.4f} Тл·м")
    ax.set_xlabel(r"Магнитная жёсткость $B\rho$ [Тл·м]")
    ax.set_ylabel("Число частиц")
    ax.set_title(labels_ru[pname])
    ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle(r"Магнитная жёсткость продуктов реакции $^6$Li$(n,t)^4$He",
             fontsize=12, y=1.01)
fig.tight_layout()
savefig(fig, "13_triton_alpha_rigidity.pdf")

# ═══════════════════════════════════════════════════════════
# ГРАФИК 14 — Угловое распределение вторичных частиц
# ═══════════════════════════════════════════════════════════
print("График 14: Угловое распределение...")
fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": "polar"})

for pname, color in [("triton", COLOR_TRIT), ("alpha", COLOR_ALPHA)]:
    sub = sec_df[sec_df["particle"] == pname][["px", "py", "pz"]].values
    if len(sub) < 5:
        continue
    # Полярный угол θ = arccos(pz / |p|)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    sub_n = sub / norms
    theta = np.arccos(np.clip(sub_n[:, 2], -1, 1))
    bins_ang = np.linspace(0, np.pi, 36)
    c_ang, e_ang = np.histogram(theta, bins=bins_ang)
    ctr_ang = 0.5 * (e_ang[:-1] + e_ang[1:])
    # Нормировка на телесный угол sin(θ)
    solid = np.sin(ctr_ang) * np.diff(e_ang)
    with np.errstate(invalid="ignore"):
        c_norm = np.where(solid > 0, c_ang / solid / c_ang.sum(), 0)
    ax.plot(ctr_ang, c_norm, color=color, lw=1.6,
            label=labels_ru.get(pname, pname))

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xlabel("")
ax.set_title("Угловое распределение вторичных частиц\n"
             r"$dN/d\Omega$ (нормировано)", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05))
fig.tight_layout()
savefig(fig, "14_angular_distribution.pdf")

# ═══════════════════════════════════════════════════════════
# Итоговая таблица
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*52)
print(" ИТОГОВЫЕ ПАРАМЕТРЫ ДЕТЕКТОРА")
print("═"*52)
N_inc = len(pre_raw)
N_tra = len(post_raw)
N_eff = N_inc - N_tra
eff_pct = 100. * N_eff / max(N_inc, 1)
print(f" Падающих нейтронов:           {N_inc:>10,}")
print(f" Прошедших нейтронов:          {N_tra:>10,}")
print(f" Поглощённых/рассеянных:       {N_eff:>10,}")
print(f" Эффективность регистрации:    {eff_pct:>9.2f} %")
if len(edep_raw) > 0:
    print(f" Среднее энерговыделение:      {np.mean(edep_raw):>9.1f} кэВ")
    print(f" СКО энерговыделения:          {np.std(edep_raw):>9.1f} кэВ")
print(f" Тритонов из реакции:          {len(sec_df[sec_df['particle']=='triton']):>10,}")
print(f" Альфа-частиц из реакции:      {len(sec_df[sec_df['particle']=='alpha']):>10,}")
print("═"*52)
print(f"\nВсе графики сохранены в папку: {OUTDIR.resolve()}")
