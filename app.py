import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioFuzzy — Prediksi Risiko Kardiovaskular",
    page_icon="🫀",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0f1117; }

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.03em;
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #e8f4f8;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    color: #7a9ab0;
    font-weight: 300;
    margin-bottom: 2rem;
}

.result-card {
    background: #161b22;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    border-left: 4px solid;
    margin-bottom: 1rem;
}
.result-card.rendah  { border-color: #2dd4bf; }
.result-card.sedang  { border-color: #f59e0b; }
.result-card.tinggi  { border-color: #f87171; }

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7a9ab0;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #e8f4f8;
}
.result-unit {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #7a9ab0;
    margin-left: 0.4rem;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.badge-rendah { background: #0d3331; color: #2dd4bf; }
.badge-sedang { background: #3b2200; color: #f59e0b; }
.badge-tinggi { background: #3b0d0d; color: #f87171; }

.rule-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    padding: 4px 0;
    border-bottom: 1px solid #1e2a35;
}
.rule-active {
    color: #38bdf8 !important;
}

.stSlider > div { padding-top: 0.3rem; }

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #1e2a35;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Fuzzy Logic Engine (from scratch) ─────────────────────────────────────────

def mu_segitiga(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if (c - b) != 0 else 1.0
    return 0.0

def mu_trapesium(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 1.0
    elif b < x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c) if (d - c) != 0 else 1.0
    return 0.0

def proses_fuzzifikasi(age_years, bmi, ap_hi):
    u_muda   = mu_trapesium(age_years, 0, 0, 35, 45)
    u_sedang = mu_segitiga(age_years, 40, 52, 65)
    u_tua    = mu_trapesium(age_years, 60, 70, 100, 100)

    bmi_kurang = mu_trapesium(bmi, 0, 0, 17, 18.5)
    bmi_normal = mu_segitiga(bmi, 18.0, 22.0, 25.0)
    bmi_lebih  = mu_trapesium(bmi, 24.0, 28.0, 60, 60)

    bp_normal = mu_trapesium(ap_hi, 0, 0, 115, 125)
    bp_tinggi = mu_trapesium(ap_hi, 120, 140, 300, 300)

    return {
        'umur':     {'muda': u_muda,      'sedang': u_sedang,    'tua': u_tua},
        'bmi':      {'kurang': bmi_kurang, 'normal': bmi_normal,  'lebih': bmi_lebih},
        'sistolik': {'normal': bp_normal,  'tinggi': bp_tinggi},
    }

RULE_DEFINITIONS = [
    ("Tinggi", "tua",    "lebih",  "tinggi", "R1"),
    ("Tinggi", "tua",    "normal", "tinggi", "R2"),
    ("Tinggi", "sedang", "lebih",  "tinggi", "R3"),
    ("Sedang", "sedang", "normal", "tinggi", "R4"),
    ("Tinggi", "tua",    "lebih",  "normal", "R5"),
    ("Sedang", "tua",    "normal", "normal", "R6"),
    ("Rendah", "muda",   "normal", "normal", "R7"),
    ("Sedang", "muda",   "lebih",  "tinggi", "R8"),
    ("Rendah", "muda",   "kurang", "normal", "R9"),
    ("Sedang", "sedang", "kurang", "tinggi", "R10"),
    ("Sedang", "sedang", "lebih",  "normal", "R11"),
    ("Rendah", "muda",   "lebih",  "normal", "R12"),
    ("Rendah", "muda",   "normal", "tinggi", "R13"),
    ("Sedang", "sedang", "normal", "normal", "R14"),
    ("Tinggi", "tua",    "kurang", "tinggi", "R15"),
]

def evaluasi_inferensi(fz):
    u, b, s = fz['umur'], fz['bmi'], fz['sistolik']
    rules = []
    for output, umur_key, bmi_key, bp_key, label in RULE_DEFINITIONS:
        alpha = min(u[umur_key], b[bmi_key], s[bp_key])
        rules.append((output, alpha, label))
    return rules

def defuzz_mamdani(rules):
    agregat = {k: max((v for o, v, _ in rules if o == k), default=0) for k in ['Rendah','Sedang','Tinggi']}
    pembilang = penyebut = 0.0
    for z in range(0, 101, 2):
        mu_r = min(agregat['Rendah'], mu_trapesium(z, 0, 0, 25, 45))
        mu_s = min(agregat['Sedang'], mu_segitiga(z, 35, 50, 65))
        mu_t = min(agregat['Tinggi'], mu_trapesium(z, 55, 75, 100, 100))
        mu_z = max(mu_r, mu_s, mu_t)
        pembilang += z * mu_z
        penyebut  += mu_z
    return pembilang / penyebut if penyebut != 0 else 0.0

def defuzz_sugeno(rules):
    konstanta = {'Rendah': 15.0, 'Sedang': 50.0, 'Tinggi': 85.0}
    pembilang = penyebut = 0.0
    for kategori, alpha, _ in rules:
        pembilang += alpha * konstanta[kategori]
        penyebut  += alpha
    return pembilang / penyebut if penyebut != 0 else 0.0

def label_risiko(score):
    if score <= 35:
        return "Rendah", "rendah"
    elif score <= 65:
        return "Sedang", "sedang"
    else:
        return "Tinggi", "tinggi"


# ─── Membership Function Plots ──────────────────────────────────────────────────

def plot_mf(title, x_range, members, xlabel):
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0f1117')
    colors = ['#38bdf8', '#a78bfa', '#f472b6', '#34d399']
    for i, (label, xs, ys) in enumerate(members):
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.8, label=label)
        ax.fill_between(xs, ys, alpha=0.08, color=colors[i % len(colors)])
    ax.set_xlim(x_range)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(title, color='#94a3b8', fontsize=9, pad=6)
    ax.set_xlabel(xlabel, color='#94a3b8', fontsize=8)
    ax.tick_params(colors='#4a5568', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a35')
    ax.legend(fontsize=7, loc='upper right',
              facecolor='#161b22', edgecolor='#1e2a35', labelcolor='#94a3b8')
    plt.tight_layout()
    return fig

def build_mf_plots():
    # Umur
    x_age = np.linspace(0, 100, 400)
    muda_y   = [mu_trapesium(v, 0, 0, 35, 45)  for v in x_age]
    sedang_y = [mu_segitiga(v, 40, 52, 65)      for v in x_age]
    tua_y    = [mu_trapesium(v, 60, 70, 100, 100) for v in x_age]
    fig_age = plot_mf("Variabel Umur", (0, 100),
        [("Muda", x_age, muda_y), ("Sedang", x_age, sedang_y), ("Tua", x_age, tua_y)], "Tahun")

    # BMI
    x_bmi = np.linspace(10, 50, 400)
    kurang_y = [mu_trapesium(v, 0, 0, 17, 18.5) for v in x_bmi]
    normal_y = [mu_segitiga(v, 18.0, 22.0, 25.0) for v in x_bmi]
    lebih_y  = [mu_trapesium(v, 24.0, 28.0, 60, 60) for v in x_bmi]
    fig_bmi = plot_mf("Variabel BMI", (10, 50),
        [("Kurang", x_bmi, kurang_y), ("Normal", x_bmi, normal_y), ("Lebih", x_bmi, lebih_y)], "BMI (kg/m²)")

    # Sistolik
    x_bp = np.linspace(80, 200, 400)
    bp_normal_y = [mu_trapesium(v, 0, 0, 115, 125)   for v in x_bp]
    bp_tinggi_y = [mu_trapesium(v, 120, 140, 300, 300) for v in x_bp]
    fig_bp = plot_mf("Variabel Tekanan Darah Sistolik", (80, 200),
        [("Normal", x_bp, bp_normal_y), ("Tinggi", x_bp, bp_tinggi_y)], "mmHg")

    return fig_age, fig_bmi, fig_bp

def plot_output_mf(score_mamdani, score_sugeno):
    x = np.linspace(0, 100, 500)
    rendah_y = [mu_trapesium(v, 0, 0, 25, 45)    for v in x]
    sedang_y = [mu_segitiga(v, 35, 50, 65)        for v in x]
    tinggi_y = [mu_trapesium(v, 55, 75, 100, 100) for v in x]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0f1117')
    ax.plot(x, rendah_y, color='#2dd4bf', lw=1.8, label='Rendah')
    ax.fill_between(x, rendah_y, alpha=0.08, color='#2dd4bf')
    ax.plot(x, sedang_y, color='#f59e0b', lw=1.8, label='Sedang')
    ax.fill_between(x, sedang_y, alpha=0.08, color='#f59e0b')
    ax.plot(x, tinggi_y, color='#f87171', lw=1.8, label='Tinggi')
    ax.fill_between(x, tinggi_y, alpha=0.08, color='#f87171')

    ax.axvline(score_mamdani, color='#60a5fa', lw=2, linestyle='--', label=f'Mamdani ({score_mamdani:.1f})')
    ax.axvline(score_sugeno,  color='#c084fc', lw=2, linestyle=':',  label=f'Sugeno ({score_sugeno:.1f})')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Fungsi Keanggotaan Output + Hasil Defuzzifikasi", color='#94a3b8', fontsize=9)
    ax.set_xlabel("Skor Risiko", color='#94a3b8', fontsize=8)
    ax.tick_params(colors='#4a5568', labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2a35')
    ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#1e2a35', labelcolor='#94a3b8')
    plt.tight_layout()
    return fig


# ─── Layout ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🫀 CardioFuzzy</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Sistem Prediksi Risiko Penyakit Kardiovaskular · Fuzzy Mamdani & Sugeno · From Scratch</div>', unsafe_allow_html=True)

tab_pred, tab_mf, tab_rules, tab_batch = st.tabs(["🔬 Prediksi", "📐 Fungsi Keanggotaan", "📋 Rule Base", "📊 Evaluasi Batch"])

# ── TAB 1: Prediksi ─────────────────────────────────────────────────────────────
with tab_pred:
    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown("#### Input Pasien")
        age_input = st.slider("Umur (tahun)", 20, 80, 45, 1)
        height_input = st.slider("Tinggi Badan (cm)", 140, 200, 165, 1)
        weight_input = st.slider("Berat Badan (kg)", 40, 130, 70, 1)
        ap_hi_input  = st.slider("Tekanan Darah Sistolik (mmHg)", 80, 200, 120, 1)

        bmi_calc = weight_input / ((height_input / 100) ** 2)
        st.markdown(f"**BMI terhitung:** `{bmi_calc:.2f}` kg/m²")

        st.markdown("---")
        run_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True)

    with col_out:
        if run_btn or True:  # auto-run
            fz    = proses_fuzzifikasi(age_input, bmi_calc, ap_hi_input)
            rules = evaluasi_inferensi(fz)
            score_m = defuzz_mamdani(rules)
            score_s = defuzz_sugeno(rules)

            label_m, cls_m = label_risiko(score_m)
            label_s, cls_s = label_risiko(score_s)

            st.markdown("#### Hasil Prediksi")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="result-card {cls_m}">
                    <div class="result-label">Mamdani</div>
                    <div class="result-value">{score_m:.1f}<span class="result-unit">/ 100</span></div>
                    <br><span class="badge badge-{cls_m}">{label_m}</span>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="result-card {cls_s}">
                    <div class="result-label">Sugeno</div>
                    <div class="result-value">{score_s:.1f}<span class="result-unit">/ 100</span></div>
                    <br><span class="badge badge-{cls_s}">{label_s}</span>
                </div>
                """, unsafe_allow_html=True)

            st.pyplot(plot_output_mf(score_m, score_s))

            with st.expander("🔍 Detail Nilai Fuzzifikasi"):
                fz_data = {
                    "Variabel": ["Umur-Muda","Umur-Sedang","Umur-Tua",
                                 "BMI-Kurang","BMI-Normal","BMI-Lebih",
                                 "Sistolik-Normal","Sistolik-Tinggi"],
                    "Nilai μ": [
                        fz['umur']['muda'], fz['umur']['sedang'], fz['umur']['tua'],
                        fz['bmi']['kurang'], fz['bmi']['normal'], fz['bmi']['lebih'],
                        fz['sistolik']['normal'], fz['sistolik']['tinggi'],
                    ]
                }
                df_fz = pd.DataFrame(fz_data)
                df_fz["Nilai μ"] = df_fz["Nilai μ"].map("{:.4f}".format)
                st.dataframe(df_fz, use_container_width=True, hide_index=True)

            with st.expander("⚡ Aktivasi Rule (top 5 aktif)"):
                rule_df = sorted(
                    [(label, output, alpha) for output, alpha, label in rules if alpha > 0],
                    key=lambda x: -x[2]
                )[:5]
                if rule_df:
                    for label, output, alpha in rule_df:
                        st.markdown(f"`{label}` → **{output}** · α = `{alpha:.4f}`")
                else:
                    st.info("Tidak ada rule yang aktif.")

# ── TAB 2: Fungsi Keanggotaan ────────────────────────────────────────────────────
with tab_mf:
    st.markdown("#### Fungsi Keanggotaan Variabel Input")
    fig_age, fig_bmi, fig_bp = build_mf_plots()
    c1, c2, c3 = st.columns(3)
    with c1: st.pyplot(fig_age)
    with c2: st.pyplot(fig_bmi)
    with c3: st.pyplot(fig_bp)

    st.markdown("---")
    st.markdown("#### Fungsi Keanggotaan Variabel Output")
    x = np.linspace(0, 100, 500)
    rendah_y = [mu_trapesium(v, 0, 0, 25, 45)    for v in x]
    sedang_y = [mu_segitiga(v, 35, 50, 65)        for v in x]
    tinggi_y = [mu_trapesium(v, 55, 75, 100, 100) for v in x]
    fig_out = plot_mf("Output: Skor Risiko Kardiovaskular", (0,100),
        [("Rendah", x, rendah_y), ("Sedang", x, sedang_y), ("Tinggi", x, tinggi_y)], "Skor Risiko (0–100)")
    st.pyplot(fig_out)

    st.markdown("""
    **Keterangan fungsi keanggotaan:**
    - **Trapesium** → digunakan untuk kategori di tepi (Muda, Tua, BMI Kurang, BMI Lebih, dsb.)
    - **Segitiga** → digunakan untuk kategori tengah (Umur Sedang, BMI Normal, Output Sedang)
    """)

# ── TAB 3: Rule Base ──────────────────────────────────────────────────────────────
with tab_rules:
    st.markdown("#### Rule Base (15 Aturan)")
    rule_table = []
    for output, umur_k, bmi_k, bp_k, label in RULE_DEFINITIONS:
        rule_table.append({
            "ID": label,
            "IF Umur": umur_k.capitalize(),
            "AND BMI": bmi_k.capitalize(),
            "AND Sistolik": bp_k.capitalize(),
            "THEN Risiko": output,
        })
    df_rules = pd.DataFrame(rule_table)
    st.dataframe(df_rules, use_container_width=True, hide_index=True)

    st.markdown("""
    **Interpretasi singkat:**
    - Kombinasi *tua + BMI lebih + sistolik tinggi* → **Risiko Tinggi** (paling berbahaya)
    - Kombinasi *muda + BMI normal + sistolik normal* → **Risiko Rendah** (baseline sehat)
    - Rule 13 menunjukkan bahwa meskipun muda, tekanan darah tinggi tetap memberikan sinyal risiko
    """)

# ── TAB 4: Evaluasi Batch ──────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Evaluasi Performa pada Dataset")
    st.info("Upload file `cardio_train.csv`")

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

    n_eval = st.slider("Jumlah baris yang dievaluasi", 100, 5000, 1000, 100)

    if st.button("▶️ Jalankan Evaluasi", use_container_width=True):
        with st.spinner("Sedang mengevaluasi..."):
            try:
                if uploaded:
                    df = pd.read_csv(uploaded, sep=';')
                else:
                    import os
                    csv_path = os.path.join(os.path.dirname(__file__), 'cardio_train.csv')
                    df = pd.read_csv(csv_path, sep=';')

                df['age_years'] = df['age'] / 365.25
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
                sample = df.head(n_eval).to_dict(orient='records')

                pred_m, pred_s, gt = [], [], []
                scores_m, scores_s = [], []

                for row in sample:
                    fz = proses_fuzzifikasi(row['age_years'], row['bmi'], row['ap_hi'])
                    r  = evaluasi_inferensi(fz)
                    sm = defuzz_mamdani(r)
                    ss = defuzz_sugeno(r)
                    scores_m.append(sm)
                    scores_s.append(ss)
                    pred_m.append(1 if sm > 50 else 0)
                    pred_s.append(1 if ss > 50 else 0)
                    gt.append(row['cardio'])

                acc_m = sum(m == g for m, g in zip(pred_m, gt)) / len(gt)
                acc_s = sum(s == g for s, g in zip(pred_s, gt)) / len(gt)

                c1, c2, c3 = st.columns(3)
                c1.metric("Akurasi Mamdani", f"{acc_m*100:.2f}%")
                c2.metric("Akurasi Sugeno",  f"{acc_s*100:.2f}%")
                c3.metric("Selisih",         f"{abs(acc_m - acc_s)*100:.2f}%")

                fig_scatter, ax = plt.subplots(figsize=(6, 3))
                fig_scatter.patch.set_facecolor('#161b22')
                ax.set_facecolor('#0f1117')
                idx = list(range(min(200, len(scores_m))))
                ax.scatter(idx, [scores_m[i] for i in idx], s=6, color='#60a5fa', alpha=0.6, label='Mamdani')
                ax.scatter(idx, [scores_s[i] for i in idx], s=6, color='#c084fc', alpha=0.6, label='Sugeno')
                ax.axhline(50, color='#f87171', lw=1, linestyle='--', label='Threshold=50')
                ax.set_title("Distribusi Skor Prediksi (200 sampel pertama)", color='#94a3b8', fontsize=9)
                ax.set_xlabel("Sampel ke-", color='#94a3b8', fontsize=8)
                ax.set_ylabel("Skor", color='#94a3b8', fontsize=8)
                ax.tick_params(colors='#4a5568', labelsize=7)
                for spine in ax.spines.values(): spine.set_edgecolor('#1e2a35')
                ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#1e2a35', labelcolor='#94a3b8')
                plt.tight_layout()
                st.pyplot(fig_scatter)

                st.markdown("""
                **Interpretasi Perbandingan:**
                - **Mamdani** menggunakan defuzzifikasi centroid (CoA) → output lebih halus dan gradual
                - **Sugeno** menggunakan weighted average dengan singleton → lebih cepat, output lebih stabil di tepi
                - Keduanya menggunakan rule base yang sama sehingga selisih akurasi umumnya kecil
                """)

            except Exception as e:
                st.error(f"Gagal memuat data: {e}")
