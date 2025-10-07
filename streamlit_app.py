import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =============================
# CONFIG GÉNÉRALE
# =============================
st.set_page_config(
    page_title="Billets – Détection Vrai/Faux",
    page_icon="💶",
    layout="wide",
)

# --- Charte graphique
PRIMARY = "#22c55e"  # vert
DARK_BG = "#0b1021"   # fond sombre
LIGHT_BG = "#ffffff"  # fond clair
ACCENT = "#ef4444"    # rouge
TEXT_MUTED = "#6b7280"

# --- State par défaut
if "auth" not in st.session_state:
    st.session_state.auth = {
        "logged_in": False,
        "user": None,
    }
if "theme" not in st.session_state:
    st.session_state.theme = {
        "mode": "clair",        # "clair" | "sombre"
        "primary": PRIMARY,      # couleur principale (logo/cta)
    }
if "history" not in st.session_state:
    st.session_state.history = []  # liste d'analyses simulées (date, nb, vrais, faux)

# --- Thème CSS dynamique
mode = st.session_state.theme["mode"]
primary = st.session_state.theme["primary"]

BG = DARK_BG if mode == "sombre" else LIGHT_BG
FG = "#E5E7EB" if mode == "sombre" else "#111827"
CARD = "#131A35" if mode == "sombre" else "#F9FAFB"
BORDER = "rgba(255,255,255,0.08)" if mode == "sombre" else "#E5E7EB"

st.markdown(f"""
<style>
body {{ background: {BG}; color: {FG}; }}

.card {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px; 
  box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}}
.badge {{
  display:inline-block; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight:700;
}}
.pill {{ display:inline-block; padding:8px 14px; border-radius:999px; margin-right:8px; font-weight:700;}}
.hero {{
  background: linear-gradient(135deg, {primary} 0%, #16a34a 35%, #0ea5e9 100%);
  padding: 32px; border-radius: 18px; color: white; box-shadow: 0 12px 30px rgba(0,0,0,.25); margin-bottom: 16px;
}}
.footer {{
  margin-top: 24px; padding: 8px 12px; border-top: 1px solid {BORDER}; opacity:.8; text-align:center;
}}
.cta {{
  background:{primary}; color:white; border:none; padding:10px 18px; border-radius:999px; font-weight:800;
}}
.kpi h3 {{ margin:0; font-size: 0.95rem; color:{TEXT_MUTED}; }}
.kpi .val {{ font-size: 1.8rem; font-weight: 800; }}
</style>
""", unsafe_allow_html=True)

# =============================
# OUTILS
# =============================
REQUIRED_COLS = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]

def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLS if c not in df.columns]

def call_api_predict(api: str, payload: Dict[str, Any]) -> Any:
    headers = {"Content-Type": "application/json"}
    r = requests.post(api, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def simulate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    proba = rng.uniform(0.35, 0.98, size=len(df))
    pred = np.where(proba >= 0.5, "Vrai", "Faux")
    out = df.copy()
    out["prediction"] = pred
    out["proba_vrai"] = np.round(proba, 3)
    return out

def stats_from_pred(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    vrais = int((df["prediction"].str.lower() == "vrai").sum())
    faux = total - vrais
    pct_vrai = (vrais / total * 100) if total else 0
    pct_faux = 100 - pct_vrai
    avg = float(df["proba_vrai"].mean()) if "proba_vrai" in df.columns else None
    return {"total": total, "vrais": vrais, "faux": faux, "pct_vrai": pct_vrai, "pct_faux": pct_faux, "avg": avg}

# =============================
# SIDEBAR: NAV + AUTH
# =============================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Euro_symbol_black.svg", width=56)
st.sidebar.markdown(f"**Billets ML — <span style='color:{primary}'>Green/Black</span>**", unsafe_allow_html=True)

# Connexion / Profil
if not st.session_state.auth["logged_in"]:
    st.sidebar.subheader("🔐 Connexion")
    email = st.sidebar.text_input("Email")
    pwd = st.sidebar.text_input("Mot de passe", type="password")
    if st.sidebar.button("Se connecter", use_container_width=True):
        # ⚠️ Auth fictive pour portfolio — à remplacer par un vrai backend si besoin
        if email and pwd:
            st.session_state.auth.update({"logged_in": True, "user": email})
            st.sidebar.success("Connecté ✨")
        else:
            st.sidebar.error("Renseigne email et mot de passe")
else:
    st.sidebar.write(f"Connecté en tant que **{st.session_state.auth['user']}**")
    if st.sidebar.button("Se déconnecter", use_container_width=True):
        st.session_state.auth = {"logged_in": False, "user": None}
        st.rerun()

st.sidebar.markdown("---")
# Thème
st.sidebar.subheader("🎨 Apparence")
mode_select = st.sidebar.radio("Mode", ["clair", "sombre"], index=0 if mode=="clair" else 1, horizontal=True)
if mode_select != mode:
    st.session_state.theme["mode"] = mode_select
    st.rerun()

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🔮 Analyse", "📊 Dashboard", "🕓 Historique", "👤 Profil"], index=0)

# API
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ API")
api_url = st.sidebar.text_input("Endpoint de prédiction (POST)", placeholder="https://.../predict")
demo_mode = st.sidebar.toggle("Mode démo (sans API)", value=True)

# =============================
# PAGES
# =============================

# --- Accueil
if page == "🏠 Accueil":
    st.markdown("""
    <div class='hero'>
      <h1 style='margin:0;'>💶 Détection des billets – Vrai ou Faux</h1>
      <p style='margin:6px 0 0;opacity:.95;'>Interface moderne et professionnelle pour analyser vos billets en un clic.</p>
      <div style='margin-top:12px;'>
        <span class='pill' style='background:rgba(255,255,255,.18);'>Fiable</span>
        <span class='pill' style='background:rgba(255,255,255,.18);'>Précis</span>
        <span class='pill' style='background:rgba(255,255,255,.18);'>Sécurisé</span>
        <span class='pill' style='background:rgba(255,255,255,.18);'>Rapide</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cta1, cta2 = st.columns([1,1])
    with cta1:
        if st.button("🚀 Commencer l'analyse", type="primary"):
            st.session_state["__go_analyse__"] = True
            st.experimental_rerun()
    with cta2:
        st.link_button("Voir les dernières analyses", "#DernieresAnalyses")

    st.markdown("### Pourquoi cette application ?")
    st.write("Prépare un jeu de données (CSV) et obtiens **instantanément** la répartition Vrai/Faux, les probabilités moyennes et les résultats par billet — conforme au cahier des charges (CSV en entrée, prédictions + stats + graphes).")

    # Témoins visuels — pourcentage de répartition simulé (si historique existe)
    if st.session_state.history:
        st.markdown("#### Dernières tendances")
        df_hist = pd.DataFrame(st.session_state.history)
        line = px.line(df_hist, x="date", y="total", markers=True, title="Tendance du volume d'analyses")
        st.plotly_chart(line, use_container_width=True)

# --- Redirection vers Analyse depuis CTA
if st.session_state.get("__go_analyse__"):
    page = "🔮 Analyse"
    st.session_state["__go_analyse__"] = False

# --- Analyse
if page == "🔮 Analyse":
    st.header("Analyse des billets")
    st.caption("Uploader un CSV ou saisir une ligne pour tester. Répartition, pourcentages et export fournis.")

    tab_csv, tab_one = st.tabs(["📥 CSV", "✏️ Saisie 1 billet"])

    with tab_csv:
        up = st.file_uploader("Fichier CSV (colonnes: diagonal,height_left,height_right,margin_low,margin_up,length)", type=["csv"]) 
        colA, colB = st.columns([1,1])
        with colA:
            preview = st.toggle("Aperçu des 30 premières lignes", value=True)
        with colB:
            autotype = st.toggle("Conversion numérique automatique", value=True)

        df = None
        if up is not None:
            try:
                df = pd.read_csv(up)
                if autotype:
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception as e:
                st.error(f"Erreur de lecture: {e}")

        if df is not None:
            miss = validate_columns(df)
            if miss:
                st.error(f"Colonnes manquantes: {miss}. Requis: {REQUIRED_COLS}")
            else:
                st.success("✅ Colonnes valides")
                if preview:
                    st.dataframe(df.head(30), use_container_width=True)

                if st.button("Lancer l'analyse", type="primary"):
                    with st.spinner("Prédiction en cours..."):
                        try:
                            if demo_mode or not api_url:
                                pred_df = simulate_predictions(df[REQUIRED_COLS])
                            else:
                                payload = {"instances": df[REQUIRED_COLS].to_dict(orient="records")}
                                raw = call_api_predict(api_url, payload)
                                # normalisation minimaliste
                                if isinstance(raw, list):
                                    pred_df = df.copy()
                                    if all(isinstance(x, dict) for x in raw):
                                        pred_df["prediction"] = [str(x.get("prediction","")) for x in raw]
                                        if any(x.get("proba") for x in raw):
                                            pred_df["proba_vrai"] = [x.get("proba") for x in raw]
                                    else:
                                        pred_df["prediction"] = ["Vrai" if str(x) in ("1","True","Vrai") else "Faux" for x in raw]
                                elif isinstance(raw, dict) and "predictions" in raw:
                                    vals = raw["predictions"]
                                    pred_df = df.copy()
                                    if all(isinstance(x, dict) for x in vals):
                                        pred_df["prediction"] = [str(x.get("prediction","")) for x in vals]
                                        if any(x.get("proba") for x in vals):
                                            pred_df["proba_vrai"] = [x.get("proba") for x in vals]
                                    else:
                                        pred_df["prediction"] = ["Vrai" if str(x) in ("1","True","Vrai") else "Faux" for x in vals]
                                else:
                                    raise ValueError("Format de réponse API non supporté")
                        except Exception as e:
                            st.error(f"Erreur API: {e}")
                            pred_df = None

                    if pred_df is not None:
                        # KPIs + pourcentages
                        stats = stats_from_pred(pred_df)
                        k1,k2,k3,k4 = st.columns(4)
                        with k1:
                            st.markdown(f"<div class='card kpi'><h3>Total analysés</h3><div class='val'>{stats['total']}</div></div>", unsafe_allow_html=True)
                        with k2:
                            st.markdown(f"<div class='card kpi'><h3>Billets validés</h3><div class='val' style='color:{PRIMARY}'>{stats['vrais']}</div></div>", unsafe_allow_html=True)
                        with k3:
                            st.markdown(f"<div class='card kpi'><h3>Faux billets</h3><div class='val' style='color:{ACCENT}'>{stats['faux']}</div></div>", unsafe_allow_html=True)
                        with k4:
                            avg = f"{stats['avg']:.2f}" if stats['avg'] else "—"
                            st.markdown(f"<div class='card kpi'><h3>Proba moyenne (Vrai)</h3><div class='val'>{avg}</div></div>", unsafe_allow_html=True)

                        # Répartition en %
                        pie_df = pd.DataFrame({
                            "Classe":["Vrai","Faux"],
                            "Pourcentage":[stats["pct_vrai"], stats["pct_faux"]]
                        })
                        pie = px.pie(pie_df, names="Classe", values="Pourcentage", hole=0.35, title="Répartition des résultats (%)")
                        st.plotly_chart(pie, use_container_width=True)

                        # Tableau résultat + export
                        st.subheader("Résultats détaillés")
                        st.dataframe(pred_df, use_container_width=True, height=320)
                        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
                        st.download_button("💾 Télécharger (CSV)", csv_bytes, file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

                        # Enregistrer dans l'historique (portfolio)
                        st.session_state.history.append({
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "total": stats["total"],
                            "vrai": stats["vrais"],
                            "faux": stats["faux"],
                        })

    with tab_one:
        st.write("Saisir une seule ligne :")
        c1,c2,c3 = st.columns(3)
        with c1:
            length = st.number_input("length", value=140.0)
            margin_low = st.number_input("margin_low", value=6.0)
        with c2:
            height_left = st.number_input("height_left", value=74.0)
            margin_up = st.number_input("margin_up", value=8.0)
        with c3:
            height_right = st.number_input("height_right", value=73.8)
            diagonal = st.number_input("diagonal", value=160.0)
        if st.button("Prédire ce billet"):
            row = pd.DataFrame([{k:v for k,v in {
                "diagonal": diagonal,
                "height_left": height_left,
                "height_right": height_right,
                "margin_low": margin_low,
                "margin_up": margin_up,
                "length": length,
            }.items()}])
            if demo_mode or not api_url:
                res = simulate_predictions(row)
            else:
                payload = {"instances": row.to_dict(orient="records")}
                raw = call_api_predict(api_url, payload)
                if isinstance(raw, list) and len(raw):
                    if isinstance(raw[0], dict):
                        res = row.copy()
                        res["prediction"] = [raw[0].get("prediction","—")]
                        if raw[0].get("proba"):
                            res["proba_vrai"] = [raw[0]["proba"]]
                    else:
                        res = row.copy(); res["prediction"] = ["Vrai" if str(raw[0]) in ("1","True","Vrai") else "Faux"]
                else:
                    res = row.copy(); res["prediction"] = ["—"]
            st.success("Résultat")
            st.dataframe(res, use_container_width=True)

# --- Dashboard
if page == "📊 Dashboard":
    st.header("Dashboard global")
    st.caption("Vue d'ensemble : totaux, tendances, performances système.")

    # KPIs synthétiques
    total = sum([h["total"] for h in st.session_state.history]) if st.session_state.history else 0
    vrais = sum([h["vrai"] for h in st.session_state.history]) if st.session_state.history else 0
    faux = sum([h["faux"] for h in st.session_state.history]) if st.session_state.history else 0

    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(f"<div class='card kpi'><h3>Total analysé</h3><div class='val'>{total}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='card kpi'><h3>Billets validés</h3><div class='val' style='color:{PRIMARY}'>{vrais}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='card kpi'><h3>Faux billets</h3><div class='val' style='color:{ACCENT}'>{faux}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # Tendances
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        df_hist["date"] = pd.to_datetime(df_hist["date"]) 
        c1,c2 = st.columns([2,1])
        with c1:
            fig = px.line(df_hist, x="date", y=["vrai","faux","total"], markers=True, title="Tendances des analyses")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            last = df_hist.tail(1)
            if not last.empty:
                last_vals = last.iloc[0]
                st.markdown("**Dernière session**")
                st.write(pd.DataFrame({
                    "métrique":["total","vrai","faux"],
                    "valeur":[int(last_vals.total), int(last_vals.vrai), int(last_vals.faux)]
                }))
    else:
        st.info("Aucune donnée d'historique pour l'instant. Lance une analyse dans l'onglet 'Analyse'.")

# --- Historique
if page == "🕓 Historique":
    st.header("Historique des analyses")
    st.caption("Filtrer et consulter les analyses passées.")

    if st.session_state.history:
        dfh = pd.DataFrame(st.session_state.history)
        dfh["date"] = pd.to_datetime(dfh["date"])
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Depuis", value=(datetime.now()-timedelta(days=7)).date())
        with col2:
            end = st.date_input("Jusqu'au", value=datetime.now().date())
        mask = (dfh["date"].dt.date >= start) & (dfh["date"].dt.date <= end)
        view = dfh.loc[mask].sort_values("date", ascending=False)
        st.dataframe(view, use_container_width=True, height=360)
    else:
        st.info("Aucun historique encore. Une fois une analyse effectuée, elle s'affichera ici.")

# --- Profil
if page == "👤 Profil":
    st.header("Mon profil & préférences")
    if st.session_state.auth["logged_in"]:
        st.write(f"Connecté en tant que **{st.session_state.auth['user']}**")
    else:
        st.info("Vous n'êtes pas connecté. Utilisez la barre latérale pour vous connecter.")

    with st.expander("🎨 Apparence", expanded=True):
        st.write("Choisir votre style :")
        m = st.radio("Mode", ["clair","sombre"], index=0 if mode=="clair" else 1, horizontal=True)
        if m != mode:
            st.session_state.theme["mode"] = m
            st.rerun()
        color = st.color_picker("Couleur principale (logo/CTA)", value=primary)
        if color != primary:
            st.session_state.theme["primary"] = color
            st.rerun()

    st.markdown("---")
    st.subheader("Actions")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Voir les dernières analyses"):
            st.session_state["__go_hist__"] = True
            st.experimental_rerun()
    with c2:
        if st.button("Faire une nouvelle analyse"):
            st.session_state["__go_analyse__"] = True
            st.experimental_rerun()

if st.session_state.get("__go_hist__"):
    page = "🕓 Historique"
    st.session_state["__go_hist__"] = False

# =============================
# FOOTER
# =============================
st.markdown(
    f"<div class='footer'>© {datetime.now().year} – Sona Koulibaly • Streamlit × FastAPI × ML • Green/Black UI</div>",
    unsafe_allow_html=True,
)
