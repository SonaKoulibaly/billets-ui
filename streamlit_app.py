import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Plotly avec fallback si non dispo
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Billets — Vrai/Faux", page_icon="💶", layout="wide")

# API en ligne (défaut) —>> TON RENDER
BASE_API = "https://billets-api-1.onrender.com"
ENDPOINT_ONE = f"{BASE_API}/predict_one"
ENDPOINT_CSV = f"{BASE_API}/predict_csv"

PRIMARY = "#22c55e"   # vert
ACCENT  = "#ef4444"   # rouge
DARK_BG = "#0b1021"
LIGHT_BG= "#ffffff"
TEXT_MUTED = "#6b7280"

REQUIRED_COLS = ["diagonal","height_left","height_right","margin_low","margin_up","length"]

# States
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None}
if "theme" not in st.session_state:
    st.session_state.theme = {"mode": "clair", "primary": PRIMARY}
if "history" not in st.session_state:
    st.session_state.history = []

mode    = st.session_state.theme["mode"]
primary = st.session_state.theme["primary"]

BG     = DARK_BG if mode == "sombre" else LIGHT_BG
FG     = "#E5E7EB" if mode == "sombre" else "#111827"
CARD   = "#131A35" if mode == "sombre" else "#F9FAFB"
BORDER = "rgba(255,255,255,0.08)" if mode == "sombre" else "#E5E7EB"

# =============================
# CSS (HTML/CSS custom "vitrine")
# =============================
st.markdown(f"""
<style>
body {{ background:{BG}; color:{FG}; }}
.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:16px; box-shadow:0 6px 18px rgba(0,0,0,.12); }}
.badge {{ display:inline-block; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:700; }}
.pill  {{ display:inline-block; padding:8px 14px; border-radius:999px; margin-right:8px; font-weight:700; }}
.hero  {{
  background: linear-gradient(135deg, {primary} 0%, #16a34a 35%, #0ea5e9 100%);
  padding: 32px; border-radius: 18px; color: white; box-shadow: 0 12px 30px rgba(0,0,0,.25); margin-bottom: 16px;
}}
.footer {{ margin-top:24px; padding:8px 12px; border-top:1px solid {BORDER}; opacity:.85; text-align:center; }}
.cta    {{ background:{primary}; color:white; border:none; padding:10px 18px; border-radius:999px; font-weight:800; }}
.kpi h3 {{ margin:0; font-size:.95rem; color:{TEXT_MUTED}; }}
.kpi .val {{ font-size:1.8rem; font-weight:800; }}
</style>
""", unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLS if c not in df.columns]

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
    vrais = int((df["prediction"].astype(str).str.lower() == "vrai").sum())
    faux  = total - vrais
    pct_vrai = (vrais/total*100) if total else 0
    pct_faux = 100 - pct_vrai
    avg = float(df["proba_vrai"].mean()) if "proba_vrai" in df.columns else None
    return {"total":total,"vrais":vrais,"faux":faux,"pct_vrai":pct_vrai,"pct_faux":pct_faux,"avg":avg}

# =============================
# SIDEBAR : NAV + THEME + (Login fictif)
# =============================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Euro_symbol_black.svg", width=56)
st.sidebar.markdown(f"**Billets ML — <span style='color:{primary}'>Green/Black</span>**", unsafe_allow_html=True)

# Connexion portfolio (fictif)
if not st.session_state.auth["logged_in"]:
    st.sidebar.subheader("🔐 Connexion")
    email = st.sidebar.text_input("Email")
    pwd   = st.sidebar.text_input("Mot de passe", type="password")
    if st.sidebar.button("Se connecter", use_container_width=True):
        if email and pwd:
            st.session_state.auth = {"logged_in": True, "user": email}
            st.sidebar.success("Connecté ✨")
        else:
            st.sidebar.error("Renseigne email et mot de passe")
else:
    st.sidebar.write(f"Connecté en tant que **{st.session_state.auth['user']}**")
    if st.sidebar.button("Se déconnecter", use_container_width=True):
        st.session_state.auth = {"logged_in": False, "user": None}
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Apparence")
m = st.sidebar.radio("Mode", ["clair","sombre"], index=0 if mode=="clair" else 1, horizontal=True)
if m != mode:
    st.session_state.theme["mode"] = m
    st.rerun()
c = st.sidebar.color_picker("Couleur principale (logo/CTA)", value=primary)
if c != primary:
    st.session_state.theme["primary"] = c
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Accueil","🔮 Analyse","📊 Dashboard","🕓 Historique","👤 Profil"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{BASE_API}`\n\nEndpoints: `/predict_one`, `/predict_csv`")

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
        <span class='pill' style='background:rgba(255,255,255,.18)'>Fiable</span>
        <span class='pill' style='background:rgba(255,255,255,.18)'>Précis</span>
        <span class='pill' style='background:rgba(255,255,255,.18)'>Sécurisé</span>
        <span class='pill' style='background:rgba(255,255,255,.18)'>Rapide</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Commencer l'analyse", type="primary"):
            st.session_state["__go_analyse__"] = True
            st.experimental_rerun()
    with c2:
        if st.button("Voir les dernières analyses"):
            st.session_state["__go_hist__"] = True
            st.experimental_rerun()

    st.markdown("### Pourquoi cette application ?")
    st.write("Téléverse un CSV avec les colonnes requises et obtiens la **répartition Vrai/Faux (en %)**, les **KPIs** et un **export CSV**. Parfait pour démontrer ton pipeline ML (API FastAPI + UI Streamlit).")

# redirections
if st.session_state.get("__go_analyse__"):
    page = "🔮 Analyse"; st.session_state["__go_analyse__"] = False
if st.session_state.get("__go_hist__"):
    page = "🕓 Historique"; st.session_state["__go_hist__"] = False

# --- Analyse
if page == "🔮 Analyse":
    st.header("Analyse des billets")
    st.caption("Uploader un CSV ou saisir une ligne. Répartition en pourcentage, KPIs, export.")

    tab_csv, tab_one = st.tabs(["📥 CSV", "✏️ Saisie 1 billet"])

    # CSV -> /predict_csv (multipart)
    with tab_csv:
        up = st.file_uploader("Fichier CSV (colonnes: diagonal,height_left,height_right,margin_low,margin_up,length)", type=["csv"])
        colA, colB = st.columns(2)
        with colA:
            preview  = st.toggle("Aperçu des 30 premières lignes", value=True)
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
                st.error(f"Erreur de lecture : {e}")

        if df is not None:
            miss = validate_columns(df)
            if miss:
                st.error(f"Colonnes manquantes : {miss}. Requis : {REQUIRED_COLS}")
            else:
                st.success("✅ Colonnes valides")
                if preview:
                    st.dataframe(df.head(30), use_container_width=True)

                if st.button("Lancer l'analyse", type="primary"):
                    with st.spinner("Prédiction en cours..."):
                        try:
                            # envoi multipart
                            csv_bytes = df[REQUIRED_COLS].to_csv(index=False).encode("utf-8")
                            files = {"file": ("data.csv", csv_bytes, "text/csv")}
                            resp = requests.post(ENDPOINT_CSV, files=files, timeout=120)
                            resp.raise_for_status()
                            data = resp.json()
                        except Exception as e:
                            st.error(f"Erreur API /predict_csv : {e}")
                            data = None

                    if data is not None:
                        # KPIs de lignes (si l’API renvoie ces champs)
                        rows_received = int(data.get("rows_received", 0))
                        used         = int(data.get("rows_used_for_prediction", 0))
                        dropped      = int(data.get("rows_dropped_after_cleaning", 0))
                        k1,k2,k3 = st.columns(3)
                        k1.metric("📄 Lignes reçues", rows_received)
                        k2.metric("✅ Lignes utilisées", used)
                        k3.metric("🧹 Lignes écartées", dropped)

                        # Essayer d’obtenir un échantillon “joli”
                        sample = data.get("sample_predictions_head_labels") or data.get("sample_predictions_head")
                        if sample:
                            pred_df = pd.DataFrame(sample)
                            # mapper 0/1 -> Vrai/Faux dans colonnes 'pred_*' si besoin
                            for col in pred_df.columns:
                                if col.lower().startswith("pred_"):
                                    try:
                                        pred_df[col] = pred_df[col].map({1:"Vrai",0:"Faux"}).fillna(pred_df[col])
                                    except Exception:
                                        pass
                        else:
                            # fallback si l’API ne renvoie pas de head détaillé
                            pred_df = simulate_predictions(df[REQUIRED_COLS])

                        # Stats & % (sur la colonne finale si présente, sinon déduire naïvement)
                        if "prediction" not in pred_df.columns:
                            # prends la 1re colonne pred_* dispo
                            pred_cols = [c for c in pred_df.columns if c.lower().startswith("pred_")]
                            if pred_cols:
                                pred_df = pred_df.rename(columns={pred_cols[0]: "prediction"})
                        stats = stats_from_pred(pred_df) if "prediction" in pred_df.columns else {"total": len(pred_df), "vrais": 0, "faux": 0, "pct_vrai": 0, "pct_faux": 0, "avg": None}

                        c1,c2,c3,c4 = st.columns(4)
                        c1.markdown(f"<div class='card kpi'><h3>Total analysés</h3><div class='val'>{stats['total']}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='card kpi'><h3>Billets validés</h3><div class='val' style='color:{PRIMARY}'>{stats['vrais']}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='card kpi'><h3>Faux billets</h3><div class='val' style='color:{ACCENT}'>{stats['faux']}</div></div>", unsafe_allow_html=True)
                        avg = f"{stats['avg']:.2f}" if stats.get("avg") else "—"
                        c4.markdown(f"<div class='card kpi'><h3>Proba moyenne (Vrai)</h3><div class='val'>{avg}</div></div>", unsafe_allow_html=True)

                        # % Répartition
                        pie_df = pd.DataFrame({"Classe":["Vrai","Faux"], "Pourcentage":[stats["pct_vrai"], stats["pct_faux"]]})
                        if PLOTLY_OK:
                            fig = px.pie(pie_df, names="Classe", values="Pourcentage", hole=0.35, title="Répartition des résultats (%)")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.bar_chart(pie_df.set_index("Classe"))

                        st.subheader("Résultats détaillés")
                        st.dataframe(pred_df, use_container_width=True, height=320)
                        st.download_button("💾 Télécharger (CSV)", pred_df.to_csv(index=False).encode("utf-8"),
                                           file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

                        # Enregistre dans l’historique (pour le Dashboard/Historique)
                        st.session_state.history.append({
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "total": stats["total"], "vrai": stats["vrais"], "faux": stats["faux"],
                        })

    # 1 billet -> /predict_one (JSON)
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
            payload = {
                "length": float(length), "height_left": float(height_left), "height_right": float(height_right),
                "margin_low": float(margin_low), "margin_up": float(margin_up), "diagonal": float(diagonal),
            }
            try:
                r = requests.post(ENDPOINT_ONE, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                res = pd.DataFrame([{
                    **data.get("input", payload),
                    "prediction": (data.get("prediction", {}) or {}).get("majority_vote") or data.get("prediction"),
                    "proba_vrai": (data.get("avg_positive_probability", {}) or {}).get("rf"),
                }])
                st.success("Résultat")
                st.dataframe(res, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur API /predict_one : {e}")

# --- Dashboard
if page == "📊 Dashboard":
    st.header("Dashboard global")
    st.caption("Vue d'ensemble : totaux, tendances, dernière session.")

    total = sum([h["total"] for h in st.session_state.history]) if st.session_state.history else 0
    vrais = sum([h["vrai"] for h in st.session_state.history]) if st.session_state.history else 0
    faux  = sum([h["faux"] for h in st.session_state.history]) if st.session_state.history else 0

    k1,k2,k3 = st.columns(3)
    k1.markdown(f"<div class='card kpi'><h3>Total analysé</h3><div class='val'>{total}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='card kpi'><h3>Billets validés</h3><div class='val' style='color:{PRIMARY}'>{vrais}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='card kpi'><h3>Faux billets</h3><div class='val' style='color:{ACCENT}'>{faux}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.history:
        dfh = pd.DataFrame(st.session_state.history)
        dfh["date"] = pd.to_datetime(dfh["date"])
        c1, c2 = st.columns([2,1])
        with c1:
            if PLOTLY_OK:
                fig = px.line(dfh, x="date", y=["vrai","faux","total"], markers=True, title="Tendances des analyses")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(dfh.set_index("date")[["vrai","faux","total"]])
        with c2:
            last = dfh.tail(1)
            if not last.empty:
                last_vals = last.iloc[0]
                st.markdown("**Dernière session**")
                st.write(pd.DataFrame({"métrique":["total","vrai","faux"],
                                       "valeur":[int(last_vals.total), int(last_vals.vrai), int(last_vals.faux)]}))
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
            end   = st.date_input("Jusqu'au", value=datetime.now().date())
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
    if c1.button("Voir les dernières analyses"):
        st.session_state["__go_hist__"] = True
        st.experimental_rerun()
    if c2.button("Faire une nouvelle analyse"):
        st.session_state["__go_analyse__"] = True
        st.experimental_rerun()

# =============================
# FOOTER
# =============================
st.markdown(
    f"<div class='footer'>© {datetime.now().year} – Sona Koulibaly • Streamlit × FastAPI × ML • Green/Black UI</div>",
    unsafe_allow_html=True,
)
