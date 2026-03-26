
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Multi-Game Rarity Predictor (Pro)", page_icon="🎯", layout="centered")
st.title("🎯 Multi-Game Rarity Predictor — Pro")
st.caption("Per‑game model + decision threshold + ground truth display.")

DATA = Path(".")

def load_model_for(game: str):
    try:
        if game == "Genshin" and (DATA / "model_genshin.pkl").exists():
            return joblib.load(DATA / "model_genshin.pkl")
        return joblib.load(DATA / "combined_model.pkl")
    except Exception as e:
        st.error("Couldn't load model. If this is a version issue, retrain locally or match scikit-learn version.")
        st.code(str(e))
        st.stop()

@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path, encoding="latin-1", low_memory=False)

def unify(df: pd.DataFrame, game_name: str) -> pd.DataFrame:
    if game_name == "Genshin":
        return pd.DataFrame({
            "game":        ["Genshin"] * len(df),
            "weapon":      df.get("weapon_type"),
            "element":     df.get("vision"),
            "role":        [np.nan] * len(df),
            "region":      df.get("region"),
            "affiliation": [np.nan] * len(df),
            "model_size":  df.get("model"),
            "hp":          df.get("hp_1_20"),
            "atk":         df.get("atk_1_20"),
            "def":         df.get("def_1_20"),
            "character":   df.get("character_name", pd.Series([np.nan]*len(df)))
        })
    else:
        return pd.DataFrame({
            "game":        ["Wuthering"] * len(df),
            "weapon":      df.get("Weapon"),
            "element":     df.get("Attribute"),
            "role":        df.get("Role"),
            "region":      df.get("Birthplace"),
            "affiliation": df.get("Affiliation"),
            "model_size":  [np.nan] * len(df),
            "hp":          df.get("HP"),
            "atk":         df.get("ATK"),
            "def":         df.get("DEF"),
            "character":   df.get("Character")
        })

datasets = {}
if (DATA / "genshin.csv").exists():
    datasets["Genshin"] = DATA / "genshin.csv"
if (DATA / "wutheringwaves_character.csv").exists():
    datasets["Wuthering Waves"] = DATA / "wutheringwaves_character.csv"

if not datasets:
    st.error("Put genshin.csv and/or wutheringwaves_character.csv next to this file.")
    st.stop()

game = st.selectbox("Game", list(datasets.keys()))
raw = load_csv(datasets[game])
model = load_model_for(game)
uni_all = unify(raw, game)

st.subheader("🔎 Predict by character (uses dataset row)")

name_col = 'character_name' if (game == "Genshin" and 'character_name' in raw.columns) else ('Character' if 'Character' in raw.columns else None)
if name_col:
    name = st.selectbox("Character", options=sorted(raw[name_col].dropna().unique().tolist()))
    row_raw = raw.loc[raw[name_col] == name].head(1)
    row_uni = unify(row_raw, game).iloc[0:1]

    pred = int(model.predict(row_uni)[0])
    p5 = None
    try:
        proba = model.predict_proba(row_uni)[0]
        classes = model.classes_
        p5 = dict(zip(classes, proba)).get(5, None)
    except Exception:
        pass

    true = None
    if 'rarity' in row_raw.columns:
        try:
            true = int(row_raw['rarity'].iloc[0])
        except Exception:
            true = None

    st.markdown("### 🎯 Prediction")
    th = st.slider("Decision threshold for 5★", 0.30, 0.70, 0.50, 0.01,
                   help="If P(5★) ≥ threshold, classify as 5★")
    label = 5 if (p5 is not None and p5 >= th) else pred
    st.write(f"**Predicted Rarity:** {'5★' if label==5 else '4★'}")
    if p5 is not None:
        st.write(f"**Probability of 5★:** {p5:.2%}")
    if true is not None:
        st.write(f"**Ground truth:** {'5★' if true==5 else '4★'}")

    with st.expander("Show unified features for this character"):
        st.write(row_uni.T)

st.divider()
st.subheader("🧮 Manual input")
game_choice = st.selectbox("Game for manual input", ["Genshin", "Wuthering"], index=0)
weapon = st.text_input("Weapon", "Sword")
element = st.text_input("Element / Attribute", "Pyro" if game_choice=='Genshin' else "Spectro")
role = st.text_input("Role (optional)", "" if game_choice=='Genshin' else "Main Damage Dealer")
region = st.text_input("Region/Birthplace (optional)", "")
aff = st.text_input("Affiliation (optional)", "")
model_size = st.text_input("Model Size (optional)", "Tall Male" if game_choice=='Genshin' else "")

hp_default = float(pd.to_numeric(uni_all['hp'], errors='coerce').median(skipna=True) or 800)
atk_default = float(pd.to_numeric(uni_all['atk'], errors='coerce').median(skipna=True) or 20)
def_default = float(pd.to_numeric(uni_all['def'], errors='coerce').median(skipna=True) or 80)

hp = st.number_input("HP", value=hp_default)
atk = st.number_input("ATK", value=atk_default)
defn = st.number_input("DEF", value=def_default)

if st.button("Predict (manual)"):
    row = pd.DataFrame([{
        'game': 'Genshin' if game_choice=='Genshin' else 'Wuthering',
        'weapon': weapon or np.nan,
        'element': element or np.nan,
        'role': role or np.nan,
        'region': region or np.nan,
        'affiliation': aff or np.nan,
        'model_size': model_size or np.nan,
        'hp': hp, 'atk': atk, 'def': defn,
        'character': np.nan
    }])
    try:
        p_pred = int(model.predict(row)[0])
        p5m = None
        try:
            proba = model.predict_proba(row)[0]
            classes = model.classes_
            p5m = dict(zip(classes, proba)).get(5, None)
        except Exception:
            pass

        if p5m is not None:
            th2 = st.slider("Manual input threshold for 5★", 0.30, 0.70, 0.50, 0.01, key="th_manual")
            label2 = 5 if p5m >= th2 else 4
            st.success(f"Predicted Rarity: {'5★' if label2==5 else '4★'}")
            st.write(f"Probability of 5★: {p5m:.2%}")
        else:
            st.success(f"Predicted Rarity: {'5★' if p_pred==5 else '4★'}")
    except Exception as e:
        st.error(f"Could not predict: {e}")
