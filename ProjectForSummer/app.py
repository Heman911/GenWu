import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Multi-Game Rarity Predictor", page_icon="🎮", layout="centered")
st.title("🎮 Multi-Game Rarity Predictor (4★ vs 5★)")
st.caption("Supports **Genshin** + **Wuthering Waves** using a single combined model.")

DATA_PATH = Path(".")
MODEL_PATH = DATA_PATH / "combined_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_csv(path):
    # Accept either Genshin or WuWa schema; we'll harmonize on the fly for display
    return pd.read_csv(path, encoding="latin-1", low_memory=False)

# Detect datasets
datasets = {}
if (DATA_PATH / "genshin.csv").exists():
    datasets["Genshin"] = DATA_PATH / "genshin.csv"
if (DATA_PATH / "wutheringwaves_character.csv").exists():
    datasets["Wuthering Waves"] = DATA_PATH / "wutheringwaves_character.csv"
if not datasets:
    st.error("Place your CSVs next to app.py (genshin.csv, wutheringwaves_character.csv).")
    st.stop()

game = st.selectbox("Select Game", list(datasets.keys()))
raw = load_csv(datasets[game])

# Build a harmonized row builder so inputs match what the model saw
def to_unified(df, game_name):
    if game_name == "Genshin":
        return pd.DataFrame({
            'game':        ['Genshin']*len(df),
            'weapon':      df.get('weapon_type'),
            'element':     df.get('vision'),
            'role':        [np.nan]*len(df),
            'region':      df.get('region'),
            'affiliation': [np.nan]*len(df),
            'model_size':  df.get('model'),
            'hp':          df.get('hp_1_20'),
            'atk':         df.get('atk_1_20'),
            'def':         df.get('def_1_20'),
            'character':   df.get('character_name', pd.Series([np.nan]*len(df)))
        })
    else:
        return pd.DataFrame({
            'game':        ['Wuthering']*len(df),
            'weapon':      df.get('Weapon'),
            'element':     df.get('Attribute'),
            'role':        df.get('Role'),
            'region':      df.get('Birthplace'),
            'affiliation': df.get('Affiliation'),
            'model_size':  [np.nan]*len(df),
            'hp':          df.get('HP'),
            'atk':         df.get('ATK'),
            'def':         df.get('DEF'),
            'character':   df.get('Character')
        })

model = load_model()
uni = to_unified(raw, game)

st.subheader("🔎 Predict by character")
name_col = 'character_name' if game == "Genshin" and 'character_name' in raw.columns else ('Character' if 'Character' in raw.columns else None)
if name_col:
    name = st.selectbox("Character", options=sorted(raw[name_col].dropna().unique().tolist()))
    row_raw = raw.loc[raw[name_col] == name].head(1)
    row_uni = to_unified(row_raw, game).iloc[0:1]
    pred = int(model.predict(row_uni)[0])
    try:
        proba = model.predict_proba(row_uni)[0]
        classes = model.classes_
        p5 = dict(zip(classes, proba)).get(5, None)
    except Exception:
        p5 = None

    show_cols = [c for c in ['weapon','element','role','region','affiliation','model_size','hp','atk','def'] if c in row_uni.columns]
    st.write("**Features:**")
    st.write(row_uni[show_cols].T)

    st.markdown("### 🎯 Prediction")
    st.write(f"**Predicted Rarity:** {'5★' if pred==5 else '4★'}")
    if p5 is not None:
        st.write(f"**Probability of 5★:** {p5:.2%}")

st.divider()
st.subheader("🧮 Predict from manual inputs")

game_choice = st.selectbox("Game for manual input", ["Genshin", "Wuthering"], index=0)
weapon = st.text_input("Weapon", "Sword")
element = st.text_input("Element / Attribute", "Pyro" if game_choice=="Genshin" else "Spectro")
role = st.text_input("Role (optional)", "" if game_choice=="Genshin" else "Main Damage Dealer")
region = st.text_input("Region / Birthplace (optional)", "")
affiliation = st.text_input("Affiliation (optional)", "")
model_size = st.text_input("Model Size (optional)", "Tall Male" if game_choice=="Genshin" else "")

hp = st.number_input("HP", min_value=0.0, value=float(pd.to_numeric(uni['hp'], errors='coerce').median(skipna=True) or 800))
atk = st.number_input("ATK", min_value=0.0, value=float(pd.to_numeric(uni['atk'], errors='coerce').median(skipna=True) or 20))
defn = st.number_input("DEF", min_value=0.0, value=float(pd.to_numeric(uni['def'], errors='coerce').median(skipna=True) or 80))

if st.button("Predict (manual)"):
    row = pd.DataFrame([{
        'game': 'Genshin' if game_choice=='Genshin' else 'Wuthering',
        'weapon': weapon or np.nan,
        'element': element or np.nan,
        'role': role or np.nan,
        'region': region or np.nan,
        'affiliation': affiliation or np.nan,
        'model_size': model_size or np.nan,
        'hp': hp, 'atk': atk, 'def': defn,
        'character': np.nan
    }])
    try:
        p = int(model.predict(row)[0])
        prob = None
        try:
            proba = model.predict_proba(row)[0]
            classes = model.classes_
            prob = dict(zip(classes, proba)).get(5, None)
        except Exception:
            pass
        st.success(f"Predicted Rarity: {'5★' if p==5 else '4★'}")
        if prob is not None:
            st.write(f"Probability of 5★: {prob:.2%}")
    except Exception as e:
        st.error(f"Could not predict: {e}")
