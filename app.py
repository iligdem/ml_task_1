import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from pathlib import Path
from utils import *

st.set_page_config(page_title="Предсказание стоимости автомобилей", page_icon="🚗", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "full_pipeline.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
TRAIN_DATA_PATH = Path(__file__).resolve().parent / "cars_train.csv" 
IMAGE_PATH = Path(__file__).resolve().parent / "indian_seller.jpg" 

st.image(str(IMAGE_PATH), caption="Индийский продавец автомобилей", width=None)
@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = joblib.load(f)
    return model, feature_names

@st.cache_data
def load_train_data():
    """Загружаем обучающие данные для анализа"""
    if TRAIN_DATA_PATH.exists():
        return pd.read_csv(TRAIN_DATA_PATH)
    else:
        st.error(f"❌ Не найден файл обучающих данных: {TRAIN_DATA_PATH}")
        st.stop()

# Загружаем модель
try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# --- Загрузка обучающих данных и визуализации ---
st.title("📊 Анализ обучающих данных")

train_df = load_train_data()


# --- Визуализации ---
st.subheader("📈 Визуализации")

fuel_price = train_df.groupby('fuel')['selling_price'].mean().sort_values()
fig1 = px.pie(
    values=fuel_price.values,
    names=fuel_price.index,
    title="Распределение цены по типу топлива"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(train_df, x='selling_price', nbins=30, title="Распределение стоимости автомобилей")
st.plotly_chart(fig2, use_container_width=True)

if 'name' in train_df.columns:
    plan_df = train_df.groupby('name')['selling_price'].median().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(plan_df, x='name', y='selling_price', 
                  title="Медианная цена по названиям автомобилей (ТОП-10)")
    st.plotly_chart(fig3, use_container_width=True)

fig4 = px.scatter(train_df, x='year', y='selling_price', hover_data=['name'],
                  title="Цена в зависимости от года выпуска")
st.plotly_chart(fig4, use_container_width=True)







# --- Основной интерфейс ---
st.title("🎯 Предсказание стоимости автомобилей")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file).drop(['selling_price'], axis=1)
features_orig = list(df)

try:
    features = FEATURE_NAMES
    predictions_log = MODEL.predict(df)
    
    df['prediction_log'] = predictions_log
    predictions = np.exp(predictions_log)
    df['prediction'] = predictions

except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# --- Метрики ---
st.subheader("📊 Результаты")

col1, col2 = st.columns(2)
with col1:
    st.metric("Всего автомобилей", len(df))
with col2:
    average_cost = df['prediction'].mean()
    st.metric("Средняя предсказанная цена автомобиля ", f"{average_cost:.0f}у.е.")


# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для нового автомобиля")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}
    
    with col_left:
        st.write("**Категориальные:**")
        for col in features_orig:
            if df[col].dtype in ('object', 'bool'):
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")
    
    with col_right:
        st.write("**Числовые:**")
        for col in features_orig:
            if df[col].dtype not in ('object', 'bool'):
                val = int(df[col].median())
                input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[features_orig]
        pred = MODEL.predict(input_df)[0]
     

        st.success(f"**Результат:** Стоимость автомобиля с данными параметрами равна {np.exp(pred):.0f} у.е")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")