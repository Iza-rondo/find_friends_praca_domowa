import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import os
from io import BytesIO
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

env = dotenv_values(".env")

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

QDRANT_COLLECTION_NAME = "notes"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA, sep=";")
    # proste czyszczenie braków
    df = df.fillna("unknown")
    return df

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

df = pd.read_csv("welcome_survey_simple_v2.csv", sep=";", na_values=["", " "]).fillna("unknown")


with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.pie(same_cluster_df, names="fav_animals", title="Ulubione zwierzęta (%)")
st.plotly_chart(fig)

categories = ["Nad wodą", "W lesie", "W górach", "Inne"]
counts = same_cluster_df["fav_place"].value_counts().reindex(categories, fill_value=0)

fig = px.pie(
    names=counts.index,
    values=counts.values,
    title="Ulubione miejsca (%)"
)
st.plotly_chart(fig)
