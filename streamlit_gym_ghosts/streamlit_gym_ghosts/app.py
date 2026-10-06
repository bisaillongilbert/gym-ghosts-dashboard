
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Gym Ghosts - SCI 1402",
    page_icon="🏋️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gym_ghosts_master.csv"
DICT_PATH = BASE_DIR / "data" / "data_dictionary.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    data_dict = pd.read_csv(DICT_PATH)
    return df, data_dict

df, data_dict = load_data()

# Labels used in the interface
status_labels = {
    0: "Maintien",
    1: "Abandon",
}

df_display = df.copy()
df_display["Statut"] = df_display["churned_by_february"].map(status_labels)

st.title("Analyse et prédiction de l'abandon d'une résolution d'entraînement")
st.caption("Projet SCI 1402 - Tableau de bord Streamlit")

st.info(
    "Ce jeu de données est entièrement synthétique. Les résultats présentés servent à "
    "illustrer une démarche de science des données et ne doivent pas être généralisés "
    "directement à de véritables membres de centres d'entraînement."
)

# Sidebar filters
st.sidebar.header("Filtres")
contract_options = ["Tous"] + sorted(df["contract_type"].dropna().unique().tolist())
sex_options = ["Tous"] + sorted(df["sex"].dropna().unique().tolist())

contract_choice = st.sidebar.selectbox("Type de contrat", contract_options)
sex_choice = st.sidebar.selectbox("Sexe", sex_options)

filtered = df_display.copy()
if contract_choice != "Tous":
    filtered = filtered[filtered["contract_type"] == contract_choice]
if sex_choice != "Tous":
    filtered = filtered[filtered["sex"] == sex_choice]

tabs = st.tabs(
    [
        "Vue d'ensemble",
        "Analyse exploratoire",
        "Modèles",
        "Interprétation",
        "Dictionnaire et limites",
    ]
)

with tabs[0]:
    st.subheader("Vue d'ensemble du jeu de données")

    n = len(filtered)
    churn_rate = filtered["churned_by_february"].mean() * 100 if n else 0
    retained_rate = 100 - churn_rate if n else 0
    avg_w1 = filtered["week1_visits"].mean() if n else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations affichées", f"{n:,}".replace(",", " "))
    c2.metric("Taux d'abandon", f"{churn_rate:.2f} %")
    c3.metric("Taux de maintien", f"{retained_rate:.2f} %")
    c4.metric("Visites moyennes semaine 1", f"{avg_w1:.2f}")

    st.markdown("### Distribution de la variable cible")
    target_counts = (
        filtered["Statut"]
        .value_counts()
        .rename_axis("Statut")
        .to_frame("Nombre de membres")
    )
    st.bar_chart(target_counts)

    st.markdown("### Aperçu des données filtrées")
    st.dataframe(
        filtered.drop(columns=["Statut"]).head(100),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Analyse exploratoire")

    st.markdown("### Fréquentation moyenne pendant les quatre premières semaines")
    attendance = (
        df.groupby("churned_by_february")[
            ["week1_visits", "week2_visits", "week3_visits", "week4_visits"]
        ]
        .mean()
        .T
    )
    attendance.columns = ["Maintien", "Abandon"]
    attendance.index = ["Semaine 1", "Semaine 2", "Semaine 3", "Semaine 4"]
    st.line_chart(attendance)

    st.dataframe(
        attendance.round(2),
        use_container_width=True,
    )

    st.markdown("### Taux d'abandon selon le type de contrat")
    contract_churn = (
        df.groupby("contract_type")["churned_by_february"]
        .mean()
        .mul(100)
        .sort_values()
        .to_frame("Taux d'abandon (%)")
    )
    st.bar_chart(contract_churn)
    st.dataframe(contract_churn.round(2), use_container_width=True)

    st.markdown("### Taux d'abandon selon le nombre de visites en semaine 1")
    week1_churn = (
        df.groupby("week1_visits")["churned_by_february"]
        .mean()
        .mul(100)
        .to_frame("Taux d'abandon (%)")
    )
    st.line_chart(week1_churn)

    st.markdown("### Indicateurs d'engagement")
    engagement_vars = {
        "joined_with_friend": "Inscription avec un ami",
        "booked_induction": "Séance d'introduction",
        "personal_trainer": "Entraîneur personnel",
        "app_installed": "Application installée",
        "locker_rented": "Casier loué",
    }

    rows = []
    for col, label in engagement_vars.items():
        rates = df.groupby(col)["churned_by_february"].mean().mul(100)
        rows.append(
            {
                "Indicateur": label,
                "Oui (%)": rates.get(1, float("nan")),
                "Non (%)": rates.get(0, float("nan")),
            }
        )
    engagement_table = pd.DataFrame(rows).set_index("Indicateur")
    st.dataframe(engagement_table.round(2), use_container_width=True)

with tabs[2]:
    st.subheader("Comparaison des modèles")

    st.write(
        "Les valeurs ci-dessous correspondent aux résultats documentés dans le TN2 "
        "sur l'ensemble de test de 12 500 observations."
    )

    model_results = pd.DataFrame(
        {
            "Modèle": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            ],
            "Accuracy (%)": [74.18, 71.38, 73.88],
            "Precision (%)": [70.88, 67.89, 70.97],
            "Recall (%)": [72.59, 69.38, 71.26],
            "F1 (%)": [71.72, 68.63, 71.11],
            "ROC-AUC": [0.823, 0.786, 0.819],
        }
    ).set_index("Modèle")

    st.dataframe(model_results, use_container_width=True)
    st.markdown("### ROC-AUC sur l'ensemble de test")
    st.bar_chart(model_results[["ROC-AUC"]])

    st.markdown("### Validation croisée stratifiée à cinq plis")
    cv_results = pd.DataFrame(
        {
            "Modèle": [
                "Logistic Regression",
                "Random Forest",
                "Decision Tree",
            ],
            "ROC-AUC moyen": [0.830, 0.821, 0.781],
        }
    ).set_index("Modèle")
    st.dataframe(cv_results, use_container_width=True)
    st.bar_chart(cv_results)

    st.markdown("### Matrice de confusion de la Logistic Regression")
    confusion = pd.DataFrame(
        [[5178, 1682], [1546, 4094]],
        index=["Réel : maintien", "Réel : abandon"],
        columns=["Prédit : maintien", "Prédit : abandon"],
    )
    st.dataframe(confusion, use_container_width=True)

    st.write(
        "La Logistic Regression obtient le meilleur ROC-AUC parmi les trois modèles "
        "documentés, avec 0,823 sur l'ensemble de test et 0,830 en validation croisée."
    )

with tabs[3]:
    st.subheader("Interprétation des résultats")

    importance = pd.DataFrame(
        {
            "Variable": [
                "Visites semaine 1",
                "Cours de groupe par semaine",
                "Visites semaine 2",
                "Visites semaine 3",
                "Visites semaine 4",
                "Distance du centre",
                "Frais mensuels",
                "Durée moyenne des séances",
            ],
            "Importance (%)": [16.1, 13.7, 12.4, 9.8, 8.3, 6.8, 5.4, 5.0],
        }
    ).set_index("Variable")

    st.markdown("### Importance approximative des variables dans la Random Forest")
    st.bar_chart(importance)
    st.dataframe(importance, use_container_width=True)

    st.markdown("### Principaux constats")
    st.write(
        """
        Les comportements observés durant les premières semaines constituent les
        variables les plus utiles à la prédiction. La fréquentation en semaine 1
        occupe notamment une place importante dans les résultats.

        Le type de contrat, la distance et plusieurs indicateurs d'engagement
        présentent également des associations avec le taux d'abandon.

        La Logistic Regression obtient une performance légèrement supérieure à la
        Random Forest selon le ROC-AUC documenté. Cela montre qu'un modèle plus
        complexe n'est pas nécessairement meilleur.
        """
    )

    st.warning(
        "L'importance d'une variable dans un modèle prédictif ne démontre pas une "
        "relation causale."
    )

with tabs[4]:
    st.subheader("Dictionnaire de données")
    st.dataframe(data_dict, use_container_width=True, hide_index=True)

    st.markdown("### Limites du projet")
    st.write(
        """
        - Le jeu de données est entièrement synthétique.
        - Les résultats ne doivent pas être généralisés directement à une population réelle.
        - Les associations observées ne permettent pas d'établir des relations causales.
        - Les performances présentées décrivent les modèles testés dans le cadre du projet.
        - Une utilisation réelle nécessiterait une validation sur des données provenant de véritables centres d'entraînement.
        """
    )

    st.markdown("### Source")
    st.write(
        "S. Nefedov, *GYM. Will Your New Year's Resolution Survive?*, Kaggle."
    )

st.divider()
st.caption(
    "SCI 1402 - Projet en science des données | Dashboard réalisé avec Streamlit"
)
