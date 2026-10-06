# Dashboard Streamlit - SCI 1402

## Projet
Analyse et prédiction de l'abandon d'une résolution d'entraînement à partir de données synthétiques.

## Contenu
- `app.py` : application Streamlit
- `data/gym_ghosts_master.csv` : jeu de données principal
- `data/data_dictionary.csv` : dictionnaire des variables
- `requirements.txt` : dépendances Python

## Exécution locale

1. Installer Python 3.10 ou plus récent.
2. Ouvrir un terminal dans ce dossier.
3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Lancer le Dashboard :

```bash
streamlit run app.py
```

Le navigateur devrait s'ouvrir automatiquement.

## Déploiement sur Streamlit Community Cloud

1. Déposer ce dossier dans un dépôt GitHub.
2. Aller sur Streamlit Community Cloud.
3. Créer une nouvelle application à partir du dépôt.
4. Choisir `app.py` comme fichier principal.
5. Déployer.

## Remarque méthodologique

Les statistiques exploratoires sont calculées directement à partir du fichier CSV fourni.
Les métriques des modèles affichées dans l'onglet `Modèles` correspondent aux résultats documentés dans le TN2.

Le jeu de données est entièrement synthétique. Les résultats ne doivent pas être généralisés directement à de véritables membres de centres d'entraînement.
