import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px

# Répertoire où les modèles sont sauvegardés
MODEL_DIR = 'models'

def load_model(article):
    """Charger un modèle Prophet à partir d'un fichier .pkl"""
    article_filename = article.replace('/', '_')
    filename = os.path.join(MODEL_DIR, f'{article_filename}_prophet_model.pkl')

    if os.path.exists(filename):
        model = joblib.load(filename)
        return model
    else:
        return None

# Interface utilisateur avec Streamlit
st.title('Prévisions des ventes Lamacom')

# Saisie de l'article, du type de prévision et de l'année
article = st.text_input('Article')
forecast_type = st.selectbox('Type de prévision', ['Mensuelles', 'Annuelles'])
year = st.number_input('Année', min_value=2024, max_value=2100, value=2024)

if st.button('Prévoir'):
    with st.spinner('Chargement des prévisions...'):
        # Charger le modèle pour l'article spécifié
        model = load_model(article)
        if model is None:
            st.error(f"Article {article} non trouvé.")
        else:
            if forecast_type == 'Mensuelles':
                # Prévision mensuelle pour chaque mois de l'année sélectionnée
                future = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M').to_frame(index=False, name='ds')
                forecast = model.predict(future)
                predictions = forecast[['ds', 'yhat']]

                # Convertir les valeurs prévues en entiers
                predictions['yhat'] = predictions['yhat'].astype(int)

                # Remplacer les valeurs négatives par zéro
                predictions['yhat'] = predictions['yhat'].clip(lower=0)

                # Convertir la colonne 'ds' pour afficher seulement la date (sans l'heure)
                predictions['ds'] = predictions['ds'].dt.date

                # Renommer les colonnes
                predictions.rename(columns={'ds': 'Date', 'yhat': 'Quantité prévue'}, inplace=True)

                # Réinitialiser l'index pour enlever la colonne d'index
                predictions.reset_index(drop=True, inplace=True)

                # Formater les quantités prévues avec espace pour les milliers
                predictions['Quantité prévue'] = predictions['Quantité prévue'].apply(lambda x: f"{x:,}".replace(',', ' '))

                # Créer un tableau HTML avec formatage conditionnel
                html_table = '<table style="width:100%; border-collapse: collapse;">'
                html_table += '<tr><th>Date</th><th>Quantité prévue</th></tr>'
                for _, row in predictions.iterrows():
                    quantity_style = 'font-weight: bold; color: green;' if row['Quantité prévue'].replace(' ', '').isdigit() and int(row['Quantité prévue'].replace(' ', '')) > 50000 else ''
                    html_table += f'<tr><td>{row["Date"]}</td><td style="{quantity_style}">{row["Quantité prévue"]}</td></tr>'
                html_table += '</table>'

                st.write(f"Prévisions mensuelles pour l'article {article} en {year}:")
                st.markdown(html_table, unsafe_allow_html=True)

                # Tracer le graphique des prévisions
                fig = px.line(predictions, x='Date', y='Quantité prévue',
                              title=f"Prévisions mensuelles pour l'article {article} en {year}",
                              labels={'Date': 'Date', 'Quantité prévue': 'Quantité prévue'})
                st.plotly_chart(fig)

            elif forecast_type == 'Annuelles':
                # Prévision annuelle : Somme des prévisions mensuelles pour toute l'année
                future = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M').to_frame(index=False, name='ds')
                forecast = model.predict(future)
                total_prediction = forecast['yhat'].sum().astype(int)
                if total_prediction < 0:
                    total_prediction = 0

                # Formater le total avec espace pour les milliers
                total_prediction_formatted = f"{total_prediction:,}".replace(',', ' ')

                st.write(f"Prévision annuelle pour l'article {article} en {year} : {total_prediction_formatted}")

