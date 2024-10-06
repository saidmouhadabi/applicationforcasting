import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px
import io

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

def create_combined_excel_file(all_predictions, forecast_type, year):
    """Créer un fichier Excel avec les prévisions combinées"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        if forecast_type == 'Mensuelles':
            # Combiner les prévisions mensuelles
            combined_df = pd.DataFrame()
            for article, df in all_predictions.items():
                df_pivot = df.pivot(index=None, columns='Date', values='Quantité prévue')
                df_pivot.columns = [f'{article}_{col}' for col in df_pivot.columns]
                combined_df = pd.concat([combined_df, df_pivot], axis=1)
            combined_df.to_excel(writer, sheet_name='Prévisions_Mensuelles', index=False)
        elif forecast_type == 'Annuelles':
            # Prévisions annuelles
            combined_df = pd.DataFrame(all_predictions).T
            combined_df.columns = ['Quantité prévue']
            combined_df.to_excel(writer, sheet_name='Prévisions_Annuelles', index=True)
        writer.save()
    buffer.seek(0)
    return buffer

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

                predictions['yhat'] = predictions['yhat'].astype(int)
                predictions['yhat'] = predictions['yhat'].clip(lower=0)
                predictions['ds'] = predictions['ds'].dt.date
                predictions.rename(columns={'ds': 'Date', 'yhat': 'Quantité prévue'}, inplace=True)
                predictions.reset_index(drop=True, inplace=True)
                predictions['Quantité prévue'] = predictions['Quantité prévue'].apply(lambda x: f"{x:,}".replace(',', ' '))

                html_table = '<table style="width:100%; border-collapse: collapse;">'
                html_table += '<tr><th>Date</th><th>Quantité prévue</th></tr>'
                for _, row in predictions.iterrows():
                    quantity_style = 'font-weight: bold; color: green;' if row['Quantité prévue'].replace(' ', '').isdigit() and int(row['Quantité prévue'].replace(' ', '')) > 50000 else ''
                    html_table += f'<tr><td>{row["Date"]}</td><td style="{quantity_style}">{row["Quantité prévue"]}</td></tr>'
                html_table += '</table>'

                st.write(f"Prévisions mensuelles pour l'article {article} en {year}:")
                st.markdown(html_table, unsafe_allow_html=True)

                fig = px.line(predictions, x='Date', y='Quantité prévue',
                              title=f"Prévisions mensuelles pour l'article {article} en {year}",
                              labels={'Date': 'Date', 'Quantité prévue': 'Quantité prévue'})
                st.plotly_chart(fig)

            elif forecast_type == 'Annuelles':
                future = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M').to_frame(index=False, name='ds')
                forecast = model.predict(future)
                total_prediction = forecast['yhat'].sum().astype(int)
                if total_prediction < 0:
                    total_prediction = 0

                total_prediction_formatted = f"{total_prediction:,}".replace(',', ' ')
                st.write(f"Prévision annuelle pour l'article {article} en {year} : {total_prediction_formatted}")

# Nouveau formulaire pour prévisions de tous les articles
if st.button('Prévoir Tout'):
    with st.form(key='all_articles_form'):
        year_all = st.number_input('Année pour toutes les prévisions', min_value=2024, max_value=2100, value=2024)
        forecast_type_all = st.selectbox('Type de prévision pour toutes les prévisions', ['Mensuelles', 'Annuelles'])
        submit_button = st.form_submit_button(label='Télécharger')

        if submit_button:
            with st.spinner('Génération du fichier Excel...'):
                # Liste d'articles à traiter (ajoutez vos articles ici ou obtenez-les dynamiquement)
                articles = ['article1', 'article2', 'article3']  # Remplacez par la liste de vos articles
                all_predictions = {}

                for article in articles:
                    model = load_model(article)
                    if model is None:
                        st.error(f"Article {article} non trouvé.")
                        continue

                    if forecast_type_all == 'Mensuelles':
                        future = pd.date_range(start=f'{year_all}-01-01', end=f'{year_all}-12-31', freq='M').to_frame(index=False, name='ds')
                        forecast = model.predict(future)
                        predictions = forecast[['ds', 'yhat']]

                        predictions['yhat'] = predictions['yhat'].astype(int)
                        predictions['yhat'] = predictions['yhat'].clip(lower=0)
                        predictions['ds'] = predictions['ds'].dt.date
                        predictions.rename(columns={'ds': 'Date', 'yhat': 'Quantité prévue'}, inplace=True)
                        predictions.reset_index(drop=True, inplace=True)
                        predictions['Quantité prévue'] = predictions['Quantité prévue'].apply(lambda x: f"{x:,}".replace(',', ' '))

                        all_predictions[article] = predictions

                    elif forecast_type_all == 'Annuelles':
                        future = pd.date_range(start=f'{year_all}-01-01', end=f'{year_all}-12-31', freq='M').to_frame(index=False, name='ds')
                        forecast = model.predict(future)
                        total_prediction = forecast['yhat'].sum().astype(int)
                        if total_prediction < 0:
                            total_prediction = 0

                        total_prediction_formatted = f"{total_prediction:,}".replace(',', ' ')
                        annual_data = pd.DataFrame({'Année': [year_all], 'Quantité prévue': [total_prediction_formatted]})
                        all_predictions[article] = annual_data

                if all_predictions:
                    excel_buffer = create_combined_excel_file(all_predictions, forecast_type_all, year_all)
                    st.download_button(label='Télécharger les prévisions de tous les articles', data=excel_buffer, file_name=f'prévisions_combinées_{forecast_type_all}_{year_all}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                else:
                    st.warning("Aucune prévision disponible pour les articles sélectionnés.")
