from tqdm import tqdm
import os
from preprocess import load_csv
from predict import predict_category1,predict_category2, train_model
from evaluate import evaluate_predictions1, evaluate_predictions2,plot_evaluate_classification2
import streamlit as st
from visualization import plot_intentions, chie_pie_words_count, boxplot_cat
import pandas as pd


def main():

    st.title("Intent Detection App")

    # Upload csv
    uploaded_file = st.file_uploader("Choose a csv file")

    # Bouton pour run le code
    if st.button("Run"):
        # Add spinner
        with st.spinner("Running..."):
            
            if uploaded_file is not None:
                with open("data/intent-detection-train.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())


            input_csv = "data/intent-detection-train.csv"
            output_csv = "output/predictions.csv"
            openai_api_key = "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"

            
            # Charger et préparer les données
            df, df_labels1, df_labels2 = load_csv(input_csv)

            # Visualization 
            plot_intentions(df)
            chie_pie_words_count(df)
            boxplot_cat(df)

            # Vérifier si le fichier data-augmented.csv existe déjà
            data_augmented_path = "data/data-augmented.csv"

            if not os.path.exists(data_augmented_path):
                print("Le fichier data-augmented.csv n'existe pas. Lancement de la data augmentation...")
                
                # Appliquer le preprocessing pour un modèle de classification
                category_counts = Counter(df["label"])
                target_size = max(category_counts.values())  # Taille cible = max des échantillons

                # Appliquer la Data Augmentation
                df_NLP = apply_data_augmentation(df, category_counts, target_size)

                print("Data augmentation terminée et sauvegardée.")
            else:
                print("Le fichier data-augmented.csv existe déjà. Chargement des données avec augmentation pour réaliser la Random Forest.")
                df_NLP = pd.read_csv(data_augmented_path)  # Charger directement le fichier

            st.write("J'ai recrée une data set avec des nouvelles données pour nos mdèles de classifications. On peut observer qu'il y a une meilleure répartition des données :")
            plot_intentions(df_NLP)

             # Appliquer Random Forest & Logistic Regression avec GridSearchCV
            X_test,y_test, y_pred, best_model_name = train_model(df_NLP)
            
            st.write(f" Meilleur modèle sélectionné : {best_model_name}")

            # Afficher la Matrice de Confusion
            df_final=plot_evaluate_classification2(X_test,y_test, y_pred, best_model_name)
            st.write(df_final)


            df.drop(columns=["label"], inplace=True)  # Suppression de la colonne label

            # Ajout des colonnes de prédictions vides
            df["cat_prediction_openai"] = None  
            df["cat_prediction_ollama"] = None 

            # Effectuer les prédictions avec OpenAI
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions OpenAI"):
                df.at[index, "cat_prediction_openai"] = predict_category1(row["text"], openai_api_key)

            final_score_openai = evaluate_predictions1(df, df_labels1)


            df_labels1["cat_prediction"] = df["cat_prediction_openai"]
            
            if not os.path.exists(output_csv):
                os.makedirs(output_dir)

            df_labels1.to_csv("output_csv/predictions_openai.csv", index=False)

            st.write("Ensuite, voici les prédictions de Open AI :")
            st.write(df_labels1)
            st.write(f"Pourcentage de bonnes prédictions OpenAI : {final_score_openai:.2f}%")

            # Effectuer les prédictions avec Ollama
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions Ollama"):
                df.at[index, "cat_prediction_ollama"] = predict_category2(row["text"])

          
            final_score_ollama = evaluate_predictions2(df, df_labels2)
    

            df_labels2["cat_prediction"] = df["cat_prediction_ollama"]
            df_labels2.to_csv("output_csv/predictions_ollama.csv", index=False)
            # write csv
            st.write("Pour finir, voici les prédictions de Llama :")
            st.write(df_labels2)
            st.write(f"Pourcentage de bonnes prédictions llama : {final_score_ollama:.2f}%")
          


if __name__ == "__main__":
    main()