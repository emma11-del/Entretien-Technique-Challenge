import os
from tqdm import tqdm
from preprocess import load_csv, apply_data_augmentation
from predict import predict_category1, predict_category2,train_model
from evaluate import  evaluate_predictions1, evaluate_predictions2,plot_evaluation_classification1
from collections import Counter
import pandas as pd
from dotenv import load_dotenv
import openai

# Charger les variables d'environnement depuis "api.env"
load_dotenv("api.env")

# Récupérer la clé API
openai_api_key = os.getenv("openai_key")

def main():
    
    input_csv = "data/intent-detection-train.csv"
    output_csv = "output/predictions.csv"

    openai.api_key = openai_api_key # Assigner la clé API directement

    # 1. Modèle de classification :

    # Vérifier et créer le dossier de sortie : output
    if not os.path.exists(output_csv):
        os.makedirs(output_dir)

    # Définition du chemin du fichier de données augmentées
    data_augmented_path = "data/data-augmented.csv"

    if os.path.exists(data_augmented_path):
    # Si le fichier existe déjà, on le charge directement
        print("Le fichier data-augmented.csv existe déjà. Chargement des données !!")
        df_NLP = pd.read_csv(data_augmented_path)
        
    else: # Sinon, on lance la Data Augmentation
        print("Le fichier data-augmented.csv n'existe pas. Lancement de la data augmentation ...")

        df = pd.read_csv("data/original-dataset.csv") 

        # Déterminer la taille cible pour équilibrer les catégories
        category_counts = Counter(df["label"])
        target_size = max(category_counts.values())  # Nombre max d'échantillons par catégorie (celui d'out_of_scope)

        # Appliquer la Data Augmentation
        df_NLP = apply_data_augmentation(df, category_counts, target_size, openai_api_key)

        print(" Data augmentation terminée et sauvegardée.")


    # Récupération des scores du meilleur modèle parmis Random Forest et Logistic Regression depuisnotre dataset augmentée
    y_test, y_pred, best_model_name = train_model(df_NLP)

    print(f"\nLe modèle sélectionné est {best_model_name}")

    # Afficher la Matrice de Confusion
    plot_evaluation_classification1(y_test, y_pred, best_model_name)
    
    # 2. Modèle de LLM

    df, df_labels1, df_labels2 = load_csv(input_csv)
    df.drop(columns=["label"], inplace=True)  # Suppression de la colonne label

    # Ajout des colonnes de prédictions vides
    df["cat_prediction_openai"] = None  
    df["cat_prediction_ollama"] = None 

    # Prédictions avec OpenAI

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions OpenAI"):
        df.at[index, "cat_prediction_openai"] = predict_category1(row["text"], openai_api_key)

    final_score_openai = evaluate_predictions1(df, df_labels1)

    # Ajouter les prédictions OpenAI au dataframe contenant les labels
    df_labels1["cat_prediction"] = df["cat_prediction_openai"]

    output_dir = "output_csv"
    # On vérifie si le fichier existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_labels1.to_csv(os.path.join(output_dir, "predictions_openai.csv"), index=False)

    print(f"Pourcentage de réussite des prédictions OpenAI : {final_score_openai:.2f}%")

    # Prédictions avec Llama
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions Ollama"):
        df.at[index, "cat_prediction_ollama"] = predict_category2(row["text"])

    final_score_ollama = evaluate_predictions2(df, df_labels2)

    # Ajouter les prédictions Llama au dataframe contenant les labels
    df_labels2["cat_prediction"] = df["cat_prediction_ollama"]
    df_labels2.to_csv(os.path.join(output_dir, "predictions_ollama.csv"), index=False)

    print(f"Pourcentage de réussite des prédictions Ollama : {final_score_ollama:.2f}%")

   
if __name__ == "__main__":
    main()
