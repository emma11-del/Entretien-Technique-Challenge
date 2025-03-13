from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
import streamlit as st

# Evaluation de la prédiction d'Open AI en comparant nos prédictions aux valeurs réelles

def evaluate_predictions1(df, df_labels1):

    df["label"] = df_labels1["label"] # On ajoute les labels réels pour permettre la comparaison avec les prédictions

    df["correct_prediction"] = df["cat_prediction_openai"] == df["label"] # Comparer les prédictions avec les labels réels

    accuracy_openai = df["correct_prediction"].mean()  # Calcul le taux de précision global (accuracy)

    # Évaluation de la précision sur la classe "lost_luggage"
    # -> Un FP est coûteux : on ne doit pas signaler une perte de bagage à tort car le coût téléphonique est élevé
    precision_openai = precision_score(
        df["label"], df["cat_prediction_openai"], 
        labels=["lost_luggage"], average="macro", zero_division=1
    )

    # Évaluation du recall sur la classe "out_of_scope"
    # -> Un FN est problématique : on doit bien identifier les demandes hors-sujet 
    recall_openai = recall_score(
        df["label"], df["cat_prediction_openai"], 
        labels=["out_of_scope"], average="macro", zero_division=1
    )

    # Score final pondéré
    final_score_openai = 0.7 * accuracy_openai + 0.2 * precision_openai + 0.1 * recall_openai

    return final_score_openai*100

# Evaluation de la prédiction d'Ollama en comparant nos prédictions aux valeurs réelles

def evaluate_predictions2(df, df_labels2):

    df["label"] = df_labels2["label"] 

    df["correct_ollama"] = df["cat_prediction_ollama"] == df["label"]

    accuracy_ollama = df["correct_ollama"].mean() 
    
    precision_ollama = precision_score(
        df["label"], df["cat_prediction_ollama"], 
        labels=["lost_luggage"], average="macro", zero_division=1
    )

    recall_ollama = recall_score(
        df["label"], df["cat_prediction_ollama"], 
        labels=["out_of_scope"], average="macro", zero_division=1
    )

    final_score_ollama = 0.7 * accuracy_ollama + 0.2 * precision_ollama + 0.1 * recall_ollama

    return  final_score_ollama*100

# Modèles de classification :
# 1. Pour compilation via le terminal : Évaluation des performances de modèles de classification (Random Forest et Logistic Regression) et afficher les résultats

def plot_evaluation_classification1(y_test, y_pred,model_name):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # Calcul du score global pondéré :
    final_score = (0.7 * accuracy + 0.2 * precision + 0.1 * recall)*100
    
    # Affichage des résultats
    print(f"\nLe score de prédiction pour {model_name} :{final_score}%")
    print(f"Rapport de classification pour {model_name} :\n")
    print(classification_report(y_test, y_pred, digits=3))

    
# 2. Pour compilation via le Streamlit : Évaluation des performances de modèles de classification (Random Forest et Logistic Regression) et afficher les résultats

def plot_evaluate_classification2(text,y_test, y_pred, model_name):
   
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    final_score = (0.7 * accuracy + 0.2 * precision + 0.1 * recall)*100

    report = classification_report(y_test, y_pred, output_dict=True) # Extraction des métriques de classification sous forme de dictionnaire

    df_report = pd.DataFrame(report).transpose() # Transformation du dictionnaire en DataFrame

    st.write(f"Le score de prédiction pour {model_name} : {final_score} %")
    st.write(f"Rapport de classification pour **{model_name}** :")
    st.dataframe(df_report)  # Affiche un tableau interactif

    # Détection des cas out_of_scope
    df_results = pd.DataFrame({
        "texte" :text,
        "label": y_test,
        "prediction": y_pred
    })

    df_results["info"] = None

    # Pour chaque prédiction 'out_of_scope', on ajoute un texte dans la colonne
    df_results.loc[df_results["prediction"] == "out_of_scope", "info"] = (
        "Hors-scope détecté ! Désolé, je ne peux pas répondre à cette question, "
        "elle ne fait pas partie de mon domaine de compétence."
    )

    return df_results