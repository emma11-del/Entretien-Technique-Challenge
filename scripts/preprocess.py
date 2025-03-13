import pandas as pd
import re
import spacy
from tqdm import tqdm
import openai


def load_csv(df):

    df = pd.read_csv("data/intent-detection-train.csv")  # Charger le CSV
    df_labels1 = df.copy()  # Je garde une copie de la colonne label pour pouvoir la comparer ensuite avec mes prédictions de Open AI
    df_labels2 = df.copy()  # Je garde une copie de la colonne label pour pouvoir la comparer ensuite avec mes prédictions de Llama

    return df, df_labels1, df_labels2

# Preprocessing de texte que je vais utiliser pour mes modèles de classification :

def clean_text(text):
    text = text.lower()  # Met en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprime la ponctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Supprime les espaces inutiles
    return text


nlp = spacy.load("fr_core_news_sm")  # Modèle français

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


# Fonction utile seulement pour mes modèles de classification car pas assez de données
# Appel à l'API d'Open AI pour créer de la data pour les catégories minoritaires

def data_augmentation(text,api_key):
    
    prompt = f"""
    Tu es un expert en reformulation de phrases pour la data augmentation.
    Ta mission est de générer une phrase alternative en conservant le sens original, mais en variant la structure, les mots et le ton si possible.
    
     **Règles :**
    - Conserve le même sens et la même intention que la phrase originale.
    - Utilise des synonymes et des variations grammaticales.
    - Change la construction de la phrase si pertinent.
    - Évite d’introduire de nouvelles informations ou de changer le contexte.

     **Exemple :**
    ➡️ Phrase originale : "Comment puis-je réserver un hôtel à Paris ?"
     Reformulation 1 : "Pouvez-vous m'indiquer la procédure pour réserver un hôtel à Paris ?"

     **Voici la phrase à reformuler :**
    \"{text}\"
    
    Donne-moi une seule reformulation et rien d'autre.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            api_key=api_key
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erreur API : {e}")
        return None

# Application de notre data augmentation sur notre dataframe
def apply_data_augmentation(df, category_counts, target_size,api_key):
    
    openai_api_key = api_key
    new_rows = []
    
    # Boucle sur chaque catégorie pour générer des exemples supplémentaires
    for category, count in tqdm(category_counts.items(), desc="Data Augmentation via Open AI"):
        if category == "out_of_scope":
            continue  # On ne touche pas à cette classe car déjà beaucoup de données

        # Sélectionner les phrases existantes de cette catégorie
        category_rows = df[df["label"] == category]

        while count < target_size:  # Générer des phrases jusqu'à atteindre la taille max
            for _, row in category_rows.iterrows():
                new_text = data_augmentation(row["text"],openai_api_key)  # Génération de données via Open AI
                if new_text:
                    new_rows.append({"text": new_text, "label": category})
                    count += 1
                    if count >= target_size:
                        break  # Stop si on a atteint la taille max

    # Ajouter les nouvelles phrases au dataframe
    df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Sauvegarder le dataset augmenté
    df_augmented.to_csv("data/data-augmented.csv", index=False)

    # Phrase de vérification que tout à bien marché
    print("La data pour le modèle Random Forest a bien était augmentée et sauvegardée !")

    return df_augmented