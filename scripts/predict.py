import openai
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import subprocess
from preprocess import clean_text, lemmatize_text 
from nltk.corpus import stopwords
import nltk
from sklearn.pipeline import Pipeline




# Modèles 1 : Classification -> Random forest et Logistic Regression

def train_model(df):

    X = df["text"]  
    y = df["label"]  # Variable cible = catégories

    # Appel aux fonctions de preprocessing pour transformer nos textes = X
    X = X.apply(clean_text)
    X = X.apply(lemmatize_text)


    # Division en train (80%)/test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    nltk.download("stopwords") # Télechargement de la liste des stopwords de NLTK
    french_word = stopwords.words("french") # Récupération de la liste des stopwords en français
    vectorizer = TfidfVectorizer(stop_words=french_word) # Création du vectoriseur TF-IDF qui va appliquer la liste des mots à supprimer pendant la vectorisation

    # Pipeline : Vectorisation + Modèle
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('model', RandomForestClassifier())  # Placeholder
    ])
    
    # Grille des hyperparamètres à tester 
    # Objectif : éviter l'overfitting et améliorer la performance du modèle
    param_grid = [
        {
            'model': [RandomForestClassifier()],
            'model__n_estimators': [50, 100, 200], # Nombre d'arbres
            'model__max_depth': [None, 10, 20] # Profondeur des arbres
        },
        {
            'model': [LogisticRegression(max_iter=500)],
            'model__C': [0.1, 1, 10],  # Coefficient de la régularisation
            'model__solver': ['liblinear', 'lbfgs'] # Ajuster les coefficients du modèle : liblinear : pour les petites dataset
        }
    ]

     
    # GridSearchCV pour optimiser la pipeline : 
    # - Teste différentes combinaisons d'hyperparamètres
    # - Effectue une cross validation  pour évaluer chaque combinaison
    # - Retient la meilleure configuration trouvée
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1) 
    
    # Entraînement du GridSearchCV sur les données d'entraînement 
    grid_search.fit(X_train, y_train)

    # Récupération du meilleur modèle trouvé par GridSearchCV
    best_model = grid_search.best_estimator_

    # Prédiction sur les données de test avec le meileur modèle
    y_pred = best_model.predict(X_test)
    
    # Récuperation le nom du modèle sélectionné
    best_model_name = type(best_model.named_steps['model']).__name__


    return X_test,y_test, y_pred,best_model_name





# Modèle 2 : Open AI

def predict_category1(text,api_key):
    openai.api_key = api_key # Définition de la clé API OpenAI pour avoir accès aux services OpenAI

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            # Ecriture du meilleur prompt possible, avec le plus de précision
            messages=[
                {"role": "system", "content": "Tu es un expert pour classifier l'intention exprimée par une phrase dans le cadre d'une agence de tourisme. Tu as pour rôle de catégorier une phrase qui peut appartenir à 9 catégories : "
                "translate: l’utilisateur souhaite traduire une phrase dans une autre langue,"
                "travel_alert: l’utilisateur demande si sa destination est concernée par une alerte de voyage, "
                "flight_status: l’utilisateur demande des informations sur le statut de son vol,"
                "lost_luggage: l’utilisateur signale la perte de ses bagages, "
                "travel_suggestion: l’utilisateur souhaite une recommandation de voyage,"
                "carry_on: l'utilisateur souhaite des informations sur les bagages à main,"
                "book_hotel: l’utilisateur souhaite réserver un hôtel,"
                "book_flight: l’utilisateur souhaite réserver un vol."
                "Il est possible que l’utilisateur formule une demande dite “hors-scope” c’est-à-dire qui n’appartient à aucune des 8 classes ci-dessus,"
                "Pour la catégorie translate par exemple, nous pouvons avoir une phrase du type : Pouvez-vous me dire comment dire «je ne parle pas beaucoup espagnol», en espagnol"
                "Pour la catégorie travel_alert par exemple, nous pouvons avoir une phrase du type : Y a-t-il des alertes de voyage pour la Syrie"
                "Pour la catégorie flight_status par exemple, nous pouvons avoir une phrase du type :Quand allons-nous commencer à monter à bord de mon vol prévu"
                "Pour la catégorie lost_luggage par exemple, nous pouvons avoir une phrase du type : Localiser les bagages perdus de Flight America Airlines à O'Hare" 
                "Pour la catégorie travel_suggestion par exemple, nous pouvons avoir une phrase du type :Donnez-moi une liste de choses à faire à Orlando"
                "Pour la catégorie carry_on par exemple, nous pouvons avoir une phrase du type :Y a-t-il une limite de poids de bagage cabine"
                "Pour la catégorie book_hotel par exemple, nous pouvons avoir une phrase du type :Veuillez me trouver une chambre d'hôtel pour 4 personnes du lundi au mercredi à Lakewood"
                "Si la catégorie identifiée est lost_luggage, sois absolument certain avant de la choisir. Si un doute existe, oriente plutôt vers une autre catégorie pertinente."
                "Ne classe pas une question en lost_luggage si elle concerne un autre problème, même proche."        
                "Pour la catégorie out_of_scope par exemple, nous pouvons avoir une phrase du type :Combien coûte des frais de découvert pour la banque"
                "dans ce cas,tu dois renvoyer : out_of_scope - Désolé, je ne peux pas répondre à cette question, elle ne fait pas partie de mon domaine de compétence."                
                "Sinon, renvoie uniquement le nom de la catégorie."},
                {"role": "user", "content": f"Quelle est la catégorie de ce texte ? {text}"}
            ],
            temperature=0 # Choix de =0 pour une classification fiable et cohérente sans variation aléatoire
        )
        return response["choices"][0]["message"]["content"].strip() # Renvoie la catégorie prédite par Open AI
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None
    


# Modèle 3 : Llama 3

def predict_category2(text):
    # Ecriture du meilleur prompt possible, avec le plus de précision
    prompt = f"""
    Tu es un expert pour classifier l'intention exprimée par une phrase dans le cadre d'une agence de tourisme.
    Tu as pour rôle de catégoriser une phrase parmi les 9 catégories suivantes :
    - translate: l’utilisateur souhaite traduire une phrase dans une autre langue
    - travel_alert: l’utilisateur demande si sa destination est concernée par une alerte de voyage
    - flight_status: l’utilisateur demande des informations sur le statut de son vol
    - lost_luggage: l’utilisateur signale la perte de ses bagages
    - travel_suggestion: l’utilisateur souhaite une recommandation de voyage
    - carry_on: l'utilisateur souhaite des informations sur les bagages à main
    - book_hotel: l’utilisateur souhaite réserver un hôtel
    - book_flight: l’utilisateur souhaite réserver un vol
    - Il est possible que l’utilisateur formule une demande dite “out_of_scope” c’est-à-dire qui n’appartient à aucune des 8 classes ci-dessus,
    dans ce cas, on attend que l’algorithme renvoie une réponse spéciale dans une catégorie : out_of_scope."

    Exemples :
    - "Pouvez-vous me dire comment dire «je ne parle pas beaucoup espagnol» en espagnol ?" → translate
    - "Y a-t-il des alertes de voyage pour la Syrie ?" → travel_alert
    - "Quand allons-nous commencer à monter à bord de mon vol prévu ?" → flight_status
    - "Localiser les bagages perdus de Flight America Airlines à O'Hare." → lost_luggage
    - "Donnez-moi une liste de choses à faire à Orlando." → travel_suggestion
    - "Y a-t-il une limite de poids de bagage cabine ?" → carry_on
    - "Veuillez me trouver une chambre d'hôtel pour 4 personnes du lundi au mercredi à Lakewood." → book_hotel
    - "Combien coûtent les frais de découvert pour la banque ?" → out_of_scope
    "Si la catégorie identifiée est lost_luggage, sois absolument certain avant de la choisir. Si un doute existe, oriente plutôt vers une autre catégorie pertinente."
    "Ne classe pas une question en lost_luggage si elle concerne un autre problème, même proche."
    "Si la catégorie identifiée est 'out_of_scope', tu dois fournir une réponse spécifique. Par exemple : 'Désolé, je ne peux pas répondre à cette question, elle ne fait pas partie de mon domaine de compétence.'" 
     Autrement, je veux que tu me renvoies UNIQUEMENT le nom de la catégorie, sans texte additionnel (si ce n'est pas out_of_scope).

    Voici la phrase à catégoriser :
    {text}
    """

    # Appel à Ollama 
    result = subprocess.run(
        ["ollama", "run", "llama3"],  
        input=prompt,
        capture_output=True,
        text=True
    )

    # Récupération de la réponse
    response = result.stdout.strip()
    
    return response  # Renvoie directement la catégorie prédite par Llama



