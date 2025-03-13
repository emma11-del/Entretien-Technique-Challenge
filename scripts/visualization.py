import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


def plot_intentions(df):
    plt.figure(figsize=(8, 5))
    df["label"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Répartition des intentions")
    plt.xlabel("Intentions")
    plt.ylabel("Nombre de textes")
    plt.xticks(rotation=45)
    st.pyplot(plt)  


def chie_pie_words_count(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, label in enumerate(df["label"].unique()):
        text = " ".join(df[df["label"] == label]["text"])
        word_counts = Counter(text.split())
        top_words = dict(word_counts.most_common(5))

        axes[i].pie(top_words.values(), labels=top_words.keys(), autopct='%1.1f%%', startangle=140)
        axes[i].set_title(f"Top 5 mots - {label}")

    st.pyplot(fig)  


def boxplot_cat(df):
    plt.figure(figsize=(10, 6))
    df["text_length"] = df["text"].apply(len)
    sns.boxplot(x="label", y="text_length", data=df, hue="label", palette="bright")
    plt.title("Distribution des longueurs des verbatims par catégorie")
    plt.xlabel("Catégorie d'intention")
    plt.ylabel("Longueur du texte")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(plt)
