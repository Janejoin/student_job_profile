import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import scipy.stats as stats

df_originale = pd.read_csv("../dataset/Student Placement.csv")
columns = ["Aptitute", "Problem Solving"]
df_selected = df_originale[columns]

df_selected['Aptitute_quartili'] = pd.qcut(df_selected['Aptitute'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
quartili_range = df_selected['Aptitute_quartili'].cat.categories

def analisi_dataframe(df):
    print(df.head(10).to_string())
    print(df.tail(10).to_string())
    df.info()
    print(df.describe().to_string())
    valori_unici = {"nome_colonna":df.columns,
                    "valori": [df[col].unique() for col in df.columns]}
    df_unici= pd.DataFrame(valori_unici)
    print(df_unici.to_string())

# Selezionare tutte le variabili di interesse
selected_vars_all = ["DSA", "DBMS", "OS", "CN", "Mathmetics", "Aptitute", "Comm", "Problem Solving", "Creative"]
correlation_selected_all = df_originale[selected_vars_all].corr()

# Visualizzare la heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_selected_all, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlazioni tra Competenze Tecniche e Soft Skills")
plt.show()

# Funzione per analizzare la correlazione tra aptitude e problem solving, usando i quartili
def correlazione_aptitude_quartili_problem_solving(df_selected):
    # Utilizzo dei quartili per la tabella di contingenza
    tabella = pd.crosstab(df_selected["Aptitute_quartili"], df_selected["Problem Solving"])
    print("TABELLA DI CONTINGENZA APTITUDE QUARTILI - PROBLEM SOLVING", tabella, sep="\n")
    chi2, p, dof, expected = stats.chi2_contingency(tabella)
    spearman_corr, p_value = stats.spearmanr(df_selected["Aptitute_quartili"].cat.codes, df_selected["Problem Solving"])
    print(
        f"p-value del Chi-Square: {p}, Correlazione di Spearman: {spearman_corr}, p-value della Correlazione di Spearman: {p_value}")

    # Creare i quartili con intervalli numerici reali
    quartili = pd.qcut(df_originale["Aptitute"], q=4, precision=1)

    # Estrarre solo l'estremo superiore di ogni intervallo
    df_selected["Aptitute_quartili"] = quartili.apply(lambda x: x.right)  # `.right` prende solo l'estremo superiore

    # Stampare i quartili unici per verifica
    print(df_selected["Aptitute_quartili"].unique())

    # Calcolo della media di Problem Solving per quartile di Aptitude
    media_problem_solving = df_selected.groupby("Aptitute_quartili")["Problem Solving"].mean().reset_index()

    # Converti i valori dei quartili in interi
    media_problem_solving["Aptitute_quartili"] = media_problem_solving["Aptitute_quartili"].astype(int)

    # Creazione del diagramma a barre
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="Aptitute_quartili",
        y="Problem Solving",
        data=media_problem_solving,
        palette="Set3"
    )
    plt.title("Distribuzione Punteggi Medi")
    plt.xlabel("Attitudine ad Apprendere")
    plt.ylabel("Problem Solving")
    plt.ylim(0, df_selected["Problem Solving"].max() * 1.1)  # Per dare più spazio visivo
    plt.show()


def correlazione_aptitude_mathmetics(df_mat):
    spearman_corr, p = spearmanr(df_mat["Aptitute"], df_mat["Mathmetics"])
    print(f"La correlazione di Spearman risultante tra Aptitude"
          f" e Mathematics è {spearman_corr}")
    print(f"il p-value sulla corelazione di Spearman Aptitude - Mathematics è  {p}")

def correlazione_aptitude_DSA(df_dsa):
    spearman_corr, p = spearmanr(df_dsa["Aptitute"], df_dsa["DSA"])
    print(f"La correlazione di Spearman risultante tra Aptitude"
          f" e DSA è {spearman_corr}")
    print(f"il p-value sulla corelazione di Spearman Aptitude - DSA è  {p}")

# invocazione funzioni
#analisi_dataframe(df_originale)
correlazione_aptitude_quartili_problem_solving(df_selected)
correlazione_aptitude_mathmetics(df_originale)
correlazione_aptitude_DSA(df_originale)