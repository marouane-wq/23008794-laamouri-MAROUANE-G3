# ============================================================
# Régimes de marché du S&P 500 (k-means / GMM) vs récessions USREC
# Script unique regroupant les étapes 1 à 8 (à exécuter dans Google Colab)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from google.colab import files
from IPython.display import display


# ============================================================
# Étape 1 : Télécharger les données
# ============================================================

# Sélectionne SP500.csv ET USREC.csv
uploaded = files.upload()
f_sp  = [f for f in uploaded if "SP500" in f.upper()][0]
f_rec = [f for f in uploaded if "USREC" in f.upper()][0]

# S&P 500 : un prix par jour (les "." des jours fériés deviennent des cases vides, puis supprimées)
sp = pd.read_csv(f_sp)
sp.columns = ["date", "prix"]
sp["date"] = pd.to_datetime(sp["date"])
sp["prix"] = pd.to_numeric(sp["prix"], errors="coerce")
sp = sp.dropna().set_index("date").sort_index()
prix = sp["prix"]

# USREC : un chiffre par mois (1 = récession)
rec = pd.read_csv(f_rec)
rec.columns = ["date", "usrec"]
rec["date"] = pd.to_datetime(rec["date"])
rec = rec.set_index("date").sort_index()

print("S&P 500 :", len(prix), "jours,", prix.index.min().date(), "→", prix.index.max().date())
print("Mois en récession dans la période :")
print(rec.loc[prix.index.min():prix.index.max()].query("usrec == 1").index.strftime("%Y-%m").tolist())

plt.figure(figsize=(12, 3))
prix.plot(color="grey", title="S&P 500 (vérification)")
plt.show()


# ============================================================
# Étape 2 : Volatilité et rendement sur 20 jours
# ============================================================

FENETRE_FEATURES = 20

rendement_jour = prix.pct_change()   # (prix aujourd'hui - hier) / hier

features = pd.DataFrame({
    "volatilite": rendement_jour.rolling(FENETRE_FEATURES).std(),   # agitation du dernier mois
    "rendement":  rendement_jour.rolling(FENETRE_FEATURES).mean(),  # tendance du dernier mois
}).dropna()   # les 20 premiers jours n'ont pas assez d'historique

print("Lignes après étape 2 :", len(features))
display(features.describe())


# ============================================================
# Étape 3 : Standardiser avec les 252 jours précédents
# ============================================================

FENETRE_NORMALE = 252

# La "normale" du jour J = les 252 jours AVANT J.
# shift(1) : J n'est jamais comparé à lui-même ni au futur (pas de fuite d'information).
moy_passee = features.rolling(FENETRE_NORMALE).mean().shift(1)
std_passee = features.rolling(FENETRE_NORMALE).std().shift(1)

Z = ((features - moy_passee) / std_passee).dropna()   # valeurs standardisées
X = Z.values
features_ok = features.loc[Z.index]                   # valeurs d'origine alignées (plus lisibles)

print("Lignes après étape 3 :", len(Z))
print(Z.describe().loc[["mean", "std", "min", "max"]].round(2))


# ============================================================
# Étape 4 : K-means avec 2, 3, 4, 5 groupes + silhouette
# ============================================================

resultats_sil = {}
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    resultats_sil[k] = silhouette_score(X, km.labels_)

print("Scores de silhouette (k-means) :")
for k, s in resultats_sil.items():
    print(f"  k = {k} : {s:.3f}")

k_silhouette = max(resultats_sil, key=resultats_sil.get)
print(f"\n→ Meilleur k selon la silhouette : {k_silhouette}")

plt.figure(figsize=(5, 3))
plt.plot(list(resultats_sil.keys()), list(resultats_sil.values()), marker="o")
plt.xlabel("Nombre de groupes k")
plt.ylabel("Silhouette")
plt.title("Choix de k")
plt.xticks([2, 3, 4, 5])
plt.grid(alpha=.3)
plt.show()

# ===== TON CHOIX =====
# La fiche parle de 3 régimes. Garde 3 en justifiant dans le rapport,
# ou remplace par : K = k_silhouette
K = 3
print(f"K retenu pour la suite : {K}")


# ============================================================
# Étape 5 : Nommer les régimes
# ============================================================

NOMS = {
    2: ["Calme", "Tempête"],
    3: ["Calme", "Intermédiaire", "Tempête"],
    4: ["Calme", "Intermédiaire bas", "Intermédiaire haut", "Tempête"],
    5: ["Calme", "Modéré", "Intermédiaire", "Agité", "Tempête"],
}
COULEURS = {"Calme": "green", "Modéré": "yellowgreen", "Intermédiaire bas": "gold",
            "Intermédiaire": "orange", "Intermédiaire haut": "darkorange",
            "Agité": "orangered", "Tempête": "red"}
ORDRE = NOMS[K]

def nommer(labels):
    """Classe les groupes du moins au plus volatil (valeurs d'origine) et leur donne un nom."""
    vol_moy = features_ok["volatilite"].groupby(labels).mean().sort_values()
    mapping = {cluster: nom for cluster, nom in zip(vol_moy.index, ORDRE)}
    return pd.Series(labels, index=features_ok.index).map(mapping)

# K-means final
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
reg_km = nommer(kmeans.labels_)

tab5 = features_ok.assign(regime=reg_km).groupby("regime").agg(
    volatilite=("volatilite", "mean"),
    rendement=("rendement", "mean"),
    jours=("volatilite", "size"),
).loc[ORDRE]

print("Régimes k-means :")
display(tab5.style.format({"volatilite": "{:.4f}", "rendement": "{:.5f}"}))


# ============================================================
# Étape 6 : GMM et comparaison avec k-means (Tableau 1)
# ============================================================

gmm = GaussianMixture(n_components=K, covariance_type="full", n_init=5, random_state=42).fit(X)
labels_gmm = gmm.predict(X)
reg_gmm = nommer(labels_gmm)
certitude = gmm.predict_proba(X).max(axis=1).mean()

# --- Tableau 1 ---
jours = pd.DataFrame({"K-means": reg_km.value_counts(), "GMM": reg_gmm.value_counts()}).loc[ORDRE]

tableau1 = pd.DataFrame({
    "Silhouette": [silhouette_score(X, kmeans.labels_), silhouette_score(X, labels_gmm)],
}, index=["K-means", "GMM"]).round(3)
for nom in ORDRE:
    tableau1[f"Jours {nom}"] = [jours.loc[nom, "K-means"], jours.loc[nom, "GMM"]]

print("===== TABLEAU 1 : k-means contre GMM =====")
display(tableau1)

print(f"Accord k-means / GMM (adjusted Rand) : {adjusted_rand_score(reg_km, reg_gmm):.3f}")
print(f"Certitude moyenne du GMM : {certitude:.1%}")

print("\nTableau croisé (lignes = k-means, colonnes = GMM) :")
display(pd.crosstab(reg_km, reg_gmm).reindex(index=ORDRE, columns=ORDRE, fill_value=0))


# ============================================================
# Étape 7 : Comparer avec les récessions (Tableau 2)
# ============================================================

# Chaque jour reçoit la valeur USREC de son mois
rec_mensuel = rec["usrec"].copy()
rec_mensuel.index = rec_mensuel.index.to_period("M")
mois_jour = features_ok.index.to_period("M")
recession_jour = pd.Series(rec_mensuel.reindex(mois_jour).values, index=features_ok.index)

df = pd.DataFrame({"regime": reg_km, "recession": recession_jour}).dropna()
df["recession"] = df["recession"].astype(int)

# --- Tableau 2 : régimes × récessions ---
t2_n = pd.crosstab(df["regime"], df["recession"]).reindex(index=ORDRE, columns=[0, 1], fill_value=0)
t2_n.columns = ["Pas de récession (0)", "Récession (1)"]
t2_pct = (t2_n / t2_n.sum() * 100).round(1)   # pourcentages par colonne

print("===== TABLEAU 2 : nombre de jours =====")
display(t2_n)
print("Pourcentages par colonne :")
display(t2_pct)

# --- Jours Tempête par année ---
tempete = df[df["regime"] == "Tempête"]
tempete_par_an = tempete.groupby(tempete.index.year).size()
print("Jours classés 'Tempête' par année :")
display(tempete_par_an.to_frame("jours Tempête"))


# ============================================================
# Étape 8 : Graphique final
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Courbe du S&P 500, fine et grise
p = prix.loc[features_ok.index]
ax.plot(p.index, p.values, color="grey", linewidth=0.7, zorder=1)

# Un point par jour, coloré selon le régime
for nom in ORDRE:
    masque = (reg_km == nom).values
    ax.scatter(p.index[masque], p.values[masque], s=6, color=COULEURS[nom], label=nom, zorder=2)

# Bandes grises : mois de récession
debut, fin = p.index.min(), p.index.max()
premiere = True
for date, val in rec["usrec"].items():
    if val == 1:
        d0, d1 = date, date + pd.offsets.MonthEnd(0)
        if d1 >= debut and d0 <= fin:
            ax.axvspan(d0, d1, color="grey", alpha=0.3, zorder=0,
                       label="Récession (USREC)" if premiere else None)
            premiere = False

ax.set_title("S&P 500 coloré par régime de marché (k-means), récessions en gris", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Indice S&P 500")
ax.legend(loc="upper left", markerscale=3)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("regimes_sp500.png", dpi=150)
plt.show()

files.download("regimes_sp500.png")
