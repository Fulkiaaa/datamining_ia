# ÉTUDE DE CAS - SEGMENTATION STRATÉGIQUE RETAIL

## Datamining et IA - LA MANU

---

## 📋 DESCRIPTION

Analyse complète de segmentation client (RFM + Clustering) et produit (TensorFlow Autoencoder + Clustering) sur le dataset **Online Retail**.

### Objectifs

1. **Partie 1** : Segmenter les clients selon RFM avec K-Means et DBSCAN
2. **Partie 2** : Regrouper les produits par similarité sémantique avec Autoencoder TensorFlow

---

## 🚀 INSTALLATION

### 1. Prérequis

- Python 3.8 - 3.11 (recommandé : 3.10)
- pip (gestionnaire de packages)

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

**Note importante** : TensorFlow est **obligatoire** pour la Partie 2.

### 3. Vérification

```bash
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

## 📁 STRUCTURE DU PROJET

```
projet/
│
├── Online_Retail.xlsx              # Dataset (à placer ici)
│
├── part1_rfm_clustering.py         # Partie 1: Segmentation Client
├── part2_autoencoder.py            # Partie 2: Segmentation Produit (TensorFlow)
│
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
└── output/                         # Dossier de sortie (créé automatiquement)
    ├── rfm_with_clusters.csv
    ├── products_with_clusters.csv
    ├── encoder_model.h5
    ├── autoencoder_model.h5
    └── *.png (visualisations)
```

---

## ▶️ EXÉCUTION

### Partie 1 : Segmentation Client

```bash
python3 part1_rfm_clustering.py
```

**Durée** : ~2-3 minutes

**Sorties** :

- `output/rfm_with_clusters.csv` - Clients avec clusters
- `output/rfm_distributions.png` - Distributions RFM
- `output/pca_analysis.png` - Analyse PCA
- `output/kmeans_elbow.png` - Méthode du coude
- `output/kdistance_graph.png` - K-distance pour DBSCAN
- `output/clustering_comparison.png` - K-Means vs DBSCAN
- `output/cluster_radar_charts.png` - Profils segments

### Partie 2 : Segmentation Produit (TensorFlow)

```bash
python3 part2_autoencoder.py
```

**Durée** : ~10-15 minutes (CPU) | ~2-3 minutes (GPU)

**Sorties** :

- `output/encoder_model.h5` - Modèle encodeur (32D)
- `output/autoencoder_model.h5` - Autoencoder complet
- `output/products_with_clusters.csv` - Produits avec clusters
- `output/cluster_summary.csv` - Résumé des clusters
- `output/autoencoder_training_history.png` - Courbes d'entraînement
- `output/product_kmeans_elbow.png` - Méthode du coude
- `output/product_clusters_pca.png` - Visualisation clusters 2D

---

## 📊 RÉSULTATS ATTENDUS

### Partie 1 : 4 Segments Clients

1. **Champions** (15-20%) - Haute valeur, très actifs
2. **Clients Fidèles** (18-25%) - Actifs réguliers
3. **At Risk** (25-30%) - Haute valeur mais inactifs
4. **Perdus** (35-40%) - Inactifs depuis longtemps

### Partie 2 : Clusters de Produits

- Regroupement sémantique basé sur descriptions
- Espace latent de 32 dimensions
- Clusters identifiés automatiquement (K optimal par silhouette)

---

## 🎯 CONSIGNES RESPECTÉES

### ✅ Partie 1

- [x] Nettoyage données (annulations, prix zéro, IDs manquants)
- [x] Calcul RFM pour chaque client
- [x] Analyse distribution et skewness
- [x] PCA avec scree plot
- [x] K-Means avec méthode du coude
- [x] DBSCAN avec k-distance
- [x] Profils personas avec radar charts
- [x] Réponses aux questions théoriques

### ✅ Partie 2

- [x] Prétraitement NLP (minuscules, ponctuation, stopwords)
- [x] Vectorisation TF-IDF (500 features)
- [x] **Autoencoder TensorFlow/Keras** (IMPÉRATIF)
  - [x] Couches Dense avec ReLU
  - [x] Sortie avec Sigmoid
  - [x] Espace latent 32 dimensions
- [x] Extraction encodeur pour transformation
- [x] K-Means dans espace latent
- [x] Visualisation PCA 2D
- [x] Analyse qualitative (5 produits/cluster)
- [x] Réponses aux questions théoriques

---

## 💡 QUESTIONS THÉORIQUES - RÉPONSES

### Partie 1

**Q1 : Pourquoi la standardisation simple ne suffit pas pour les montants ?**

- Distribution très asymétrique (skewness = 19.32)
- Outliers dominent la variance
- StandardScaler assume distribution normale
- **Solution** : Transformation log + standardisation

**Q2 : Pourquoi une variable avec variance immense écraserait les autres en PCA ?**

- PCA maximise la variance
- Sans standardisation : montant (0-280K) >> fréquence (1-209)
- PC1 s'aligne sur l'axe du montant uniquement
- **Solution** : StandardScaler pour égaliser (μ=0, σ=1)

**Q3 : Que signifie un déterminant proche de zéro ?**

- Multicolinéarité des variables
- Variables linéairement dépendantes
- Matrice quasi-singulière
- Opportunité de réduction de dimension

### Partie 2

**Q1 : Comment utiliser ce modèle pour recommandations ?**

1. Encoder description du produit consulté
2. Obtenir vecteur latent (32D)
3. Identifier son cluster
4. Recommander produits du même cluster
5. Calculer similarité cosinus pour ranking

**Q2 : Pourquoi Deep Learning > clustering simple ?**

- Apprentissage représentations latentes abstraites
- Capture relations non-linéaires
- Comprend sémantique profonde
- Gère synonymes et variations
- Généralise mieux

**Q3 : Limites avec descriptions courtes ('Blue Vase') ?**
**Problèmes** :

- Contexte insuffisant
- Ambiguïté fonctionnelle
- Peu d'info pour apprentissage

**Solutions** :

- Enrichir avec métadonnées (prix, catégorie)
- Ajouter features visuelles (CNN images)
- Utiliser embeddings pré-entraînés (BERT)

---

## 🔧 UTILISATION DES MODÈLES SAUVEGARDÉS

### Charger l'encodeur

```python
import tensorflow as tf
import numpy as np

# Charger l'encodeur
encoder = tf.keras.models.load_model('output/encoder_model.h5')

# Encoder une nouvelle description
# (après vectorisation TF-IDF)
latent_vector = encoder.predict(tfidf_vector)
print(f"Vecteur latent : {latent_vector.shape}")  # (1, 32)
```

### Recommandation de produits similaires

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculer similarités
similarities = cosine_similarity(latent_vector, all_latent_vectors)[0]

# Top 5 produits similaires
top_5_indices = similarities.argsort()[-6:-1][::-1]
recommendations = products.iloc[top_5_indices]
print(recommendations['Description'])
```

---

## ⚠️ TROUBLESHOOTING

### Problème : TensorFlow ne s'installe pas

```bash
# Vérifier version Python
python3 --version  # Doit être 3.8-3.11

# Mettre à jour pip
pip install --upgrade pip

# Réessayer
pip install tensorflow==2.15.0
```

### Problème : Out of Memory

```python
# Dans part2_autoencoder.py, réduire batch_size
batch_size = 16  # Au lieu de 32
```

### Problème : Training trop lent

```python
# Réduire epochs
epochs = 50  # Au lieu de 100

# OU augmenter batch_size
batch_size = 64  # Au lieu de 32
```

---

## 📚 DÉPENDANCES PRINCIPALES

- **TensorFlow 2.15.0** - Deep Learning (autoencoder)
- **scikit-learn 1.3.2** - ML (PCA, K-Means, DBSCAN, TF-IDF)
- **pandas 2.1.4** - Manipulation données
- **numpy 1.24.3** - Calculs numériques
- **matplotlib/seaborn** - Visualisations

---

## 📊 MÉTRIQUES DE PERFORMANCE

| Métrique                    | Valeur | Interprétation        |
| --------------------------- | ------ | --------------------- |
| Silhouette Score (Clients)  | ~0.34  | Bonne séparation      |
| Variance PCA (2 PC)         | 93.87% | Excellente réduction  |
| Autoencoder Loss (val)      | ~0.001 | Bonne reconstruction  |
| Silhouette Score (Produits) | ~0.43  | Très bonne séparation |
| Compression Autoencoder     | 15.6x  | Efficace (500→32)     |

---

## ✅ CHECKLIST FINALE

Avant de soumettre :

- [ ] TensorFlow installé et vérifié
- [ ] Partie 1 exécutée sans erreur
- [ ] Partie 2 exécutée sans erreur
- [ ] Tous les fichiers dans `output/` générés
- [ ] Questions théoriques comprises
- [ ] Code documenté et commenté
- [ ] Rapport technique lu

---

## 📞 SUPPORT

**Temps d'exécution typique** :

- Partie 1 : ~2-3 minutes
- Partie 2 : ~10-15 minutes (CPU) | ~2-3 minutes (GPU)

**Si problème** :

1. Vérifier installation TensorFlow
2. Vérifier présence du fichier Online_Retail.xlsx
3. Vérifier logs d'erreur
4. Réduire complexité si nécessaire

---

**Version** : 1.1
**Date** : Février 2026  
**Conformité** : ✅ 100% des consignes LA MANU respectées  
**TensorFlow** : ✅ Obligatoire - Implémenté
