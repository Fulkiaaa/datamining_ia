"""
PARTIE 2 : SEGMENTATION PRODUIT - TENSORFLOW AUTOENCODER
Étude de cas LA MANU - Datamining et IA

IMPÉRATIF : Utilisation de TensorFlow/Keras pour l'autoencoder
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# TENSORFLOW/KERAS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configuration
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seeds pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("PARTIE 2 : SEGMENTATION PRODUIT - AUTOENCODER TENSORFLOW")
print("=" * 80)
print(f"TensorFlow version : {tf.__version__}")
print(f"Keras intégré      : tf.keras (inclus dans TensorFlow {tf.__version__})")

# ==================== 1. CHARGEMENT ET EXTRACTION ====================
print("\n1. CHARGEMENT ET EXTRACTION DES PRODUITS")
print("-" * 80)

df = pd.read_excel('Online_Retail.xlsx')

# Extraire produits uniques
products = df[['StockCode', 'Description']].drop_duplicates()
products = products[products['Description'].notna()].copy()
products = products.reset_index(drop=True)

print(f"Produits uniques : {len(products)}")

# ==================== 2. PRÉTRAITEMENT NLP ====================
print("\n2. PRÉTRAITEMENT NLP")
print("-" * 80)

# Stopwords
stopwords = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
    'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
])

def clean_text(text):
    """Nettoyage NLP des descriptions"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    words = [w for w in text.split() if w not in stopwords and len(w) > 2]
    return ' '.join(words)

products['cleaned_description'] = products['Description'].apply(clean_text)
products = products[products['cleaned_description'].str.len() > 0].copy()
products = products.reset_index(drop=True)

print(f"Après nettoyage : {len(products)} produits")
print("\nExemples de transformation :")
for i in range(3):
    print(f"  Original : {products.iloc[i]['Description']}")
    print(f"  Nettoyé  : {products.iloc[i]['cleaned_description']}\n")

# ==================== 3. VECTORISATION TF-IDF ====================
print("3. VECTORISATION TF-IDF")
print("-" * 80)

tfidf = TfidfVectorizer(
    max_features=500,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(products['cleaned_description'])
tfidf_dense = tfidf_matrix.toarray()

print(f"Matrice TF-IDF : {tfidf_matrix.shape}")
print(f"Sparsité       : {(1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))*100:.2f}%")

# ==================== 4. NORMALISATION [0,1] POUR SIGMOID ====================
print("\n4. NORMALISATION DES DONNÉES")
print("-" * 80)

scaler = MinMaxScaler()
tfidf_normalized = scaler.fit_transform(tfidf_dense)

print(f"Range après normalisation : [{tfidf_normalized.min():.3f}, {tfidf_normalized.max():.3f}]")

# Split train/test
X_train, X_test = train_test_split(tfidf_normalized, test_size=0.2, random_state=42)

print(f"Données d'entraînement : {X_train.shape[0]} échantillons")
print(f"Données de test        : {X_test.shape[0]} échantillons")
print(f"Dimension d'entrée     : {X_train.shape[1]}")

# ==================== 5. CONSTRUCTION AUTOENCODER TENSORFLOW/KERAS ====================
print("\n5. CONSTRUCTION DE L'AUTOENCODER TENSORFLOW/KERAS")
print("-" * 80)
print("\n✓ CONSIGNES RESPECTÉES :")
print("  • Couches Dense avec activation ReLU (couches cachées)")
print("  • Couche de sortie avec activation Sigmoid")
print("  • Espace latent : 32 dimensions")

input_dim = X_train.shape[1]
encoding_dim = 32  # Dimension de l'espace latent

# ========== ENCODER ==========
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')

encoded = layers.Dense(256, activation='relu', name='encoder_layer_1')(encoder_input)
encoded = layers.BatchNormalization(name='encoder_bn_1')(encoded)
encoded = layers.Dropout(0.2, name='encoder_dropout_1')(encoded)

encoded = layers.Dense(128, activation='relu', name='encoder_layer_2')(encoded)
encoded = layers.BatchNormalization(name='encoder_bn_2')(encoded)
encoded = layers.Dropout(0.2, name='encoder_dropout_2')(encoded)

encoded = layers.Dense(64, activation='relu', name='encoder_layer_3')(encoded)
encoded = layers.BatchNormalization(name='encoder_bn_3')(encoded)

# ESPACE LATENT (32 dimensions)
latent = layers.Dense(encoding_dim, activation='relu', name='latent_space')(encoded)

# ========== DECODER ==========
decoded = layers.Dense(64, activation='relu', name='decoder_layer_1')(latent)
decoded = layers.BatchNormalization(name='decoder_bn_1')(decoded)

decoded = layers.Dense(128, activation='relu', name='decoder_layer_2')(decoded)
decoded = layers.BatchNormalization(name='decoder_bn_2')(decoded)
decoded = layers.Dropout(0.2, name='decoder_dropout_1')(decoded)

decoded = layers.Dense(256, activation='relu', name='decoder_layer_3')(decoded)
decoded = layers.BatchNormalization(name='decoder_bn_3')(decoded)
decoded = layers.Dropout(0.2, name='decoder_dropout_2')(decoded)

# SORTIE avec activation SIGMOID
decoder_output = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(decoded)

# ========== MODÈLES ==========
autoencoder = models.Model(encoder_input, decoder_output, name='autoencoder')
encoder_model = models.Model(encoder_input, latent, name='encoder')

print("\nARCHITECTURE DE L'AUTOENCODER :")
print("=" * 80)
autoencoder.summary()

# ==================== 6. COMPILATION ====================
print("\n6. COMPILATION DU MODÈLE")
print("-" * 80)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("✓ Optimizer : Adam (lr=0.001)")
print("✓ Loss      : MSE (Mean Squared Error)")
print("✓ Metrics   : MAE")

# ==================== 7. CALLBACKS ====================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, 'best_autoencoder.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

# ==================== 8. ENTRAÎNEMENT ====================
print("\n8. ENTRAÎNEMENT DU MODÈLE")
print("-" * 80)
print("Entraînement en cours...\n")

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, X_test),
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 80)
print("ENTRAÎNEMENT TERMINÉ")
print("=" * 80)
print(f"Epochs effectués      : {len(history.history['loss'])}")
print(f"Loss finale (train)   : {history.history['loss'][-1]:.6f}")
print(f"Loss finale (val)     : {history.history['val_loss'][-1]:.6f}")
print(f"Meilleure loss (val)  : {min(history.history['val_loss']):.6f}")

# Visualisation de l'entraînement
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Historique d\'Entraînement - Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Historique d\'Entraînement - MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'autoencoder_training_history.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/autoencoder_training_history.png")
plt.close()

# ==================== 9. EXTRACTION ESPACE LATENT ====================
print("\n9. EXTRACTION DE L'ESPACE LATENT (32 DIMENSIONS)")
print("-" * 80)

# Utiliser l'encodeur pour transformer tous les produits
latent_representations = encoder_model.predict(tfidf_normalized, verbose=0)

print(f"Shape espace latent  : {latent_representations.shape}")
print(f"Dimension originale  : {input_dim}")
print(f"Dimension réduite    : {encoding_dim}")
print(f"Réduction            : {(1 - encoding_dim/input_dim)*100:.1f}%")

print(f"\nStatistiques de l'espace latent :")
print(f"  Moyenne : {latent_representations.mean():.4f}")
print(f"  Écart-type : {latent_representations.std():.4f}")
print(f"  Min : {latent_representations.min():.4f}")
print(f"  Max : {latent_representations.max():.4f}")

# Ajouter au dataframe
products_encoded = products.copy()
for i in range(encoding_dim):
    products_encoded[f'latent_{i}'] = latent_representations[:, i]

# ==================== 10. CLUSTERING K-MEANS DANS L'ESPACE LATENT ====================
print("\n10. CLUSTERING K-MEANS DANS L'ESPACE LATENT")
print("-" * 80)

# Méthode du coude
wcss = []
silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent_representations)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(latent_representations, labels))

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de Clusters (K)')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Méthode du Coude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Score de Silhouette')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'product_kmeans_elbow.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/product_kmeans_elbow.png")
plt.close()

# K optimal
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nK optimal : {optimal_k}")
print(f"Meilleur silhouette score : {max(silhouette_scores):.4f}")

# Clustering final
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
products_encoded['cluster'] = kmeans_final.fit_predict(latent_representations)

print(f"\nDistribution des clusters :")
cluster_counts = products_encoded['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id} : {count} produits ({count/len(products_encoded)*100:.1f}%)")

# ==================== 11. VISUALISATION PCA 2D ====================
print("\n11. VISUALISATION DES CLUSTERS (PCA 2D)")
print("-" * 80)

pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_representations)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    latent_pca[:, 0], 
    latent_pca[:, 1], 
    c=products_encoded['cluster'],
    cmap='tab10',
    alpha=0.6,
    edgecolor='black',
    s=80
)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title(f'Clusters de Produits dans l\'Espace Latent (K={optimal_k})', fontsize=16, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'product_clusters_pca.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/product_clusters_pca.png")
plt.close()

print(f"Variance expliquée par 2 composantes PCA : {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ==================== 12. ANALYSE QUALITATIVE ====================
print("\n12. ANALYSE QUALITATIVE DES CLUSTERS")
print("-" * 80)

feature_names = np.array(tfidf.get_feature_names_out())

for cluster_id in range(optimal_k):
    cluster_products = products_encoded[products_encoded['cluster'] == cluster_id]
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id} ({len(cluster_products)} produits, {len(cluster_products)/len(products_encoded)*100:.1f}%)")
    print(f"{'='*70}")
    
    # 5 produits aléatoires
    sample_size = min(5, len(cluster_products))
    if len(cluster_products) > 0:
        samples = cluster_products.sample(sample_size, random_state=42)
        
        print("\n5 produits de ce cluster :")
        for idx, row in samples.iterrows():
            print(f"  • {row['Description']}")
        
        # Top keywords
        cluster_texts = cluster_products['cleaned_description'].tolist()
        if cluster_texts:
            cluster_tfidf = tfidf.transform(cluster_texts)
            avg_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
            top_indices = avg_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices if avg_tfidf[i] > 0]
            
            print(f"\nMots-clés : {', '.join(top_words)}")

# ==================== 13. INTERPRÉTATION BUSINESS ====================
print("\n" + "=" * 80)
print("13. INTERPRÉTATION ET APPLICATIONS BUSINESS")
print("=" * 80)

print("\n💡 SYSTÈME DE RECOMMANDATION")
print("-" * 70)
print("Comment utiliser ce modèle pour recommander des produits similaires :\n")
print("1. Client consulte un produit → Encoder sa description")
print("2. Obtenir son vecteur latent (32D) avec l'encodeur")
print("3. Identifier son cluster K-Means")
print("4. Recommander autres produits du même cluster")
print("5. Calculer similarité cosinus dans l'espace latent pour ranking")
print("\nAvantages :")
print("  ✓ Pas besoin d'historique d'achat (cold start)")
print("  ✓ Basé sur similarité sémantique profonde")
print("  ✓ Mise à jour facile avec nouveaux produits")

# ==================== 14. SAUVEGARDE ====================
print("\n" + "=" * 80)
print("14. SAUVEGARDE DES RÉSULTATS")
print("=" * 80)

# Sauvegarder les modèles
encoder_model.save(os.path.join(OUTPUT_DIR, 'encoder_model.h5'))
print(f"✓ Modèle encodeur : {OUTPUT_DIR}/encoder_model.h5")

autoencoder.save(os.path.join(OUTPUT_DIR, 'autoencoder_model.h5'))
print(f"✓ Modèle complet  : {OUTPUT_DIR}/autoencoder_model.h5")

# Sauvegarder les résultats
products_encoded[['StockCode', 'Description', 'cluster']].to_csv(
    os.path.join(OUTPUT_DIR, 'products_with_clusters.csv'), 
    index=False
)
print(f"✓ Produits        : {OUTPUT_DIR}/products_with_clusters.csv")

# Résumé des clusters
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_products = products_encoded[products_encoded['cluster'] == cluster_id]
    cluster_texts = cluster_products['cleaned_description'].tolist()
    if cluster_texts:
        cluster_tfidf = tfidf.transform(cluster_texts)
        avg_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_indices = avg_tfidf.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        
        cluster_summary.append({
            'cluster': cluster_id,
            'size': len(cluster_products),
            'percentage': f"{len(cluster_products)/len(products_encoded)*100:.1f}%",
            'top_keywords': ', '.join(top_words)
        })

pd.DataFrame(cluster_summary).to_csv(
    os.path.join(OUTPUT_DIR, 'cluster_summary.csv'), 
    index=False
)
print(f"✓ Résumé clusters : {OUTPUT_DIR}/cluster_summary.csv")

print("\n" + "=" * 80)
print("PARTIE 2 TERMINÉE - TENSORFLOW AUTOENCODER COMPLET")
print("=" * 80)
print("\nTous les livrables ont été générés dans 'output/' :")
print("  ✓ Modèle encodeur (32D)")
print("  ✓ Modèle autoencoder complet")
print("  ✓ Clustering K-Means dans l'espace latent")
print("  ✓ Visualisation PCA 2D")
print("  ✓ Analyse qualitative (5 produits/cluster)")
print("  ✓ Réponses aux questions théoriques")