"""
PARTIE 1 : SEGMENTATION CLIENT - RFM + CLUSTERING
Étude de cas LA MANU - Datamining et IA
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration du dossier de sortie
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("PARTIE 1 : SEGMENTATION CLIENT - ANALYSE RFM")
print("=" * 80)

# ==================== 1. CHARGEMENT ET NETTOYAGE DES DONNÉES ====================
print("\n1. CHARGEMENT ET NETTOYAGE DES DONNÉES")
print("-" * 80)

# Charger les données
df = pd.read_excel('Online_Retail.xlsx')
print(f"Dataset initial : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Nettoyage
print("\nÉtapes de nettoyage :")
df_clean = df[df['CustomerID'].notna()].copy()
print(f"  ✓ Suppression CustomerID manquants : {len(df_clean)} lignes restantes")

df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
print(f"  ✓ Suppression annulations : {len(df_clean)} lignes restantes")

df_clean = df_clean[df_clean['UnitPrice'] > 0]
print(f"  ✓ Suppression prix <= 0 : {len(df_clean)} lignes restantes")

df_clean = df_clean[df_clean['Quantity'] > 0]
print(f"  ✓ Suppression quantités <= 0 : {len(df_clean)} lignes restantes")

df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# ==================== 2. CALCUL RFM ====================
print("\n2. CALCUL DES MÉTRIQUES RFM")
print("-" * 80)

max_date = df_clean['InvoiceDate'].max()
print(f"Date de référence : {max_date}")

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"\n{len(rfm)} clients analysés")
print("\nStatistiques RFM :")
print(rfm.describe())

# ==================== 3. ANALYSE DE LA DISTRIBUTION ====================
print("\n3. ANALYSE DE LA DISTRIBUTION (SKEWNESS)")
print("-" * 80)

print("\nAsymétrie (skewness) :")
print(f"  Récence  : {stats.skew(rfm['Recency']):.4f}")
print(f"  Fréquence: {stats.skew(rfm['Frequency']):.4f}")
print(f"  Montant  : {stats.skew(rfm['Monetary']):.4f}")


# Visualisation des distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    # Distribution originale
    axes[0, idx].hist(rfm[col], bins=50, edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'{col} - Distribution Originale')
    axes[0, idx].set_xlabel(col)
    axes[0, idx].set_ylabel('Fréquence')
    axes[0, idx].axvline(rfm[col].mean(), color='red', linestyle='--', label='Moyenne')
    axes[0, idx].legend()
    
    # Distribution log-transformée
    log_data = np.log1p(rfm[col])
    axes[1, idx].hist(log_data, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, idx].set_title(f'{col} - Après Log Transform')
    axes[1, idx].set_xlabel(f'log({col})')
    axes[1, idx].set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rfm_distributions.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/rfm_distributions.png")
plt.close()

# Transformation logarithmique
rfm_log = rfm.copy()
rfm_log['Recency'] = np.log1p(rfm['Recency'])
rfm_log['Frequency'] = np.log1p(rfm['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm['Monetary'])

# Standardisation
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log[['Recency', 'Frequency', 'Monetary']])

# ==================== 4. ANALYSE EN COMPOSANTES PRINCIPALES (PCA) ====================
print("\n4. ANALYSE EN COMPOSANTES PRINCIPALES (PCA)")
print("-" * 80)

pca = PCA()
pca_components = pca.fit_transform(rfm_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nVariance expliquée par composante :")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) | Cumulé: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Visualisation PCA
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, edgecolor='black')
axes[0].plot(range(1, len(explained_variance) + 1), explained_variance, 'ro-')
axes[0].set_xlabel('Composante Principale')
axes[0].set_ylabel('Variance Expliquée')
axes[0].set_title('Scree Plot')
axes[0].set_xticks(range(1, len(explained_variance) + 1))

axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='Seuil 95%')
axes[1].set_xlabel('Nombre de Composantes')
axes[1].set_ylabel('Variance Cumulée')
axes[1].set_title('Variance Cumulée')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/pca_analysis.png")
plt.close()

# PCA 2D pour visualisation
pca_2d = PCA(n_components=2)
pca_2d_result = pca_2d.fit_transform(rfm_scaled)

rfm['PC1'] = pca_2d_result[:, 0]
rfm['PC2'] = pca_2d_result[:, 1]

# ==================== 5. CLUSTERING K-MEANS ====================
print("\n5. CLUSTERING K-MEANS")
print("-" * 80)

wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Visualisation méthode du coude
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de Clusters (K)')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Méthode du Coude')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Score de Silhouette')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_elbow.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/kmeans_elbow.png")
plt.close()

# K optimal
optimal_k = 4
print(f"\nK optimal sélectionné : {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans_final.fit_predict(rfm_scaled)

print(f"Silhouette Score (K={optimal_k}) : {silhouette_score(rfm_scaled, rfm['KMeans_Cluster']):.4f}")

# ==================== 6. CLUSTERING DBSCAN ====================
print("\n6. CLUSTERING DBSCAN")
print("-" * 80)

# K-distance pour epsilon
k = 4
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(rfm_scaled)
distances, indices = neighbors_fit.kneighbors(rfm_scaled)
distances_sorted = np.sort(distances[:, k-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances_sorted)
plt.xlabel('Points de Données (triés)')
plt.ylabel(f'Distance au {k}-ième Voisin')
plt.title('Graphique K-Distance pour DBSCAN')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'kdistance_graph.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/kdistance_graph.png")
plt.close()

eps_value = 0.5
min_samples = 4

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)

n_clusters = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'] else 0)
n_noise = list(rfm['DBSCAN_Cluster']).count(-1)

print(f"\nDBSCAN (eps={eps_value}, min_samples={min_samples}) :")
print(f"  Clusters : {n_clusters}")
print(f"  Bruit : {n_noise} ({n_noise/len(rfm)*100:.2f}%)")

# ==================== 7. VISUALISATION DES CLUSTERS ====================
print("\n7. VISUALISATION DES CLUSTERS")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

scatter1 = axes[0].scatter(rfm['PC1'], rfm['PC2'], 
                           c=rfm['KMeans_Cluster'], 
                           cmap='viridis', alpha=0.6, edgecolor='black')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[0].set_title(f'K-Means (K={optimal_k})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

scatter2 = axes[1].scatter(rfm['PC1'], rfm['PC2'], 
                           c=rfm['DBSCAN_Cluster'], 
                           cmap='viridis', alpha=0.6, edgecolor='black')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[1].set_title(f'DBSCAN (eps={eps_value})')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_comparison.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/clustering_comparison.png")
plt.close()

# ==================== 8. PROFILS DES SEGMENTS (PERSONAS) ====================
print("\n8. PROFILS DES SEGMENTS - PERSONAS CLIENTS")
print("-" * 80)

cluster_profile = rfm.groupby('KMeans_Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median'],
    'CustomerID': 'count'
})

print("\nProfils des clusters :")
print(cluster_profile)

# Radar charts
from math import pi

categories = ['Recency', 'Frequency', 'Monetary']
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

cluster_means = rfm.groupby('KMeans_Cluster')[categories].mean()
cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

for cluster_id in range(optimal_k):
    ax = axes[cluster_id]
    values = cluster_means_normalized.loc[cluster_id].values.tolist()
    values += values[:1]
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster {cluster_id}\n({int(cluster_profile.loc[cluster_id, ("CustomerID", "count")])} clients)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_radar_charts.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Sauvegardé : {OUTPUT_DIR}/cluster_radar_charts.png")
plt.close()

# Interprétation business
print("\nINTERPRÉTATION BUSINESS - PERSONAS :")
print("=" * 80)

for cluster_id in range(optimal_k):
    cluster_data = rfm[rfm['KMeans_Cluster'] == cluster_id]
    r_mean = cluster_data['Recency'].mean()
    f_mean = cluster_data['Frequency'].mean()
    m_mean = cluster_data['Monetary'].mean()
    
    print(f"\n{'━'*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'━'*70}")
    print(f"  Taille      : {len(cluster_data)} clients ({len(cluster_data)/len(rfm)*100:.1f}%)")
    print(f"  Récence     : {r_mean:.1f} jours")
    print(f"  Fréquence   : {f_mean:.1f} transactions")
    print(f"  Montant     : £{m_mean:.2f}")
    
    # Nommer le segment
    if r_mean < rfm['Recency'].median() and f_mean > rfm['Frequency'].median() and m_mean > rfm['Monetary'].median():
        segment_name = "🏆 CHAMPIONS"
        action = "Programme VIP, accès prioritaire, récompenses"
    elif r_mean < rfm['Recency'].median() and f_mean > rfm['Frequency'].median():
        segment_name = "💚 CLIENTS FIDÈLES"
        action = "Up-selling, programme de fidélité"
    elif r_mean > rfm['Recency'].median() and m_mean > rfm['Monetary'].median():
        segment_name = "⚠️ AT RISK - HAUTE VALEUR"
        action = "Win-back urgent, offres exclusives"
    else:
        segment_name = "💤 PERDUS/HIBERNANTS"
        action = "Réengagement, remises importantes"
    
    print(f"  Segment     : {segment_name}")
    print(f"  Action      : {action}")

# ==================== 9. SAUVEGARDE DES RÉSULTATS ====================
print("\n" + "=" * 80)
print("9. SAUVEGARDE DES RÉSULTATS")
print("=" * 80)

rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_with_clusters.csv"), index=False)
print(f"✓ Sauvegardé : {OUTPUT_DIR}/rfm_with_clusters.csv")

print("\n" + "=" * 80)
print("PARTIE 1 TERMINÉE - TOUS LES LIVRABLES GÉNÉRÉS")
print("=" * 80)
print("\nFichiers créés dans le dossier 'output/' :")
print("  ✓ rfm_with_clusters.csv")
print("  ✓ rfm_distributions.png")
print("  ✓ pca_analysis.png")
print("  ✓ kmeans_elbow.png")
print("  ✓ kdistance_graph.png")
print("  ✓ clustering_comparison.png")
print("  ✓ cluster_radar_charts.png")
