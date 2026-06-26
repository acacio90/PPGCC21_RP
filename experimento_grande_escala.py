"""
Experimento: acurácia do ML em grafos onde o backtracking falha.

Para grafos com n > limiar de timeout, o backtracking exato não consegue
calcular χ(G). Este script avalia se o modelo treinado consegue predizer
χ(G) com boa acurácia nessa faixa, usando famílias com χ analiticamente
conhecido como ground truth.

Famílias de teste:
  - Grafos k-partidos completos (n > 100): χ = k  (combinações fora do treino)
  - Grafos de Mycielski M_k (k = 6..9): χ = k, n = 47..383
  - Cliques K_k (k = 25..50): χ = k  (controle — estrutura trivial)
"""
import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
FIGURES_DIR  = Path('figures')
DATASET_PATH = Path('data/processed/dataset.csv')

FEATURE_COLS = [
    'n_vertices', 'n_edges', 'density',
    'min_degree', 'max_degree', 'mean_degree', 'std_degree',
    'avg_clustering', 'diameter', 'is_disconnected',
    'avg_betweenness', 'largest_clique', 'n_components',
]

# ---------------------------------------------------------------------------
# Extração de características (idêntica ao notebook 2)
# ---------------------------------------------------------------------------
def _find_largest_clique(G_s, timeout=10):
    result    = [1]
    stop_flag = threading.Event()
    def run():
        best = 1
        try:
            for clique in nx.find_cliques(G_s):
                if stop_flag.is_set():
                    result[0] = best
                    return
                if len(clique) > best:
                    best = len(clique)
                    result[0] = best
            result[0] = best
        except Exception:
            result[0] = best
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        stop_flag.set()
    return result[0]

def extract_features(G, known_clique=None):
    G_s = nx.Graph(G)
    G_s.remove_edges_from(nx.selfloop_edges(G_s))
    G_s = nx.convert_node_labels_to_integers(G_s)
    n = G_s.number_of_nodes()
    if n == 0:
        return None
    m       = G_s.number_of_edges()
    degrees = [d for _, d in G_s.degree()]
    is_conn = nx.is_connected(G_s)
    n_comp  = nx.number_connected_components(G_s)
    if is_conn:
        if n <= 500:
            diameter = nx.diameter(G_s)
        else:
            sources  = list(np.random.choice(n, min(10, n), replace=False))
            diameter = max(
                max(nx.single_source_shortest_path_length(G_s, s).values())
                for s in sources)
    else:
        diameter = -1
    k_bc = min(50, n) if n > 200 else None
    bc   = nx.betweenness_centrality(G_s, k=k_bc, normalized=True,
                                     seed=RANDOM_STATE)
    largest_clique = known_clique if known_clique is not None \
                     else _find_largest_clique(G_s, timeout=10)
    return {
        'n_vertices'     : n,
        'n_edges'        : m,
        'density'        : nx.density(G_s),
        'min_degree'     : int(np.min(degrees)),
        'max_degree'     : int(np.max(degrees)),
        'mean_degree'    : float(np.mean(degrees)),
        'std_degree'     : float(np.std(degrees)),
        'avg_clustering' : nx.average_clustering(G_s),
        'diameter'       : diameter,
        'is_disconnected': int(not is_conn),
        'avg_betweenness': float(np.mean(list(bc.values()))),
        'largest_clique' : largest_clique,
        'n_components'   : n_comp,
    }

# ---------------------------------------------------------------------------
# 1. Treinar modelo no dataset completo
# ---------------------------------------------------------------------------
print('Carregando dataset e treinando modelo...')
df = pd.read_csv(DATASET_PATH)
X  = df[FEATURE_COLS].values.astype(float)
y  = df['chi'].values.astype(int)

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

model = DecisionTreeClassifier(random_state=RANDOM_STATE)
model.fit(X_sc, y)
print(f'  Modelo treinado em {len(df)} instâncias, {len(np.unique(y))} classes.')

# ---------------------------------------------------------------------------
# 2. Gerar grafos de teste com χ analítico (zona de timeout)
# ---------------------------------------------------------------------------
rng      = np.random.default_rng(RANDOM_STATE)
test_set = []   # (label, G, true_chi, known_clique)

# --- Grafos k-partidos completos (k > 10, s > 6 → n > 60, fora do treino) ---
print('Gerando k-partidos completos (fora da distribuição de treino)...')
for k in range(10, 22):
    for s in [7, 8, 10, 12]:
        G = nx.complete_multipartite_graph(*([s] * k))
        test_set.append((f'kpartite k={k}', G, k, k))

# --- Grafos de Mycielski M_k (χ = k, não estavam no treino) ---
print('Gerando grafos de Mycielski...')
for k in range(5, 10):
    G   = nx.mycielski_graph(k)
    chi = k
    # clique number de Mycielski: M_2=K_2 (ω=2), M_k triangle-free para k≥3 (ω=2)
    clique_val = 2 if k >= 3 else k
    test_set.append((f'mycielski k={k}', G, chi, clique_val))

# --- Cliques K_k com k > 25 (controle: estrutura trivial, χ = k) ---
print('Gerando cliques grandes (controle)...')
for k in range(25, 55, 5):
    G = nx.complete_graph(k)
    test_set.append((f'clique k={k}', G, k, k))

print(f'Total de grafos de teste: {len(test_set)}')

# ---------------------------------------------------------------------------
# 3. Extrair features e predizer
# ---------------------------------------------------------------------------
print('Extraindo features e predizendo...')
rows = []
for label, G, true_chi, known_clique in tqdm(test_set, unit='grafo'):
    feats = extract_features(G, known_clique=known_clique)
    if feats is None:
        continue
    X_test   = scaler.transform([list(feats[c] for c in FEATURE_COLS)])
    pred_chi = model.predict(X_test)[0]
    rows.append({
        'familia'  : label.split(' ')[0],
        'label'    : label,
        'n'        : feats['n_vertices'],
        'true_chi' : true_chi,
        'pred_chi' : pred_chi,
        'correto'  : int(pred_chi == true_chi),
    })

results = pd.DataFrame(rows)
print('\n=== Acurácia por família ===')
acc_by_family = results.groupby('familia')['correto'].agg(['sum','count'])
acc_by_family['acuracia'] = acc_by_family['sum'] / acc_by_family['count']
print(acc_by_family.round(3).to_string())
print(f'\nAcurácia global: {results["correto"].mean():.3f}')

# ---------------------------------------------------------------------------
# 4. Figura: predito vs real
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Scatter: predito vs real ---
ax = axes[0]
familias   = results['familia'].unique()
color_map  = {'kpartite': '#1f77b4', 'mycielski': '#ff7f0e', 'clique': '#2ca02c'}
for fam in familias:
    sub = results[results['familia'] == fam]
    ax.scatter(sub['true_chi'], sub['pred_chi'],
               label=fam, alpha=0.6, s=50,
               color=color_map.get(fam, 'gray'))
chi_max = max(results['true_chi'].max(), results['pred_chi'].max()) + 1
ax.plot([1, chi_max], [1, chi_max], 'k--', linewidth=0.8, label='perfeito')
ax.set_xlabel('χ(G) real')
ax.set_ylabel('χ(G) predito')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Acurácia por n_vertices (faixas) ---
ax = axes[1]
bins   = [0, 50, 80, 120, 180, 250, 9999]
labels_b = ['≤50', '51–80', '81–120', '121–180', '181–250', '>250']
results['n_bin'] = pd.cut(results['n'], bins=bins, labels=labels_b)
acc_n = results.groupby('n_bin', observed=True)['correto'].mean() * 100
bars = ax.bar(acc_n.index.astype(str), acc_n.values,
              color='steelblue', edgecolor='black', width=0.6)
ax.axhline(50, color='red', linestyle='--', linewidth=0.8, label='50%')
ax.set_xlabel('Número de vértices')
ax.set_ylabel('Acurácia (%)')
ax.set_ylim(0, 110)
for bar, val in zip(bars, acc_n.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2,
            f'{val:.0f}%', ha='center', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = FIGURES_DIR / 'experimento_grande_escala.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nFigura salva: {out}')
