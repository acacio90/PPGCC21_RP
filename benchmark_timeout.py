"""
Benchmark: taxa de timeout do backtracking por número de vértices.
Gera grafos Erdős-Rényi com densidades variadas e mede sucesso/timeout.
Execute com: python benchmark_timeout.py
"""
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from tqdm import tqdm

TIMEOUT_SEC = 10       # timeout por grafo (s)
N_GRAPHS    = 8        # amostras por (n, p)
FIGURES_DIR = Path('figures')
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Backtracking exato com timeout (idêntico ao notebook 1)
# ---------------------------------------------------------------------------
def chromatic_number_backtracking(G, timeout=TIMEOUT_SEC):
    G_s = nx.Graph(G)
    G_s.remove_edges_from(nx.selfloop_edges(G_s))
    G_s = nx.convert_node_labels_to_integers(G_s)
    n = G_s.number_of_nodes()
    if n == 0: return 0
    if G_s.number_of_edges() == 0: return 1
    order = sorted(range(n), key=lambda v: -G_s.degree(v))
    adj   = [list(G_s.neighbors(v)) for v in range(n)]
    try:
        lb = max(len(c) for c in nx.find_cliques(G_s))
    except Exception:
        lb = 1
    greedy = nx.greedy_color(G_s, strategy='largest_first')
    ub = max(greedy.values()) + 1
    if lb == ub:
        return lb
    result    = [None]
    stop_flag = threading.Event()

    def is_colorable(k):
        colors = [-1] * n
        def bt(idx):
            if stop_flag.is_set(): return None
            if idx == n: return True
            v    = order[idx]
            used = {colors[u] for u in adj[v] if colors[u] >= 0}
            for c in range(k):
                if c not in used:
                    colors[v] = c
                    r = bt(idx + 1)
                    if r is None: return None
                    if r: return True
                    colors[v] = -1
            return False
        return bt(0)

    def compute():
        for k in range(lb, ub):
            if stop_flag.is_set(): return
            r = is_colorable(k)
            if r is None: return
            if r:
                result[0] = k
                return
        result[0] = ub

    t = threading.Thread(target=compute, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        stop_flag.set()
        return None
    return result[0]

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
sizes     = list(range(5, 35, 5)) + list(range(35, 75, 10))
densities = [0.3, 0.5, 0.7]
results   = {p: {'n': [], 'timeout_rate': []} for p in densities}

total = len(sizes) * len(densities) * N_GRAPHS
bar   = tqdm(total=total, desc='Benchmark', unit='grafo',
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

for p in densities:
    for n in sizes:
        timeouts = 0
        for _ in range(N_GRAPHS):
            seed = int(rng.integers(100_000))
            G    = nx.erdos_renyi_graph(n, p, seed=seed)
            chi  = chromatic_number_backtracking(G, timeout=TIMEOUT_SEC)
            if chi is None:
                timeouts += 1
            bar.set_postfix(n=n, p=p, timeout_rate=f'{timeouts}/{_ + 1}')
            bar.update(1)
        rate = timeouts / N_GRAPHS * 100
        results[p]['n'].append(n)
        results[p]['timeout_rate'].append(rate)

bar.close()

# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

colors_map = {0.3: '#1f77b4', 0.5: '#ff7f0e', 0.7: '#d62728'}
for p in densities:
    ax.plot(results[p]['n'], results[p]['timeout_rate'],
            marker='o', linewidth=2, label=f'p = {p}',
            color=colors_map[p])

ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
           label='50% timeout')
ax.set_xlabel('Número de vértices ($n$)')
ax.set_ylabel(f'Taxa de timeout ({TIMEOUT_SEC}s) [%]')
ax.set_ylim(-5, 105)
ax.set_xticks(sizes)
ax.legend(title='Densidade')
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = FIGURES_DIR / 'timeout_benchmark.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nFigura salva em: {out}')
