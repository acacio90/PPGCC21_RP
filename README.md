# PPGCC21_RP — Predição do Número Cromático de Grafos

Projeto PPGCC21: predição do número cromático χ(G) de grafos via aprendizado de máquina, utilizando características estruturais e topológicas como entrada.

## Estrutura

```
PPGCC21_RP/
├── 1 - carregamento_e_rotulacao.ipynb      # Geração de grafos sintéticos e cálculo de χ(G)
├── 2 - extracao_de_caracteristicas.ipynb   # Extração de features e construção do dataset
├── 3 - analise_exploratoria.ipynb          # EDA: distribuição, correlação, informação mútua
├── 4 - modelagem_e_validacao.ipynb         # Nested CV com DT, RF, SVM, KNN e MLP
├── 5 - analise_estatistica.ipynb           # Teste de Friedman e post-hoc de Nemenyi
├── benchmark_timeout.py                    # Benchmark de timeout do backtracking por n_vertices
├── experimento_grande_escala.py            # Avalia acurácia do modelo em grafos onde o backtracking falha
├── data/
│   ├── dimacs/                             # Instâncias DIMACS (.col)
│   └── processed/                          # dataset.csv e results.pkl — não versionados
├── figures/                                # Figuras geradas — não versionadas
├── requirements.txt
└── .venv/                                  # Ambiente virtual — não versionado
```

## Como rodar

**1. Crie e ative o ambiente virtual**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
```

**2. Instale as dependências**

```bash
pip install -r requirements.txt
```

**3. Registre o kernel no Jupyter**

```bash
python -m ipykernel install --user --name=ppgcc21 --display-name "Python (PPGCC21)"
```

**4. Execute os notebooks em ordem**

Abra o Jupyter e execute os notebooks de 1 a 5 sequencialmente. Cada notebook salva seus artefatos em `data/processed/` ou `figures/`, que são reutilizados pelos seguintes.

```bash
jupyter notebook
```

> **Instâncias DIMACS:** os arquivos `.col` estão versionados em `data/dimacs/` e são carregados automaticamente pelo notebook 1.

## Benchmark de Timeout

Para analisar até qual tamanho de grafo o backtracking consegue calcular χ(G) dentro do limite de tempo, execute:

```bash
python benchmark_timeout.py
```

O script gera grafos Erdős-Rényi com densidades variadas (p = 0,3 / 0,5 / 0,7) e mede a taxa de timeout para cada tamanho. A figura é salva em `figures/timeout_benchmark.png`.

Parâmetros ajustáveis no topo do arquivo:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `TIMEOUT_SEC` | `10` | Timeout por grafo em segundos |
| `N_GRAPHS` | `8` | Amostras por combinação (n, p) |
| `sizes` | 5 a 65 | Tamanhos de grafo testados |

## Experimento de Generalização em Grande Escala

Para avaliar se o modelo treinado prediz χ(G) corretamente em grafos onde o backtracking falha por timeout, execute:

```bash
python experimento_grande_escala.py
```

O script treina a Árvore de Decisão no dataset completo e testa em grafos de famílias com χ analiticamente conhecido (k-partidos completos, Mycielski e cliques grandes). A figura é salva em `figures/experimento_grande_escala.png`.

## Dependências principais

| Pacote | Uso |
|--------|-----|
| networkx | Geração e análise de grafos |
| scikit-learn | Modelos e validação cruzada |
| imbalanced-learn | SMOTE para balanceamento |
| scikit-posthocs | Teste post-hoc de Nemenyi |
| scipy | Teste de Friedman |
