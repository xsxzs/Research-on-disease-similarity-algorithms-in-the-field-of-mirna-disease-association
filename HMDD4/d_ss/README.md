# Disease Semantic Similarity Calculation (Detailed Pipeline)

This project provides a professional pipeline to calculate a semantic similarity matrix for diseases based on the **MeSH 2025** ontology. It specifically addresses the challenge of merging general descriptors with highly specific supplemental concepts.

---

##  Stage 1: Identity Discovery & Mapping (`missing_disease.py`)

This stage ensures every disease in your input has a valid MeSH ID.

### 1. Dual-Layer Search strategy
- **Primary Search (Descriptors):** Matches names against `desc2025.xml` to find "Main Headings" (D-series IDs). The results are stored in **`matchmesh.csv`**.
- **Secondary Search (Supplemental):** For unmatched terms, the script searches `supp2025.xml` (C-series IDs). The results are stored in **`fixed_matches1.csv`**.

### 2. Output
- `final_data.csv`: Initial matches from the main descriptor database.
- `missing_dis_supp.csv`: Recovered matches from the supplemental database.

---

##  Stage 2: Data Integration & Merging (`DataDeal.py`)

Since your data sources have different lengths and levels of granularity, this stage synchronizes them into a single, clean input for calculation.

### 1. Name-Based Merging Logic
Instead of relying on row numbers, the script uses **`disease_name`** as the unique key to align the two datasets:
- **Baseline:** It starts with `final_data.csv` as the foundation.
- **Overwrite Mechanism:** It iterates through `missing_dis_supp.csv`. If a disease exists in both files, the more specific **Supplemental ID (C-series)** from the second file overwrites/fills the entry in the first.
- **Why this matters:** Supplemental concepts often provide the "missing link" for rare diseases that are too specific for the main descriptor tree.

### 2. Final Output
- **`final_data.csv`**: A consolidated file containing the most accurate MeSH ID for every disease in your study. This is the **mandatory input** for the next stage.

---

## 🧮 Stage 3: Semantic Computation (`compute_disease_semantic.py`)

Implements **Wang's Algorithm** to convert hierarchical structures into numerical similarity scores.

### 1. DAG (Directed Acyclic Graph) Construction
The script builds a graph for each disease, tracing all its ancestors back to the root of the MeSH tree.

### 2. Semantic Value Calculation ($S$-value)
- **Decay Step:** Contribution of ancestors decreases as they get further from the disease: 
  $$
  S_D(t) = \max \{ \Delta \cdot S_D(t') \mid t' \in children(t) \}
  $$
  Δ=0.5

### 3. Similarity Scoring
The final score is based on the overlap of two diseases' DAGs:
$$
DSS_2(d_i, d_j) = \frac{\sum_{t \in T_i \cap T_j} (S_i(t) + S_j(t)) \cdot IC(t)}{\sum_{t \in T_i \cup T_j} (S_i(t) + S_j(t)) \cdot IC(t)}
$$


---

## 📊 Final Output: `d_ss.csv`
A symmetric matrix where each cell $[i, j]$ (0.0 to 1.0) represents the semantic similarity between disease $i$ and disease $j$.
