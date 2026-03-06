# Disease Functional Similarity Calculation Pipeline

This project calculates the functional similarity between diseases based on their associated genes and Gene Ontology (GO) annotations. The pipeline follows these structured phases:

---

### Phase 1: Gene Mapping
The goal of this phase is to identify as many associated genes as possible for the disease list.

1. **`match_genes_with_DisGeNET.py` (Exact Matching)**
   - **Function**: Performs the initial wave of basic matching. It reads the input file and the local gene database `DisGeNET.txt`. Genes are matched to diseases using an exact name-matching principle (case-insensitive).
   - **Output**: Generates `disease_genes_first.csv`, adding `genes` (gene list) and `gene_count` columns.

2. **`match_genes_with_synonyms.py` (Synonym Recovery Matching)**
   - **Function**: For diseases that failed to match in the first step, this script builds an extensive "synonym dictionary" by parsing MeSH XML files (such as `desc2025.xml`). If the original disease name has no genes, the script checks all its synonyms in the gene database to find a match.
   - **Output**: Generates `disease_genes_second.csv`.

---

### Phase 2: Online Scraping & API Retrieval
For stubborn diseases that cannot be resolved via local databases or synonyms, we utilize the online DisGeNET database. Since genes cannot be queried directly by disease name, we first search for the corresponding CUI and then use the CUI to retrieve the genes.

5. **`selenium_scrape_cui_by_diseasename.py` (CUI Scraping via Disease Name)**
   - **Function**: A Selenium crawler that searches for the `disease_name` directly on the DisGeNET website to capture the corresponding `cui`.
   - **Output**: Generates `disease_genes_third.csv`.

6. **`scrape_gene.py` (Gene Retrieval via API)**
   - **Function**: After obtaining the CUIs, this script calls the official DisGeNET API. it iterates through the table and fetches the top 50 associated genes (sorted by score) for any disease that has a CUI but no gene data.
   - **Output**: Generates `scrape_with_genes_final.csv`. At this stage, all diseases should have associated genes.

---

### Phase 3: Gene Ontology Annotation (GO Terms)

1. **`Gene_Ontology.py` (GO Annotation Retrieval)**
   - **Function**: Reads the gene-populated table (`scrape_with_genes_final.csv`). For each disease's gene list, it calls the **g:Profiler API** to query the Biological Processes (GO:BP), Molecular Functions (GO:MF), and Cellular Components (GO:CC). All GO IDs are aggregated into a set and assigned to the disease.
   - **Output**: Generates a file with a `go_terms` column named `disease_go.csv`.

---

### Phase 4: Similarity Calculation

2. **`calculate_similarity.py` (Core Calculation)**
   - **Function**: Reads the `disease_go.csv` file containing diseases and their GO terms.
   - **Algorithm Logic**:
     - Parses GO terms for each disease into sets.
     - Performs pairwise disease comparisons using the **Jaccard Coefficient** (intersection divided by union).
     - **Innovation**: Introduces an **Exponential Decay Penalty Mechanism** based on count differences (`math.exp(-alpha * difference / max_value)`). This means if Disease A has 1000 annotations and Disease B has only 10, the similarity score is suppressed to avoid "false positives" caused by extreme baseline disparities (e.g., well-studied vs. rare diseases).
   - **Output**: Generates the final disease functional similarity matrix file: `d_fs.csv`.

---

### Automation: `run_pipeline.py`
*(Added based on recent updates)* A master script to execute the above sequence automatically.