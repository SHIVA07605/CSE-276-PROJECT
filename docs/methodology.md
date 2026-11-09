# Methodology & Mathematical Formulations

## 1. Problem Formulation
Let $\mathcal{R} = \{r_1, r_2, \dots, r_N\}$ be a collection of material records across multiple public sector enterprises.
Each record $r_i$ is defined by a tuple:
$$r_i = (E_i, C_i, D_i, \mathbf{A}_i)$$
where $E_i$ is enterprise identity, $C_i$ is local material code, $D_i$ is unstructured description text, and $\mathbf{A}_i$ is an extracted dictionary of technical attributes (e.g., diameter, grade, pressure rating).

Our goal is twofold:
1. **Pairwise Entity Resolution**: Learn a binary decision function $f(r_i, r_j) \in \{0, 1\}$ predicting whether $r_i$ and $r_j$ represent functionally equivalent materials.
2. **Canonical Clustering**: Partition $\mathcal{R}$ into disjoint equivalence classes (clusters) $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ such that records in the same cluster represent the same physical entity, assigning a single unified harmonised code $\mathcal{H}(c_k)$ with supporting evidence.

---

## 2. Baseline Model: Exact String Matching
The simplest baseline tests whether normalised descriptions or codes match verbatim:
$$f_{\text{baseline}}(r_i, r_j) = \mathbb{I}(\text{norm}(D_i) = \text{norm}(D_j))$$
- **Hypothesis**: High precision, very poor recall due to vocabulary mismatch, abbreviations, and word order differences.

---

## 3. Candidate Generation (Blocking)
Evaluating all $\binom{N}{2} = \frac{N(N-1)}{2}$ pairs is computationally infeasible ($O(N^2)$).
We construct an inverted index using standard TF-IDF token n-grams or attribute blocks.
Pairs are retrieved only if:
$$\text{TF-IDF-Cosine}(\mathbf{t}_i, \mathbf{t}_j) \ge \tau_{\text{block}}$$
This reduces candidate comparisons to $O(N \log N)$ while preserving high pair completeness (recall of true matches).

---

## 4. Multi-Modal Feature Extraction
For each candidate pair $(r_i, r_j)$, we extract a hybrid feature vector $\mathbf{x}_{ij} \in \mathbb{R}^D$:

1. **Lexical Similarities**:
   - Token-Sort Levenshtein Ratio: $\text{LevRatio}(\text{sort}(D_i), \text{sort}(D_j))$
   - Jaccard Similarity on n-grams: $\frac{|Tokens(D_i) \cap Tokens(D_j)|}{|Tokens(D_i) \cup Tokens(D_j)|}$
2. **Dense Semantic Similarity**:
   - Compute embeddings using pre-trained `sentence-transformers/all-MiniLM-L6-v2`:
     $$\mathbf{e}_i = \text{TransformerEncoder}(D_i)$$
   - Semantic Cosine Similarity:
     $$\text{sim}_{\text{sem}}(r_i, r_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$
3. **Engineering Attribute Compatibility**:
   - For extracted attributes (dimensions, material grade, thread type):
     $$\text{AttrMatch}(k) = \begin{cases} 1 & \text{if } A_i^{(k)} = A_j^{(k)} \\ -1 & \text{if } A_i^{(k)} \ne A_j^{(k)} \text{ (conflict)} \\ 0 & \text{if either is missing} \end{cases}$$

---

## 5. Supervised Pairwise Classifier
A Random Forest classifier or Regularized Logistic Regression is trained on labelled pairs:
$$\hat{y}_{ij} = P(y_{ij} = 1 \mid \mathbf{x}_{ij}) = \text{Classifier}(\mathbf{x}_{ij})$$
A match is asserted when $\hat{y}_{ij} \ge \tau_{\text{match}}$.

---

## 6. Graph Clustering & Canonical Harmonisation
Construct an undirected graph $G = (V, E)$, where $V = \mathcal{R}$ and $(r_i, r_j) \in E \iff \hat{y}_{ij} \ge \tau_{\text{match}}$.
- Connected components or Agglomerative Clustering groups records into clusters.
- For each cluster $c_k$:
  - Synthesise a canonical harmonised code: `HARMONISED-MAT-<HASH/SEQ>`
  - Compile audit evidence: intersecting technical attributes, source enterprises, confidence intervals.

---

## 7. Evaluation Metrics
1. **Pairwise Metrics**:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
2. **Cluster Purity**:
   Given predicted clusters $\Omega = \{\omega_1, \dots, \omega_K\}$ and ground-truth classes $C = \{c_1, \dots, c_J\}$:
   $$\text{Purity}(\Omega, C) = \frac{1}{N} \sum_{k} \max_{j} |\omega_k \cap c_j|$$
