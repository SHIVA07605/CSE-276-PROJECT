# System Architecture

## Overview
The Material Code Harmonisation platform is designed as a modular, tiered pipeline combining lexical preprocessing, candidate blocking, deep semantic embeddings, pairwise supervised classification, graph-based entity clustering, and a human-in-the-loop reviewer interface.

```mermaid
flowchart TD
    subgraph Ingestion["1. Data Ingestion & Schema Alignment"]
        EntA["Enterprise A Catalogue"] --> Loader["src/data_loader.py"]
        EntB["Enterprise B Catalogue"] --> Loader
        EntC["Enterprise C Catalogue"] --> Loader
    end

    subgraph Preprocess["2. Text Preprocessing & Attribute Parsing"]
        Loader --> Cleaner["src/preprocess.py"]
        Cleaner --> UnitStd["Unit Standardisation (mm, M10, SS)"]
        Cleaner --> AttrExt["Attribute Extraction (Grade, Dims, Rating)"]
    end

    subgraph Candidate["3. Candidate Generation (Blocking)"]
        AttrExt --> Blocker["src/candidate_generation.py"]
        Blocker --> Pairs["Candidate Pairs (Sub-quadratic O(N log N))"]
    end

    subgraph FeatureEng["4. Feature Extraction Engine"]
        Pairs --> Feat["src/feature_extraction.py"]
        Feat --> LexSim["Lexical Similarities (Token, Levenshtein, Jaccard)"]
        Feat --> SemSim["Semantic Similarity (all-MiniLM-L6-v2 Cosine)"]
        Feat --> AttrComp["Attribute Compatibility (Match / Conflict)"]
    end

    subgraph ModelLayer["5. Entity Resolution Model"]
        LexSim --> PairClassifier["src/model.py (Random Forest Classifier)"]
        SemSim --> PairClassifier
        AttrComp --> PairClassifier
        PairClassifier --> MatchProb["Pairwise Match Probability & Decision"]
    end

    subgraph ClusteringLayer["6. Entity Grouping & Harmonisation"]
        MatchProb --> GraphClust["src/clustering.py (Connected Components)"]
        GraphClust --> Harmoniser["src/harmonisation.py (Canonical Code Synthesis)"]
    end

    subgraph UI["7. Human-in-the-Loop Reviewer Workspace"]
        Harmoniser --> App["app/app.py (Streamlit)"]
        App --> Reviewer["Reviewer Workspace (Approve / Reject / Uncertain)"]
        Reviewer --> Feedback["Audit Log & Active Learning Store"]
    end
```

## Architectural Principles
1. **Modularity**: Core business and ML logic resides strictly in `src/`. Streamlit UI components and Jupyter notebooks import from `src/`.
2. **Sub-quadratic Scalability**: Pairwise comparison of $N$ items requires $O(N^2)$ checks without blocking. `candidate_generation.py` uses inverted TF-IDF / category blocking to prune unlikely pairs to $O(N \log N)$ or $O(K \cdot N)$.
3. **Evidence-Driven Decisions**: Rather than presenting an opaque prediction, the model outputs lexical overlap, semantic distance, and specific matching/conflicting attributes (e.g. diameter, material grade).
4. **Human Oversight**: High-confidence pairs can be auto-staged, but merges affecting master procurement records require human confirmation.
