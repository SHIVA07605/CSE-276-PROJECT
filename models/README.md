# Models Directory

This directory stores serialised models and vectorizers produced during training:
- TF-IDF vectorizers
- Sentence-transformer cached embeddings / fine-tuned checkpoints
- Random Forest pairwise classifiers (`joblib` or `pkl`)
- Threshold configs and calibration parameters

Binary artifacts (>100MB) should be tracked via external storage or Git LFS, with metadata and configuration committed here.
