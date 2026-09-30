# Kinship Library phase notebooks

Run the notebooks in this order.

1. `01_collection_data_quality_eda.ipynb`
2. `02_nlp_semantic_foundations.ipynb`
3. `03_kmeans_pca_first_recommender.ipynb`

Notebook 01 creates `data/processed/books_clean.csv`.

Notebook 02 creates `data/processed/books_nlp_ready.csv` and `models/book_embeddings.npy`.

Notebook 03 creates `data/processed/books_modeled.csv`.

The notebooks use repository relative paths so the same structure can be used in Google Colab or a cloned repository.

Google Books is not required in this version because the current Colab test returned HTTP 429. It can be added later as optional enrichment after the quota or authentication issue is resolved.

The semantic recommender uses Sentence Transformers as a semantic embedding layer. TF IDF remains the interpretable baseline required for the project.
