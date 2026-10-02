# Kinship Library
## Books for Living in Connection

Kinship Library is an interactive semantic book recommender exploring relationships between ecology, Indigenous knowledge of the Americas, Yoga philosophy, Ayurveda, contemplative traditions and embodied ways of knowing.

Rather than beginning with a genre or a known title, the library invites readers to begin with a question.

> How are you arriving today?

A reader can describe experiences such as ecological uncertainty, disconnection from Nature, search for meaning or living through change. Kinship Library places that inquiry in the same semantic space as its books and suggests works whose descriptions resonate with the reader's words.

## Why Kinship Library?

Kinship Library grew from a personal and artistic inquiry into ecological grief, reconnection with Nature, embodied practice, philosophy and contemplative traditions.

The project begins from a simple proposition:

> We do not stand outside Nature. We belong to it.

Technology is used here not as an answer to ecological and existential questions, but as a tool for encountering different forms of knowledge.

## The collection

The final Kinship Library contains **1,121 book records** built through complementary layers:

* 713 records from the original NLP ready collection
* 25 creator curated foundational works
* 383 records from semantic expansion

The collection moves across five paths of knowledge:

* Humans and Nature
* Indigenous Knowledge of the Americas
* Yoga Philosophy
* Ayurveda and Indian Spirituality
* Yoga as a Way of Life

These paths support discovery but are not treated as rigid genres or definitive intellectual classifications.

## How it works

### 1. Collection and data quality

Book metadata was collected through bibliographic APIs and curated sources. The catalogue was cleaned, deduplicated and enriched with available descriptions and metadata.

The first processed catalogue contained 925 records. Books needed at least 15 words of semantic text to enter the initial NLP modeling stage, leaving 713 NLP eligible records.

### 2. NLP and semantic embeddings

TF IDF was used as an interpretable text baseline.

Sentence Transformer embeddings were then used to represent books semantically. The project uses `sentence-transformers/all-MiniLM-L6-v2`, producing a 384 dimensional representation for each modeled book.

### 3. Unsupervised learning

K Means was explored for values of K from 2 to 10. The Elbow Method and Silhouette Score were used to examine broad semantic structure.

The highest Silhouette score occurred at **K = 3**, with a score of approximately **0.098**. The relatively low score indicates substantial overlap between the clusters, which is consistent with the interdisciplinary character of the catalogue.

PCA was used to project the semantic embedding space into two dimensions for exploratory visualization.

### 4. Semantic recommendation

A reader's inquiry is embedded using the same Sentence Transformer model. Cosine similarity compares the inquiry with stored book embeddings and identifies semantically related records.

Kinship Library therefore supports discovery through meaning rather than only through keywords or predefined genres.

### 5. Multilayered expansion

The library was expanded beyond the first modeling collection through semantic discovery and creator curation. The final production catalogue combines 713 original records, 383 semantic expansion records and 25 creator curated works.

The final catalogue and embedding matrix preserve row alignment so each book record corresponds to its semantic vector.

### 6. SQL

SQLite was used to create a relational layer and explore books, themes, subjects and semantic clusters. The SQL schema and analysis queries are included in the `sql/` directory.

### 7. Streamlit application

The final interface brings the layers together in an interactive library. Readers can:

* Ask the library from their own words
* Explore books through semantic resonance
* Browse paths of knowledge
* Search by title or author
* Explore the creator curated collection
* Open an interactive **How this was created** journey with the project methodology, Elbow and Silhouette charts, PCA and SQL

## Future development

A second phase will explore connections and possible collaborations with verified public domain and open access digital libraries.

The aim is to allow readers, where legally and ethically possible, to move from discovering a book to accessing an open edition. This is particularly relevant for ancient philosophical texts, public domain works and forms of ancestral and embodied knowledge that can be responsibly shared through open digital collections.

Future access should clearly distinguish public domain, open access, digital lending and copyrighted materials. Kinship Library does not host or claim free access to copyrighted books.

## Technologies

Python · pandas · NumPy · scikit learn · Sentence Transformers · TF IDF · K Means · PCA · cosine similarity · SQLite · REST APIs · Streamlit · Plotly

## Project structure

```text
kinship-library/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_collection_data_quality_eda.ipynb
│   ├── 02_process_enriched_book_data.ipynb
│   ├── 03_nlp_embeddings_kmeans_recommender.ipynb
│   ├── 04_sql_final_data_streamlit_prep.ipynb
│   ├── 05_3_curated_library_expansion_final.ipynb
│   └── 06_final_files_curated_cover_audit.ipynb
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
├── requirements.txt
└── README.md
```

## Run the app

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app/app.py
```

## Creator

**Leidy Angélica Roa**

Bio artist · Yogi · Physicist

Kinship Library emerges from an interdisciplinary practice connecting embodied knowledge, ecology, science, contemplative traditions and more than human perspectives.

---

*Can technology help us find our way back into relationship?*
