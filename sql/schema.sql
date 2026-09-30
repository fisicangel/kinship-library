CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    source_id TEXT,
    title TEXT,
    author TEXT,
    subjects TEXT,
    published_year REAL,
    isbn TEXT,
    language TEXT,
    average_rating REAL,
    ratings_count REAL,
    collection_theme TEXT,
    search_query TEXT,
    description_final TEXT,
    google_categories TEXT,
    semantic_text TEXT,
    semantic_text_words INTEGER,
    nlp_text_quality TEXT,
    cluster INTEGER,
    pca_1 REAL,
    pca_2 REAL
);

CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    author_name TEXT
);

CREATE TABLE book_authors (
    book_id INTEGER,
    author_id INTEGER,
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

CREATE TABLE subjects (
    subject_id INTEGER PRIMARY KEY,
    subject_name TEXT
);

CREATE TABLE book_subjects (
    book_id INTEGER,
    subject_id INTEGER,
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);