-- Books by collection theme
SELECT
    collection_theme,
    COUNT(*) AS number_of_books
FROM books
GROUP BY collection_theme
ORDER BY number_of_books DESC;

-- Books by semantic cluster
SELECT
    cluster,
    COUNT(*) AS number_of_books,
    ROUND(AVG(semantic_text_words), 1) AS avg_text_words
FROM books
GROUP BY cluster
ORDER BY cluster;

-- Most represented authors
SELECT
    a.author_name,
    COUNT(DISTINCT ba.book_id) AS number_of_books
FROM authors AS a
JOIN book_authors AS ba
    ON a.author_id = ba.author_id
GROUP BY a.author_id, a.author_name
ORDER BY number_of_books DESC, a.author_name
LIMIT 20;

-- Most represented subjects
SELECT
    s.subject_name,
    COUNT(DISTINCT bs.book_id) AS number_of_books
FROM subjects AS s
JOIN book_subjects AS bs
    ON s.subject_id = bs.subject_id
GROUP BY s.subject_id, s.subject_name
ORDER BY number_of_books DESC, s.subject_name
LIMIT 25;

-- Collection themes inside semantic clusters
SELECT
    cluster,
    collection_theme,
    COUNT(*) AS number_of_books
FROM books
GROUP BY cluster, collection_theme
ORDER BY cluster, number_of_books DESC;