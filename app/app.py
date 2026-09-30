from pathlib import Path
from urllib.parse import quote_plus
import html
import re

import requests

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Kinship Library",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BOOKS_FILE = PROJECT_ROOT / "data" / "processed" / "app_books.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "app_book_embeddings.npy"

THEME_LABELS = {
    "humans_and_nature": "Humans and Nature",
    "indigenous_knowledge_americas": "Indigenous Knowledge of the Americas",
    "yoga_philosophy": "Yoga Philosophy",
    "ayurveda_indian_spirituality": "Ayurveda and Indian Spirituality",
    "yoga_way_of_life": "Yoga as a Way of Life",
}

KNOWLEDGE_ORIENTATIONS = {
    "Open to all approaches": "",
    "Philosophical and contemplative": (
        "philosophical and contemplative knowledge, worldviews, meaning, consciousness, "
        "ethics, spiritual philosophy, reflection and ways of understanding existence"
    ),
    "Scientific and ecological": (
        "scientific and ecological knowledge, ecology, biology, environmental science, "
        "systems thinking, research, evidence and empirical perspectives"
    ),
    "Embodied and practical": (
        "embodied and practical knowledge, lived experience, somatic practice, movement, "
        "Yoga, meditation, Ayurveda, exercises, methods and everyday application"
    ),
    "Ancestral and relational": (
        "ancestral and relational knowledge, Indigenous knowledge, land based knowledge, "
        "reciprocity, community, traditional ecological knowledge and relationships "
        "with more than human worlds"
    ),
}

STARTING_QUESTIONS = {
    "🌱 Disconnection from Nature": "I feel disconnected from Nature and the living world. I want to recover a sense of belonging, kinship and relationship with plants, animals, land and ecosystems.",
    "🌍 Ecological uncertainty": "I feel anxious and uncertain about the ecological future. I am looking for perspectives on climate change, ecological grief, care, responsibility, hope and our relationship with the Earth.",
    "🕯 Search for meaning": "I am searching for meaning, purpose and a deeper understanding of life, consciousness and spiritual practice.",
    "🌀 Living through change": "I am going through a period of change and uncertainty. I am looking for wisdom about transformation, impermanence, courage and how to live with the unknown.",
}

st.markdown(
    """
    <style>
    :root {
        --forest: #173A2B;
        --moss: #557A46;
        --sage: #A9B99A;
        --cream: #F4F0E6;
        --paper: #FBF8F1;
        --earth: #8A6848;
        --charcoal: #26332D;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 5%, rgba(169,185,154,.24), transparent 28rem),
            linear-gradient(180deg, #F7F3E9 0%, #F4F0E6 55%, #EEE8DB 100%);
        color: var(--charcoal);
    }

    [data-testid="stHeader"] {
        background: rgba(244,240,230,.86);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3 {
        color: var(--forest);
        letter-spacing: -.02em;
    }

    .hero {
        padding: 4.7rem 3.5rem;
        border-radius: 30px;
        background:
            linear-gradient(120deg, rgba(17,52,37,.97), rgba(52,91,59,.88)),
            radial-gradient(circle at 80% 20%, rgba(169,185,154,.45), transparent 18rem);
        box-shadow: 0 22px 55px rgba(23,58,43,.16);
        margin-bottom: 2.2rem;
    }

    .eyebrow {
        color: #C8D5BE;
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .78rem;
        font-weight: 700;
    }

    .hero h1 {
        color: #FAF6EC;
        font-size: clamp(3rem, 7vw, 6.2rem);
        line-height: .95;
        margin: .5rem 0 .8rem;
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 500;
    }

    .hero .subtitle {
        color: #DCE5D4;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.5rem;
        font-style: italic;
        margin-bottom: 2rem;
    }

    .hero .manifesto {
        color: #F4F0E6;
        max-width: 850px;
        font-size: 1.08rem;
        line-height: 1.8;
    }

    .section-intro {
        max-width: 820px;
        color: #445149;
        font-size: 1.03rem;
        line-height: 1.75;
        margin-bottom: 1.3rem;
    }

    .path-note {
        font-size: .88rem;
        color: #657168;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,.72);
        border: 1px solid #BAC8B3;
        border-radius: 18px;
        color: var(--charcoal);
        min-height: 145px;
    }

    div[data-testid="stSelectbox"] > div > div {
        border-radius: 14px;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #557A46;
        background: #173A2B;
        color: #FAF6EC;
        font-weight: 650;
        padding: .6rem 1.15rem;
    }

    .stButton > button:hover {
        background: #2E5B3D;
        border-color: #2E5B3D;
        color: white;
    }

    .book-card {
        background: rgba(251,248,241,.92);
        border: 1px solid rgba(85,122,70,.18);
        border-radius: 22px;
        padding: 1.1rem;
        min-height: 100%;
        box-shadow: 0 10px 30px rgba(23,58,43,.08);
        margin-bottom: 1rem;
    }

    .cover-wrap {
        height: 310px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #E7EBDD;
        border-radius: 15px;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .cover-wrap img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .cover-placeholder {
        width: 100%;
        height: 100%;
        display:flex;
        align-items:center;
        justify-content:center;
        text-align:center;
        padding:1.5rem;
        color:#F4F0E6;
        background: linear-gradient(145deg,#173A2B,#557A46);
        font-family: Georgia, serif;
        font-size: 1.25rem;
    }

    .book-title {
        color: #173A2B;
        font-family: Georgia, serif;
        font-size: 1.22rem;
        line-height: 1.25;
        font-weight: 700;
        margin-bottom: .35rem;
    }

    .book-author {
        color: #667168;
        font-size: .92rem;
        margin-bottom: .8rem;
    }

    .tag {
        display: inline-block;
        background: #E3E9DC;
        color: #31523B;
        border-radius: 999px;
        padding: .28rem .62rem;
        font-size: .75rem;
        margin-bottom: .8rem;
    }

    .book-description {
        color: #39463F;
        line-height: 1.55;
        font-size: .91rem;
    }

    .isbn {
        color: #718077;
        font-size: .78rem;
        margin-top: .8rem;
    }

    .creator {
        margin-top: 3rem;
        padding: 2.5rem;
        border-radius: 25px;
        background: #173A2B;
        color: #F4F0E6;
    }

    .creator h2 {
        color: #F4F0E6;
        font-family: Georgia, serif;
    }

    .creator strong {
        color: #D9E5D0;
    }

    .footer-line {
        text-align:center;
        color:#68756C;
        font-family: Georgia, serif;
        font-style: italic;
        margin-top: 2.4rem;
    }

    @media (max-width: 700px) {
        .hero { padding: 3rem 1.5rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .cover-wrap { height: 360px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_books():
    books = pd.read_csv(BOOKS_FILE)
    return books


@st.cache_resource
def load_embeddings():
    return np.load(EMBEDDINGS_FILE)


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(ttl=86400, show_spinner=False)
def cover_exists(url):
    if not url:
        return False
    try:
        response = requests.get(url, timeout=4, stream=True)
        content_type = response.headers.get("content-type", "")
        return response.status_code == 200 and content_type.startswith("image/")
    except requests.RequestException:
        return False


def normalize_title(title):
    title = safe_text(title).lower()
    title = re.sub(r"^the\s+", "", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def safe_text(value, fallback=""):
    if pd.isna(value):
        return fallback
    return str(value).strip()


def shorten(text, limit=420):
    text = safe_text(text)
    if not text:
        return "Description unavailable in the current catalogue."
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def recommend(query, books, embeddings, model, n=6, theme=None, orientation=None):
    orientation_text = KNOWLEDGE_ORIENTATIONS.get(orientation, "")
    enriched_query = query
    if orientation_text:
        enriched_query = (
            f"{query}\n\nThe reader would especially like to encounter: {orientation_text}."
        )

    query_embedding = model.encode(
        [enriched_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    similarities = embeddings @ query_embedding
    eligible = np.ones(len(books), dtype=bool)

    if theme and theme != "All paths":
        eligible = books["collection_theme"].eq(theme).to_numpy()

    candidate_indices = np.where(eligible)[0]
    candidate_scores = similarities[candidate_indices]
    ranked_indices = candidate_indices[np.argsort(candidate_scores)[::-1]]

    selected_indices = []
    seen_titles = set()

    for idx in ranked_indices:
        title_key = normalize_title(books.iloc[idx].get("title"))
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        selected_indices.append(idx)
        if len(selected_indices) == n:
            break

    results = books.iloc[selected_indices].copy()
    results["similarity"] = similarities[selected_indices]
    return results


def render_book_card(row):
    title = safe_text(row.get("title"), "Untitled")
    author = safe_text(row.get("author"), "Author unavailable")
    description = shorten(row.get("description_final"))
    theme_code = safe_text(row.get("collection_theme"))
    theme = THEME_LABELS.get(theme_code, theme_code.replace("_", " ").title())
    isbn = safe_text(row.get("isbn_clean"))
    cover = safe_text(row.get("cover_url"))

    if cover and cover_exists(cover):
        cover_html = (
            f'<img src="{html.escape(cover)}" '
            f'alt="Cover of {html.escape(title)}" '
            'onerror="this.style.display=\'none\'; '
            'this.nextElementSibling.style.display=\'flex\';">'
            f'<div class="cover-placeholder" style="display:none;">'
            f'🌿<br>{html.escape(title)}</div>'
        )
    else:
        cover_html = (
            f'<div class="cover-placeholder">🌿<br>{html.escape(title)}</div>'
        )

    isbn_html = f'<div class="isbn">ISBN {html.escape(isbn)}</div>' if isbn else ""

    st.markdown(
        f"""
        <div class="book-card">
            <div class="cover-wrap">{cover_html}</div>
            <div class="book-title">{html.escape(title)}</div>
            <div class="book-author">{html.escape(author)}</div>
            <div class="tag">{html.escape(theme)}</div>
            <div class="book-description">{html.escape(description)}</div>
            {isbn_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if isbn:
        search_url = "https://www.google.com/search?q=" + quote_plus(f'ISBN {isbn} "{title}"')
    else:
        search_url = "https://www.google.com/search?q=" + quote_plus(f'"{title}" "{author}" book')

    st.link_button("Find this book ↗", search_url, use_container_width=True)

    with st.expander("Why this recommendation?"):
        st.write(
            "This book appeared because its stored metadata is semantically "
            "similar to the inquiry you entered. The score is a model similarity "
            "measure, not a judgment of the book or a measure of how helpful it will be."
        )
        st.caption(f"Semantic similarity: {row['similarity']:.3f}")


try:
    books = load_books()
    embeddings = load_embeddings()

    if len(books) != embeddings.shape[0]:
        st.error("The book catalogue and embedding file are not aligned.")
        st.stop()
except Exception as exc:
    st.error(f"Kinship Library could not load its catalogue: {exc}")
    st.stop()


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">A semantic library for living in connection</div>
        <h1>Kinship Library</h1>
        <div class="subtitle">Books for Living in Connection</div>
        <div class="manifesto">
            We live in a time of ecological uncertainty, accelerated change and
            increasing distance from the living systems that sustain us. Questions
            about belonging, meaning, our relationship with the Earth and how to
            live well within a more than human world are becoming increasingly urgent.
            <br><br>
            Kinship Library begins from a simple proposition: perhaps some of the
            knowledge we need can be encountered by remembering that we do not stand
            outside Nature, but belong to it.
            <br><br>
            Bringing ecology, Indigenous knowledge, Yoga philosophy, Ayurveda and
            spiritual traditions into proximity, this library invites you to begin
            not with a category, but with a question.
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.header("What are you experiencing, questioning or seeking?")
st.markdown(
    '<div class="section-intro">'
    "Describe what is present for you in your own words. Kinship Library will look "
    "for books whose themes and descriptions resonate semantically with your inquiry."
    "</div>",
    unsafe_allow_html=True,
)

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

st.caption("Or begin with a question")
prompt_cols = st.columns(4)
for col, (label, prompt) in zip(prompt_cols, STARTING_QUESTIONS.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.query_text = prompt
            st.rerun()

query = st.text_area(
    "Your inquiry",
    key="query_text",
    placeholder=(
        "For example: I feel disconnected from Nature and I am looking for "
        "different ways of understanding belonging, reciprocity and our "
        "relationship with the living world."
    ),
    label_visibility="collapsed",
)

st.markdown("#### What kind of knowledge would you like to encounter?")
selected_orientation = st.radio(
    "Knowledge orientation",
    list(KNOWLEDGE_ORIENTATIONS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

control_a, control_b, control_c = st.columns([2.2, 1, 1])
with control_a:
    theme_options = ["All paths"] + list(THEME_LABELS.keys())
    selected_theme = st.selectbox(
        "Optional collection path",
        theme_options,
        format_func=lambda x: "All knowledge paths" if x == "All paths" else THEME_LABELS[x],
    )
with control_b:
    number_books = st.selectbox("Books to show", [3, 6, 9], index=1)
with control_c:
    st.write("")
    st.write("")
    explore = st.button("Explore the Library", type="primary", use_container_width=True)

st.markdown(
    '<div class="path-note">The knowledge paths describe how books entered the '
    "collection. They are not definitive genres or classifications.</div>",
    unsafe_allow_html=True,
)

if explore:
    if not query.strip():
        st.warning("Write an inquiry or choose one of the starting questions first.")
    else:
        with st.spinner("Listening for resonances in the library…"):
            model = load_model()
            results = recommend(
                query=query,
                books=books,
                embeddings=embeddings,
                model=model,
                n=number_books,
                theme=None if selected_theme == "All paths" else selected_theme,
                orientation=selected_orientation,
            )

        st.divider()
        st.header("Books that may resonate with your inquiry")
        st.markdown(
            '<div class="section-intro">'
            "These are invitations to explore rather than answers. The recommendations "
            "are based on semantic relationships between your words and the book metadata "
            "available in the Kinship Library catalogue. Your selected knowledge orientation "
            "gently steers the semantic search rather than assigning books to a fixed category."
            "</div>",
            unsafe_allow_html=True,
        )

        for start in range(0, len(results), 3):
            cols = st.columns(3)
            chunk = results.iloc[start:start + 3]
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    render_book_card(row)

st.divider()
st.header("Browse paths of knowledge")
st.markdown(
    '<div class="section-intro">'
    "Kinship Library was assembled through five thematic search routes. These paths "
    "bring different bodies of knowledge into conversation without suggesting that "
    "they are interchangeable or share a single worldview."
    "</div>",
    unsafe_allow_html=True,
)

path_cols = st.columns(5)
path_icons = ["🌿", "🌎", "🕉️", "🌾", "🧘"]
for col, icon, (code, label) in zip(path_cols, path_icons, THEME_LABELS.items()):
    with col:
        count = int((books["collection_theme"] == code).sum())
        st.markdown(f"### {icon}")
        st.markdown(f"**{label}**")
        st.caption(f"{count} books")

st.divider()
st.header("About the project")
st.markdown(
    """
    Kinship Library is an exploratory semantic book recommender. Its catalogue was
    assembled through thematic book searches and enriched with available metadata.
    Books with sufficient text were represented using Sentence Transformer embeddings.
    When you enter an inquiry, the same model represents your words in a 384 dimensional
    semantic space and compares them with the stored book embeddings.

    The project also uses TF IDF as a baseline, K Means and PCA to explore the semantic
    structure of the catalogue, SQL for the relational data layer, and Streamlit for
    this interface.

    The recommendations are based on metadata rather than full book texts. Collection
    paths are retrieval routes rather than ground truth classifications, metadata
    coverage is uneven, and semantic similarity does not establish the quality,
    authority or suitability of a book. Kinship Library is designed for exploration
    and is not medical, psychological or spiritual advice.
    """
)

st.markdown(
    """
    <section class="creator">
        <div class="eyebrow">About the Creator</div>
        <h2>Leidy Angélica Roa</h2>
        <p><strong>Bio artist · Yogi · Physicist</strong></p>
        <p>
            Leidy Angélica Roa is a Colombian bio artist, choreographer, yogi and
            physicist based in Berlin. Her interdisciplinary practice moves between
            embodied knowledge, ecology, science and contemplative traditions,
            exploring how humans understand themselves as participants in a living world.
        </p>
        <p>
            With a background in Physics and Dance, her work brings scientific inquiry
            into conversation with ecosomatic practice, Yoga philosophy, ancestral
            knowledge and more than human perspectives.
        </p>
        <p>
            <strong>Kinship Library</strong> emerged from this intersection as an
            experiment in using data science and semantic technologies not simply to
            organize information, but to help people encounter books that may open
            different ways of thinking about belonging, meaning and relationship
            with the Earth.
        </p>
    </section>
    <div class="footer-line">
        Kinship Library was created from a question: Can technology help us find
        our way back into relationship?
    </div>
    """,
    unsafe_allow_html=True,
)
