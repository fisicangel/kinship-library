from pathlib import Path
from urllib.parse import quote_plus
import html
import re

import requests

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="Kinship Library",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BOOKS_FILE = PROJECT_ROOT / "data" / "processed" / "app_books_expanded.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "app_book_embeddings_expanded.npy"

THEME_LABELS = {
    "humans_and_nature": "Humans and Nature",
    "indigenous_knowledge_americas": "Indigenous Knowledge of the Americas",
    "yoga_philosophy": "Yoga Philosophy",
    "ayurveda_indian_spirituality": "Ayurveda and Indian Spirituality",
    "yoga_way_of_life": "Yoga as a Way of Life",
}

THEME_SYMBOLS = {
    "humans_and_nature": "❧",
    "indigenous_knowledge_americas": "◉",
    "yoga_philosophy": "ॐ",
    "ayurveda_indian_spirituality": "✦",
    "yoga_way_of_life": "☼",
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

PATH_DIRECTIONS = {
    "humans_and_nature": {
        "Closest resonances": "",
        "Ecology and living systems": "ecology, living systems, biodiversity, environmental relationships and interdependence",
        "Belonging and more than human relations": "belonging, kinship, reciprocity and relationships with more than human worlds",
        "Environmental philosophy": "environmental philosophy, ethics, worldviews and ways of understanding human relationships with Nature",
        "Care in ecological uncertainty": "ecological uncertainty, grief, care, responsibility, resilience and ways of living through environmental change",
    },
    "indigenous_knowledge_americas": {
        "Closest resonances": "",
        "Land and reciprocity": "land based knowledge, reciprocity, territory, community and relationships with the living world",
        "Ancestral knowledge and cosmology": "ancestral knowledge, cosmology, memory, oral traditions and Indigenous ways of knowing",
        "Decolonial perspectives": "decolonial thought, coloniality, resistance, Indigenous sovereignty and critiques of colonial knowledge systems",
        "Traditional ecological knowledge": "traditional ecological knowledge, plants, animals, ecosystems, stewardship and intergenerational knowledge",
    },
    "yoga_philosophy": {
        "Closest resonances": "",
        "Consciousness and the self": "consciousness, self knowledge, mind, awareness, liberation and the nature of reality",
        "Classical Yoga philosophy": "Yoga Sutras, classical Yoga philosophy, Patanjali, ethics, meditation and liberation",
        "Indian philosophical traditions": "Indian philosophy, Vedanta, Upanishads, Bhagavad Gita and philosophical inquiry",
        "Practice and contemplation": "meditation, contemplation, spiritual practice and the relationship between philosophy and lived experience",
    },
    "ayurveda_indian_spirituality": {
        "Closest resonances": "",
        "Ayurveda and ways of living": "Ayurveda, daily life, balance, food, health traditions and embodied ways of living",
        "Indian spiritual traditions": "Indian spirituality, devotional traditions, sacred texts, deities and spiritual philosophy",
        "Body, mind and consciousness": "body, mind, consciousness, subtle body, embodied knowledge and holistic traditions",
        "Practice and ritual": "spiritual practice, ritual, meditation, devotion and everyday contemplative life",
    },
    "yoga_way_of_life": {
        "Closest resonances": "",
        "Practice and embodiment": "Yoga practice, embodiment, asana, pranayama, breath, meditation and lived experience",
        "Philosophy behind the practice": "Yoga philosophy, ethics, meaning, consciousness and philosophical foundations of practice",
        "Breath and meditation": "breath, pranayama, meditation, attention, nervous system and contemplative practice",
        "Yoga in everyday life": "Yoga as a way of life, discipline, self inquiry, transformation and everyday application",
    },
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

    [data-testid="stSidebar"] {
        background: #E8E8D8;
        border-right: 1px solid rgba(85,122,70,.16);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #173A2B;
    }

    .path-symbol {
        color: #557A46;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.7rem;
        line-height: 1;
        margin-right: .35rem;
    }

    .sidebar-book {
        padding: .65rem 0;
        border-bottom: 1px solid rgba(85,122,70,.16);
    }

    .sidebar-book-title {
        color: #173A2B;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1rem;
        font-weight: 700;
    }

    .sidebar-book-author {
        color: #667168;
        font-size: .82rem;
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
        max-width: 860px;
        color: #445149;
        font-size: 1.06rem;
        line-height: 1.75;
        margin-bottom: 1.3rem;
    }

    .path-note {
        font-size: .92rem;
        color: #657168;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,.72);
        border: 1px solid #BAC8B3;
        border-radius: 18px;
        color: var(--charcoal);
        min-height: 88px;\n        max-height: 150px;
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
        line-height: 1.58;
        font-size: .94rem;
    }

    .isbn {
        color: #718077;
        font-size: .82rem;
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

    div[data-baseweb="tab-list"] {
        gap: .45rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        background: rgba(227,233,220,.72);
        border: 1px solid rgba(85,122,70,.22);
        border-radius: 999px;
        padding: .58rem .9rem;
        min-height: 2.8rem;
        font-size: .98rem;
        transition: transform .18s ease, background .18s ease;
    }

    button[data-baseweb="tab"]:hover {
        background: #D8E2CF;
        transform: translateY(-1px);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #315B3E;
        color: #FAF6EC;
        border-color: #315B3E;
    }

    .journey-path {
        display:flex;
        align-items:center;
        justify-content:center;
        gap:.55rem;
        flex-wrap:wrap;
        margin:1.4rem 0 1.7rem;
    }

    .journey-node {
        padding:.72rem 1rem;
        border:1px solid rgba(85,122,70,.28);
        border-radius:999px;
        background:linear-gradient(145deg,#F8F4E9,#E3E9DC);
        color:#173A2B;
        font-family:Georgia, serif;
        font-weight:700;
        box-shadow:0 7px 18px rgba(23,58,43,.07);
    }

    .journey-arrow {
        color:#8A6848;
        font-size:1.25rem;
    }

    .engineering-note {
        padding:1.2rem 1.35rem;
        border-left:4px solid #8A6848;
        border-radius:0 16px 16px 0;
        background:rgba(231,235,221,.72);
        color:#26332D;
        margin:1rem 0;
    }

    .future-tree {
        margin-top:1.2rem;
        padding:2.2rem 1.5rem;
        border-radius:26px;
        text-align:center;
        color:#F4F0E6;
        background:
            radial-gradient(circle at 50% 110%, rgba(169,185,154,.55), transparent 35%),
            linear-gradient(145deg,#173A2B,#315B3E);
    }

    .future-tree .crown {
        font-size:2.6rem;
        letter-spacing:.18em;
        margin-bottom:.6rem;
    }

    .future-tree .branch {
        display:inline-block;
        margin:.35rem;
        padding:.55rem .85rem;
        border:1px solid rgba(244,240,230,.34);
        border-radius:999px;
        background:rgba(244,240,230,.08);
    }

    .future-tree .trunk {
        width:2px;
        height:38px;
        margin:.8rem auto;
        background:#A9B99A;
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

def normalize_search_text(value):
    value = safe_text(value).casefold()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def flexible_text_match(value, query):
    value_norm = normalize_search_text(value)
    query_norm = normalize_search_text(query)
    if not query_norm:
        return True
    terms = query_norm.split()
    return all(term in value_norm.split() for term in terms)


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


def recommend(query, books, embeddings, model, n=6, theme=None, orientation=None, prioritize_curated=False):
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
    ranking_scores = similarities.copy()

    if prioritize_curated and "is_curated" in books.columns:
        curated_mask = books["is_curated"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        ranking_scores = ranking_scores + np.where(curated_mask, 0.08, 0.0)

    eligible = np.ones(len(books), dtype=bool)

    if theme and theme != "All paths":
        eligible = books["collection_theme"].eq(theme).to_numpy()

    candidate_indices = np.where(eligible)[0]
    candidate_scores = ranking_scores[candidate_indices]
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


def resonance_recommendations(book_index, direction, books, embeddings, model, n=3):
    selected = books.iloc[book_index]
    theme_code = safe_text(selected.get("collection_theme"))
    selected_title = normalize_title(selected.get("title"))
    base_embedding = embeddings[book_index]

    direction_text = PATH_DIRECTIONS.get(theme_code, {}).get(direction, "")
    if direction_text:
        direction_embedding = model.encode(
            [direction_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        query_embedding = (0.72 * base_embedding) + (0.28 * direction_embedding)
        norm = np.linalg.norm(query_embedding)
        if norm:
            query_embedding = query_embedding / norm
    else:
        query_embedding = base_embedding

    similarities = embeddings @ query_embedding
    eligible = books["collection_theme"].eq(theme_code).to_numpy()
    ranked_indices = np.where(eligible)[0]
    ranked_indices = ranked_indices[np.argsort(similarities[ranked_indices])[::-1]]

    selected_indices = []
    seen_titles = {selected_title}
    for idx in ranked_indices:
        title_key = normalize_title(books.iloc[idx].get("title"))
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        selected_indices.append(idx)
        if len(selected_indices) == n:
            break

    return books.iloc[selected_indices].copy()


def render_book_card(row):
    title = safe_text(row.get("title"), "Untitled")
    author = safe_text(row.get("author"), "Author unavailable")
    description = shorten(row.get("description_final"))
    theme_code = safe_text(row.get("collection_theme"))
    theme = THEME_LABELS.get(theme_code, theme_code.replace("_", " ").title())
    isbn = safe_text(row.get("isbn_clean"))
    cover = safe_text(row.get("cover_url"))
    is_curated = safe_text(row.get("is_curated")).lower() in ["true", "1"]
    curated_html = '<div class="tag">Curated Kinship</div>' if is_curated else ""

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
    card_html = (
        '<div class="book-card">'
        f'<div class="cover-wrap">{cover_html}</div>'
        f'<div class="book-title">{html.escape(title)}</div>'
        f'<div class="book-author">{html.escape(author)}</div>'
        f'<div class="tag">{html.escape(theme)}</div>'
        f'{curated_html}'
        f'<div class="book-description">{html.escape(description)}</div>'
        f'{isbn_html}'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if isbn:
        search_url = "https://www.google.com/search?q=" + quote_plus(f'ISBN {isbn} "{title}"')
    else:
        search_url = "https://www.google.com/search?q=" + quote_plus(f'"{title}" "{author}" book')

    st.link_button("Find externally ↗", search_url, use_container_width=True)

try:
    books = load_books()
    embeddings = load_embeddings()

    if len(books) != embeddings.shape[0]:
        st.error("The book catalogue and embedding file are not aligned.")
        st.stop()
except Exception as exc:
    st.error(f"Kinship Library could not load its catalogue: {exc}")
    st.stop()


with st.sidebar:
    st.markdown("## Other ways to Browse the Library")
    st.caption(
        "Already have a path, author or title in mind? Use these alternative ways to move through the collection."
    )

    discovery_mode = st.radio(
        "Explore by",
        ["Path of knowledge", "Author", "Title"],
        horizontal=False,
        key="sidebar_discovery_mode",
    )

    if discovery_mode in ["Title", "Author"]:
        field = "title" if discovery_mode == "Title" else "author"
        placeholder = (
            "For example: Staying with the Trouble…"
            if discovery_mode == "Title"
            else "For example: Donna Haraway…"
        )
        browse_query = st.text_input(
            discovery_mode,
            placeholder=placeholder,
            key=f"browse_{field}_query",
        )

        if browse_query.strip():
            browse_books = books[
                books[field].apply(lambda value: flexible_text_match(value, browse_query))
            ].copy()

            browse_books = browse_books.sort_values(
                by="title",
                key=lambda s: s.fillna("").str.lower(),
            )

            st.caption(f"{len(browse_books)} matching books")

            if len(browse_books) == 0:
                st.info(f"No {discovery_mode.lower()} matched that search.")
            else:
                for _, browse_row in browse_books.head(8).iterrows():
                    browse_title = safe_text(browse_row.get("title"), "Untitled")
                    browse_author = safe_text(browse_row.get("author"), "Author unavailable")
                    browse_isbn = safe_text(browse_row.get("isbn_clean"))
                    browse_theme_code = safe_text(browse_row.get("collection_theme"))
                    browse_theme_label = THEME_LABELS.get(
                        browse_theme_code,
                        browse_theme_code.replace("_", " ").title(),
                    )
                    browse_symbol = THEME_SYMBOLS.get(browse_theme_code, "❧")

                    if browse_isbn:
                        browse_url = (
                            "https://www.google.com/search?q="
                            + quote_plus(f'ISBN {browse_isbn} "{browse_title}"')
                        )
                    else:
                        browse_url = (
                            "https://www.google.com/search?q="
                            + quote_plus(f'"{browse_title}" "{browse_author}" book')
                        )

                    st.markdown(
                        f"""
                        <div class="sidebar-book">
                            <div class="sidebar-book-title">{html.escape(browse_title)}</div>
                            <div class="sidebar-book-author">{html.escape(browse_author)}</div>
                            <div class="path-note">{browse_symbol} {html.escape(browse_theme_label)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "Explore from this book →",
                        key=f"resonate_{field}_{browse_row.name}",
                        use_container_width=True,
                    ):
                        st.session_state.resonance_book_index = int(browse_row.name)
                        st.session_state.resonance_direction = "Closest resonances"
                        st.rerun()

                if len(browse_books) > 8:
                    st.caption("Showing the first 8 matches. Add another word to narrow the search.")

    else:
        browse_theme = st.selectbox(
            "Path of knowledge",
            list(THEME_LABELS.keys()),
            format_func=lambda x: f"{THEME_SYMBOLS[x]}  {THEME_LABELS[x]}",
            key="browse_theme",
        )

        path_books = books[
            books["collection_theme"].eq(browse_theme)
        ].copy().sort_values(
            by="title",
            key=lambda s: s.fillna("").str.lower(),
        )

        st.caption(f"{len(path_books)} books in {THEME_LABELS[browse_theme]}")
        st.caption(
            "Use this path as a doorway into related books. The main semantic inquiry below remains the best way to receive recommendations."
        )

        for _, browse_row in path_books.head(8).iterrows():
            browse_title = safe_text(browse_row.get("title"), "Untitled")
            browse_author = safe_text(browse_row.get("author"), "Author unavailable")
            browse_isbn = safe_text(browse_row.get("isbn_clean"))

            if browse_isbn:
                browse_url = (
                    "https://www.google.com/search?q="
                    + quote_plus(f'ISBN {browse_isbn} "{browse_title}"')
                )
            else:
                browse_url = (
                    "https://www.google.com/search?q="
                    + quote_plus(f'"{browse_title}" "{browse_author}" book')
                )

            st.markdown(
                f"""
                <div class="sidebar-book">
                    <div class="sidebar-book-title">{html.escape(browse_title)}</div>
                    <div class="sidebar-book-author">{html.escape(browse_author)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Explore from this book →",
                key=f"resonate_path_{browse_row.name}",
                use_container_width=True,
            ):
                st.session_state.resonance_book_index = int(browse_row.name)
                st.session_state.resonance_direction = "Closest resonances"
                st.rerun()


if "resonance_book_index" in st.session_state:
    selected_idx = st.session_state.resonance_book_index
    if 0 <= selected_idx < len(books):
        selected_book = books.iloc[selected_idx]
        selected_title = safe_text(selected_book.get("title"), "Untitled")
        selected_author = safe_text(selected_book.get("author"), "Author unavailable")
        selected_theme = safe_text(selected_book.get("collection_theme"))
        selected_theme_label = THEME_LABELS.get(
            selected_theme, selected_theme.replace("_", " ").title()
        )
        selected_symbol = THEME_SYMBOLS.get(selected_theme, "❧")

        st.markdown("## Follow a thread of resonance")
        st.markdown(
            f"Begin with **{selected_title}** by {selected_author}. "
            f"This book entered Kinship Library through **{selected_symbol} {selected_theme_label}**. "
            "Choose a direction and the library will find three semantic resonances within this path."
        )

        direction_options = list(
            PATH_DIRECTIONS.get(selected_theme, {"Closest resonances": ""}).keys()
        )
        current_direction = st.session_state.get(
            "resonance_direction", direction_options[0]
        )
        if current_direction not in direction_options:
            current_direction = direction_options[0]

        chosen_direction = st.selectbox(
            "Where would you like to go from here?",
            direction_options,
            index=direction_options.index(current_direction),
            key="resonance_direction_select",
        )
        st.session_state.resonance_direction = chosen_direction

        resonance_cols = st.columns([1, 1])
        with resonance_cols[0]:
            follow = st.button(
                "Find 3 books in resonance",
                type="primary",
                use_container_width=True,
            )
        with resonance_cols[1]:
            if st.button("Close this thread", use_container_width=True):
                st.session_state.pop("resonance_book_index", None)
                st.session_state.pop("resonance_direction", None)
                st.rerun()

        if follow:
            with st.spinner("Following this thread through the library…"):
                model = load_model()
                resonance_results = resonance_recommendations(
                    selected_idx,
                    chosen_direction,
                    books,
                    embeddings,
                    model,
                    n=3,
                )

            result_cols = st.columns(3)
            for col, (_, resonance_row) in zip(
                result_cols, resonance_results.iterrows()
            ):
                with col:
                    render_book_card(resonance_row)

        st.divider()


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

st.caption(f"Explore {len(books):,} book records across five paths of knowledge, including a 25 book creator curated collection.")

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
    height=90,
    placeholder=(
        "For example: I feel disconnected from Nature and I am looking for "
        "different ways of understanding belonging, reciprocity and our "
        "relationship with the living world."
    ),
    label_visibility="collapsed",
)

st.caption("Write freely. You do not need to press Ctrl + Enter. When your inquiry is ready, choose the options below and select Explore the Library.")

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

prioritize_curated = st.toggle(
    "Prioritize the curated Kinship collection",
    value=False,
    help="Gives a gentle additional weight to 25 books selected by the creator while keeping the full library in the search.",
)

if prioritize_curated:
    st.caption("Curated Kinship gently prioritizes a small creator selected collection. Semantic relevance remains part of every recommendation.")


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
                prioritize_curated=prioritize_curated,
            )

            wider_results = recommend(
                query=query,
                books=books,
                embeddings=embeddings,
                model=model,
                n=20,
                theme=None,
                orientation=selected_orientation,
                prioritize_curated=prioritize_curated,
            )

            chosen_titles = {
                normalize_title(title)
                for title in results["title"].tolist()
            }

            if selected_theme != "All paths":
                wider_results = wider_results[
                    ~wider_results["collection_theme"].eq(selected_theme)
                ]

            wider_results = wider_results[
                ~wider_results["title"].apply(normalize_title).isin(chosen_titles)
            ].head(3)

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

        if len(wider_results) > 0:
            st.markdown("### You might also like")
            st.markdown(
                '<div class="section-intro">'
                "A few additional resonances from the wider library, beyond the "
                "collection path you selected."
                "</div>",
                unsafe_allow_html=True,
            )
            extra_cols = st.columns(3)
            for col, (_, row) in zip(extra_cols, wider_results.iterrows()):
                with col:
                    render_book_card(row)

st.divider()
st.header("Other ways to Browse the Library")
st.markdown(
    '<div class="section-intro">'
    "Your inquiry is the main doorway into Kinship Library. You can also move through "
    "the collection by <strong>Paths of Knowledge</strong>, <strong>Author</strong> or "
    "<strong>Title</strong>. Open the sidebar to use these other ways of browsing."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("### Paths of Knowledge")
st.markdown(
    '<div class="section-intro">'
    "These five thematic search routes bring different bodies of knowledge into "
    "conversation without suggesting that they are interchangeable or share a single worldview."
    "</div>",
    unsafe_allow_html=True,
)

path_cols = st.columns(5)
for col, (code, label) in zip(path_cols, THEME_LABELS.items()):
    with col:
        count = int((books["collection_theme"] == code).sum())
        symbol = THEME_SYMBOLS[code]
        st.markdown(
            f'<div class="path-symbol">{symbol}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{label}**")
        st.caption(f"{count} books")

st.caption(
    "Open the sidebar to browse by Path of Knowledge, Author or Title."
)

st.divider()
st.header("About the project")
st.markdown(
    """
    Kinship Library is an exploratory semantic book recommender. Its expanded catalogue contains 1,121 book records combining the original thematic collection, semantic expansion and a 25 book creator curated collection. Available metadata was enriched where possible.
    Books with sufficient text were represented using Sentence Transformer embeddings.
    When you enter an inquiry, the same model represents your words in a 384 dimensional
    semantic space and compares them with the stored book embeddings.

    The project also uses TF IDF as a baseline, K Means and PCA to explore the semantic
    structure of the catalogue, SQL for the relational data layer, and Streamlit for
    this interface.

    Curated Kinship is optional. When activated, it gives a gentle ranking boost to the creator curated collection while continuing to search the full catalogue.

    The recommendations are based on metadata rather than full book texts. Collection
    paths are retrieval routes rather than ground truth classifications, metadata
    coverage is uneven, and semantic similarity does not establish the quality,
    authority or suitability of a book. Kinship Library is designed for exploration
    and is not medical, psychological or spiritual advice.
    """
)

st.divider()
st.header("How this was created")
st.markdown(
    '<div class="section-intro">'
    "From a personal question to a multilayered semantic library. Explore the stages "
    "that shaped Kinship Library, from collection and enrichment to NLP, clustering, "
    "SQL and the final interactive encounter."
    "</div>",
    unsafe_allow_html=True,
)

journey_tabs = st.tabs([
    "01 · Question",
    "02 · Collect",
    "03 · Enrich",
    "04 · Understand",
    "05 · Connect",
    "06 · Structure",
    "07 · What comes next",
])

with journey_tabs[0]:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown("### From ecological anxiety to a question")
    st.markdown(
        """
        Kinship Library grew from a period of ecological grief and uncertainty.

        Artistic practice, time in natural spaces, reading, Yoga and meditation
        gradually opened another possibility: rather than imagining humans outside
        Nature, what changes when we remember ourselves as participants in a living world?

        **The project begins here:** can technology help us encounter forms of knowledge
        that support reconnection, reflection and responsible action?
        """
    )
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

with journey_tabs[1]:
    st.markdown("### A multilayered collection")
    source_counts = pd.DataFrame({
        "Layer": [
            "Original NLP ready collection",
            "Semantic expansion",
            "Creator curated collection",
        ],
        "Records": [713, 383, 25],
    })
    source_fig = px.bar(
        source_counts,
        x="Layer",
        y="Records",
        text="Records",
        title="How the final 1,121 book records were assembled",
    )
    source_fig.update_traces(marker_color=["#557A46", "#A9B99A", "#B47B4D"])
    source_fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Book records",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(source_fig, use_container_width=True)
    st.caption(
        "The final catalogue combines API discovery, semantic screening and a 25 book "
        "creator curated collection. The paths of knowledge describe how books entered "
        "the library rather than fixed genres."
    )

with journey_tabs[2]:
    st.markdown("### From metadata to semantic text")
    st.markdown(
        """
        The first processed catalogue contained **925 records**. Descriptions and other
        available metadata were combined into semantic text. Books needed at least
        **15 words of semantic text** to enter the first NLP modeling stage, leaving
        **713 NLP eligible records**.

        **Title + author + subjects + description → semantic text → embedding**

        TF IDF provided an interpretable word based baseline. Sentence Transformers
        then represented meaning in a **384 dimensional semantic space**.
        """
    )
    quality_fig = go.Figure(
        go.Funnel(
            y=["Processed catalogue", "NLP eligible", "Final expanded catalogue"],
            x=[925, 713, 1121],
            textinfo="value+percent initial",
        )
    )
    quality_fig.update_traces(
        marker={"color": ["#315B3E", "#7F956B", "#B47B4D"]}
    )
    quality_fig.update_layout(
        title="The catalogue changed as quality checks and expansion were applied",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(quality_fig, use_container_width=True)

with journey_tabs[3]:
    st.markdown("### Exploring semantic structure")
    k_results = pd.DataFrame({
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "inertia": [
            481.535400, 451.286560, 438.315247, 425.636902, 418.192017,
            412.426758, 407.518433, 401.211487, 397.608276,
        ],
        "silhouette": [
            0.096475, 0.098287, 0.085195, 0.070670, 0.070912,
            0.060081, 0.068552, 0.063257, 0.066516,
        ],
    })

    chart_a, chart_b = st.columns(2)
    with chart_a:
        elbow_fig = px.line(
            k_results,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Method",
        )
        elbow_fig.update_traces(line_color="#B47B4D", marker_color="#8A6848")
        elbow_fig.update_layout(
            xaxis_title="Number of clusters",
            yaxis_title="Inertia",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(elbow_fig, use_container_width=True)

    with chart_b:
        silhouette_fig = px.line(
            k_results,
            x="k",
            y="silhouette",
            markers=True,
            title="Silhouette Score",
        )
        silhouette_fig.update_traces(line_color="#557A46", marker_color="#315B3E")
        silhouette_fig.add_vline(x=3, line_dash="dash", line_color="#B47B4D")
        silhouette_fig.update_layout(
            xaxis_title="Number of clusters",
            yaxis_title="Silhouette score",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(silhouette_fig, use_container_width=True)

    st.markdown(
        """
        **K = 3** produced the highest Silhouette score, approximately **0.098**.
        The low score matters: it suggests substantial overlap rather than sharply
        separated categories. That is consistent with an interdisciplinary catalogue
        in which ecology, philosophy, spirituality and embodied practice can intersect.
        """
    )

    @st.cache_data
    def make_pca_view(embedding_array, theme_values, title_values, author_values):
        coords = PCA(n_components=2, random_state=42).fit_transform(embedding_array)
        return pd.DataFrame({
            "PCA 1": coords[:, 0],
            "PCA 2": coords[:, 1],
            "Path": theme_values,
            "Title": title_values,
            "Author": author_values,
        })

    pca_df = make_pca_view(
        embeddings,
        books["collection_theme"].map(THEME_LABELS).fillna("Other").to_numpy(),
        books["title"].fillna("Untitled").to_numpy(),
        books["author"].fillna("Author unavailable").to_numpy(),
    )
    pca_fig = px.scatter(
        pca_df,
        x="PCA 1",
        y="PCA 2",
        color="Path",
        color_discrete_sequence=["#315B3E", "#557A46", "#8A6848", "#B47B4D", "#7F956B"],
        hover_name="Title",
        hover_data={"Author": True, "PCA 1": False, "PCA 2": False},
        title="A two dimensional view of the final semantic space",
        opacity=0.72,
    )
    pca_fig.update_layout(
        legend_title_text="Path of knowledge",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(pca_fig, use_container_width=True)
    st.caption(
        "PCA is used here for visualization. The recommendation system works in the "
        "full 384 dimensional embedding space."
    )

with journey_tabs[4]:
    st.markdown("### From a reader's words to books in resonance")
    st.markdown(
        """
        A reader does not need to begin with a genre or a known title. Their words
        become the beginning of a path through the library.
        """
    )
    st.markdown(
        """
        <div class="journey-path">
            <span class="journey-node">Your inquiry</span>
            <span class="journey-arrow">❧</span>
            <span class="journey-node">Sentence Transformer</span>
            <span class="journey-arrow">❧</span>
            <span class="journey-node">384 dimensions</span>
            <span class="journey-arrow">❧</span>
            <span class="journey-node">Cosine similarity</span>
            <span class="journey-arrow">❧</span>
            <span class="journey-node">Books in resonance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The reader's words are embedded with the same model used for the books.
        Cosine similarity then identifies nearby records in semantic space. Optional
        knowledge orientations gently steer the inquiry without turning the paths into
        rigid classifications.

        The app also supports book to book exploration through **Follow a thread of
        resonance**, allowing one book to become a doorway to another.
        """
    )

with journey_tabs[5]:
    st.markdown("### A relational layer behind the interface")
    st.markdown(
        """
        SQLite was used to structure and inspect the modeling catalogue. The SQL layer
        supports analysis of books, themes, subjects and semantic clusters while the
        Streamlit application combines the final catalogue with its aligned embedding
        matrix.
        """
    )
    st.markdown(
        """
        <div class="engineering-note">
            <strong>The important engineering constraint is alignment.</strong><br>
            Each catalogue row must correspond to the same row in the stored embedding
            matrix. This keeps every book connected to the correct semantic representation
            throughout recommendation and exploration.
        </div>
        """,
        unsafe_allow_html=True,
    )

with journey_tabs[6]:
    st.markdown("### From discovering knowledge to accessing it")
    st.markdown(
        """
        The next phase of Kinship Library will explore connections and possible
        collaborations with **verified public domain and open access digital libraries**.

        The aim is to let readers move, where legally and ethically possible, from
        discovering a book to accessing an open edition. This is especially relevant
        for ancient philosophical texts, public domain works and forms of ancestral
        and embodied knowledge that can be responsibly shared through open collections.

        Future access labels should clearly distinguish **public domain**, **open
        access**, **digital lending** and **copyrighted works**.

        **Discover → Connect → Access**
        """
    )
    st.markdown(
        """
        <div class="future-tree">
            <div class="crown">❧ ❧ ❧</div>
            <div>
                <span class="branch">Public domain</span>
                <span class="branch">Open access</span>
                <span class="branch">Digital libraries</span>
                <span class="branch">Ancestral knowledge</span>
            </div>
            <div class="trunk"></div>
            <div><strong>DISCOVER</strong> &nbsp;→&nbsp; <strong>CONNECT</strong> &nbsp;→&nbsp; <strong>ACCESS</strong></div>
            <div style="margin-top:.8rem;opacity:.82;font-size:.9rem;">
                A future network of responsible pathways from recommendation to knowledge access
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Phase 2 remains separate from the current recommendation system. Kinship Library "
        "does not host or claim free access to copyrighted books."
    )

st.markdown(
    """
    <section class="creator">
        <div class="eyebrow">About the Creator</div>
        <h2>Angélica Roa</h2>
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
        our way back into relationship starting with the encounter between a question and a book?
    </div>
    """,
    unsafe_allow_html=True,
)
