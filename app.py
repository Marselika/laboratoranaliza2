import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURARE PAGINĂ
# ================================================================================
st.set_page_config(
    page_title="Wine Data Analysis",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CSS CUSTOM PENTRU DESIGN MODERN
# ================================================================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #722f37;
        font-family: 'Arial', sans-serif;
    }
    h2, h3 {
        color: #8b4049;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ================================================================================
# FUNCȚII UTILITARE
# ================================================================================

@st.cache_data
def load_data():
    """Încarcă datele din fișierul CSV"""
    df = pd.read_csv('wine_clean_final.csv')

    # Conversii necesare
    if 'vintage' in df.columns:
        df['vintage'] = pd.to_numeric(df['vintage'], errors='coerce')
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'points' in df.columns:
        df['points'] = pd.to_numeric(df['points'], errors='coerce')
    if 'alcohol' in df.columns:
        df['alcohol'] = pd.to_numeric(df['alcohol'], errors='coerce')

    # Calculează raportul preț/calitate dacă nu există
    if 'price_quality_ratio' not in df.columns:
        df['price_quality_ratio'] = df['price'] / df['points']

    return df


def clean_text(text):
    """Curăță textul pentru analiza textuală"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def search_wines(df, query, top_n=10):
    """Caută vinuri bazat pe descriere folosind TF-IDF"""
    # Pregătește descrierile
    descriptions = df['description'].fillna('').apply(clean_text)

    # Creează vectorizator TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Procesează query-ul
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])

    # Calculează similaritatea
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Găsește top N rezultate
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]

    return results[results['similarity_score'] > 0]


# ================================================================================
# ÎNCĂRCARE DATE
# ================================================================================
df = load_data()

# ================================================================================
# HEADER
# ================================================================================
st.title("🍷 Wine Data Analysis Dashboard")
st.markdown("---")

# ================================================================================
# SIDEBAR - FILTRE
# ================================================================================
st.sidebar.header("🔍 Filtre")

# Filtre pentru preț
price_min = float(df['price'].min())
price_max = float(df['price'].max())
price_range = st.sidebar.slider(
    "Interval Preț (USD)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=1.0
)

# Filtre pentru points
points_min = int(df['points'].min())
points_max = int(df['points'].max())
points_range = st.sidebar.slider(
    "Interval Rating (Points)",
    min_value=points_min,
    max_value=points_max,
    value=(points_min, points_max),
    step=1
)

# Filtre pentru raport preț/calitate
if 'price_quality_ratio' in df.columns:
    ratio_min = float(df['price_quality_ratio'].min())
    ratio_max = float(df['price_quality_ratio'].max())
    ratio_range = st.sidebar.slider(
        "Raport Preț/Calitate (mai mic = mai bun)",
        min_value=ratio_min,
        max_value=min(ratio_max, 5.0),  # Limitează la 5 pentru vizibilitate
        value=(ratio_min, min(ratio_max, 2.0)),
        step=0.1
    )

# Multiselect pentru țări
countries = sorted(df['country'].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Țări",
    options=countries,
    default=[]
)

# Multiselect pentru categorii
if 'category' in df.columns:
    categories = sorted(df['category'].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Categorii",
        options=categories,
        default=[]
    )

# Multiselect pentru soiuri
varieties = sorted(df['variety'].dropna().unique())
selected_varieties = st.sidebar.multiselect(
    "Soiuri (Variety)",
    options=varieties[:50],  # Limitează la primele 50 pentru performanță
    default=[]
)

# ================================================================================
# APLICARE FILTRE
# ================================================================================
df_filtered = df.copy()

# Aplicare filtre
df_filtered = df_filtered[
    (df_filtered['price'] >= price_range[0]) &
    (df_filtered['price'] <= price_range[1]) &
    (df_filtered['points'] >= points_range[0]) &
    (df_filtered['points'] <= points_range[1])
    ]

if 'price_quality_ratio' in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered['price_quality_ratio'] >= ratio_range[0]) &
        (df_filtered['price_quality_ratio'] <= ratio_range[1])
        ]

if selected_countries:
    df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]

if 'category' in df_filtered.columns and selected_categories:
    df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]

if selected_varieties:
    df_filtered = df_filtered[df_filtered['variety'].isin(selected_varieties)]

# Afișare informații filtre
st.sidebar.markdown("---")
st.sidebar.metric("Vinuri Filtrate", f"{len(df_filtered):,}")
st.sidebar.metric("Total Vinuri", f"{len(df):,}")

# ================================================================================
# TABS PRINCIPALE
# ================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Distribuții",
    "🔗 Corelații",
    "🌍 Analiză Geografică",
    "🔍 Căutare Text"
])

# ================================================================================
# TAB 1: OVERVIEW
# ================================================================================
with tab1:
    st.header("📊 Statistici Generale")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Număr Vinuri",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}" if len(df_filtered) != len(df) else None
        )

    with col2:
        avg_price = df_filtered['price'].mean()
        st.metric(
            "Preț Mediu",
            f"${avg_price:.2f}",
            delta=f"${avg_price - df['price'].mean():.2f}" if len(df_filtered) != len(df) else None
        )

    with col3:
        avg_points = df_filtered['points'].mean()
        st.metric(
            "Rating Mediu",
            f"{avg_points:.1f}",
            delta=f"{avg_points - df['points'].mean():.1f}" if len(df_filtered) != len(df) else None
        )

    with col4:
        if 'alcohol' in df_filtered.columns:
            avg_alcohol = df_filtered['alcohol'].mean()
            st.metric(
                "Alcool Mediu",
                f"{avg_alcohol:.1f}%",
                delta=f"{avg_alcohol - df['alcohol'].mean():.1f}%" if len(df_filtered) != len(df) else None
            )

    st.markdown("---")

    # Statistici descriptive detaliate
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Statistici Numerice")
        numeric_stats = df_filtered[['price', 'points', 'alcohol']].describe()
        st.dataframe(numeric_stats.style.format("{:.2f}"), use_container_width=True)

    with col2:
        st.subheader("🏆 Top 10 Vinuri după Rating")
        top_wines = df_filtered.nlargest(10, 'points')[['title', 'points', 'price', 'country']]
        st.dataframe(top_wines, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Best Value Wines
    if 'price_quality_ratio' in df_filtered.columns:
        st.subheader("💎 Best Value Wines (Cel mai bun raport calitate-preț)")
        best_value = df_filtered.nsmallest(10, 'price_quality_ratio')[
            ['title', 'points', 'price', 'price_quality_ratio', 'country']
        ]
        best_value['price_quality_ratio'] = best_value['price_quality_ratio'].round(4)
        st.dataframe(best_value, use_container_width=True, hide_index=True)

# ================================================================================
# TAB 2: DISTRIBUȚII
# ================================================================================
with tab2:
    st.header("📈 Analiza Distribuțiilor")

    col1, col2 = st.columns(2)

    # Histogramă Points
    with col1:
        st.subheader("Distribuția Rating-urilor (Points)")
        fig_points = px.histogram(
            df_filtered,
            x='points',
            nbins=30,
            title="Distribuția Punctajelor",
            labels={'points': 'Rating (Points)', 'count': 'Frecvență'},
            color_discrete_sequence=['#722f37']
        )
        fig_points.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_points, use_container_width=True)

    # Histogramă Price
    with col2:
        st.subheader("Distribuția Prețurilor")
        fig_price = px.histogram(
            df_filtered,
            x='price',
            nbins=50,
            title="Distribuția Prețurilor",
            labels={'price': 'Preț (USD)', 'count': 'Frecvență'},
            color_discrete_sequence=['#8b4049']
        )
        fig_price.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # Box plots pe categorii
    if 'category' in df_filtered.columns:
        st.subheader("Box Plots - Distribuții pe Categorii")

        col1, col2 = st.columns(2)

        with col1:
            fig_box_price = px.box(
                df_filtered,
                x='category',
                y='price',
                title="Distribuția Prețurilor pe Categorii",
                labels={'category': 'Categorie', 'price': 'Preț (USD)'},
                color='category'
            )
            fig_box_price.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_box_price, use_container_width=True)

        with col2:
            fig_box_points = px.box(
                df_filtered,
                x='category',
                y='points',
                title="Distribuția Rating-urilor pe Categorii",
                labels={'category': 'Categorie', 'points': 'Rating'},
                color='category'
            )
            fig_box_points.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_box_points, use_container_width=True)

    st.markdown("---")

    # Violin plots
    st.subheader("Violin Plots - Distribuții Detaliate")

    col1, col2 = st.columns(2)

    with col1:
        if 'category' in df_filtered.columns:
            fig_violin_price = px.violin(
                df_filtered,
                y='price',
                x='category',
                box=True,
                title="Violin Plot - Preț pe Categorii",
                labels={'category': 'Categorie', 'price': 'Preț (USD)'},
                color='category'
            )
            fig_violin_price.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_violin_price, use_container_width=True)

    with col2:
        if 'category' in df_filtered.columns:
            fig_violin_points = px.violin(
                df_filtered,
                y='points',
                x='category',
                box=True,
                title="Violin Plot - Rating pe Categorii",
                labels={'category': 'Categorie', 'points': 'Rating'},
                color='category'
            )
            fig_violin_points.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_violin_points, use_container_width=True)

# ================================================================================
# TAB 3: CORELAȚII
# ================================================================================
with tab3:
    st.header("🔗 Analiza Corelațiilor")

    # Scatter plot Price vs Points
    st.subheader("Relația Preț - Rating")

    if 'category' in df_filtered.columns:
        fig_scatter = px.scatter(
            df_filtered,
            x='points',
            y='price',
            color='category',
            size='alcohol' if 'alcohol' in df_filtered.columns else None,
            hover_data=['title', 'country', 'variety'],
            title="Scatter Plot: Preț vs Rating (colorat pe Categorii)",
            labels={'points': 'Rating (Points)', 'price': 'Preț (USD)'},
            opacity=0.6
        )
    else:
        fig_scatter = px.scatter(
            df_filtered,
            x='points',
            y='price',
            hover_data=['title', 'country', 'variety'],
            title="Scatter Plot: Preț vs Rating",
            labels={'points': 'Rating (Points)', 'price': 'Preț (USD)'},
            opacity=0.6,
            color_discrete_sequence=['#722f37']
        )

    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Heatmap corelații
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Heatmap - Matrice Corelație")

        # Selectează doar coloanele numerice
        numeric_cols = ['price', 'points']
        if 'alcohol' in df_filtered.columns:
            numeric_cols.append('alcohol')
        if 'vintage' in df_filtered.columns:
            numeric_cols.append('vintage')
        if 'price_quality_ratio' in df_filtered.columns:
            numeric_cols.append('price_quality_ratio')

        corr_matrix = df_filtered[numeric_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title="Matrice de Corelație",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader("Coeficienți Corelație")
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.3f}"),
            use_container_width=True
        )

    st.markdown("---")

    # Scatter plots adiționale
    col1, col2 = st.columns(2)

    with col1:
        if 'alcohol' in df_filtered.columns:
            st.subheader("Alcool vs Rating")
            fig_alc_points = px.scatter(
                df_filtered,
                x='alcohol',
                y='points',
                trendline='ols',
                title="Relația Conținut Alcool - Rating",
                labels={'alcohol': 'Alcool (%)', 'points': 'Rating'},
                opacity=0.5,
                color_discrete_sequence=['#8b4049']
            )
            fig_alc_points.update_layout(height=350)
            st.plotly_chart(fig_alc_points, use_container_width=True)

    with col2:
        if 'alcohol' in df_filtered.columns:
            st.subheader("Alcool vs Preț")
            fig_alc_price = px.scatter(
                df_filtered,
                x='alcohol',
                y='price',
                trendline='ols',
                title="Relația Conținut Alcool - Preț",
                labels={'alcohol': 'Alcool (%)', 'price': 'Preț (USD)'},
                opacity=0.5,
                color_discrete_sequence=['#722f37']
            )
            fig_alc_price.update_layout(height=350)
            st.plotly_chart(fig_alc_price, use_container_width=True)

# ================================================================================
# TAB 4: ANALIZĂ GEOGRAFICĂ
# ================================================================================
with tab4:
    st.header("🌍 Analiză Geografică")

    # Prețuri medii pe țară
    st.subheader("Prețuri Medii pe Țară (Top 20)")

    country_stats = df_filtered.groupby('country').agg({
        'price': 'mean',
        'points': 'mean',
        'title': 'count'
    }).round(2)
    country_stats.columns = ['Preț Mediu', 'Rating Mediu', 'Număr Vinuri']
    country_stats = country_stats.sort_values('Preț Mediu', ascending=False).head(20)

    fig_country_price = px.bar(
        country_stats.reset_index(),
        x='country',
        y='Preț Mediu',
        title="Top 20 Țări după Preț Mediu",
        labels={'country': 'Țară', 'Preț Mediu': 'Preț Mediu (USD)'},
        color='Preț Mediu',
        color_continuous_scale='Reds'
    )
    fig_country_price.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_country_price, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Rating mediu pe țară
        st.subheader("Rating Mediu pe Țară (Top 15)")

        country_rating = df_filtered.groupby('country')['points'].mean().sort_values(ascending=False).head(15)

        fig_country_rating = px.bar(
            x=country_rating.index,
            y=country_rating.values,
            title="Top 15 Țări după Rating Mediu",
            labels={'x': 'Țară', 'y': 'Rating Mediu'},
            color=country_rating.values,
            color_continuous_scale='Viridis'
        )
        fig_country_rating.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_country_rating, use_container_width=True)

    with col2:
        # Număr vinuri pe țară
        st.subheader("Număr Vinuri pe Țară (Top 15)")

        country_count = df_filtered['country'].value_counts().head(15)

        fig_country_count = px.pie(
            values=country_count.values,
            names=country_count.index,
            title="Distribuția Vinurilor pe Țări",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_country_count.update_layout(height=400)
        st.plotly_chart(fig_country_count, use_container_width=True)

    st.markdown("---")

    # Tabel detaliat statistici pe țară
    st.subheader("Statistici Detaliate pe Țări")

    detailed_stats = df_filtered.groupby('country').agg({
        'price': ['mean', 'min', 'max'],
        'points': ['mean', 'min', 'max'],
        'title': 'count'
    }).round(2)

    detailed_stats.columns = [
        'Preț Mediu', 'Preț Min', 'Preț Max',
        'Rating Mediu', 'Rating Min', 'Rating Max',
        'Număr Vinuri'
    ]
    detailed_stats = detailed_stats.sort_values('Număr Vinuri', ascending=False).head(20)

    st.dataframe(detailed_stats, use_container_width=True)

    st.markdown("---")

    # Analiză pe soiuri (Variety)
    st.subheader("Top Soiuri de Struguri")

    col1, col2 = st.columns(2)

    with col1:
        variety_count = df_filtered['variety'].value_counts().head(15)

        fig_variety = px.bar(
            x=variety_count.values,
            y=variety_count.index,
            orientation='h',
            title="Top 15 Soiuri după Număr",
            labels={'x': 'Număr Vinuri', 'y': 'Soi'},
            color=variety_count.values,
            color_continuous_scale='Oranges'
        )
        fig_variety.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_variety, use_container_width=True)

    with col2:
        variety_price = df_filtered.groupby('variety')['price'].mean().sort_values(ascending=False).head(15)

        fig_variety_price = px.bar(
            x=variety_price.values,
            y=variety_price.index,
            orientation='h',
            title="Top 15 Soiuri după Preț Mediu",
            labels={'x': 'Preț Mediu (USD)', 'y': 'Soi'},
            color=variety_price.values,
            color_continuous_scale='Reds'
        )
        fig_variety_price.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_variety_price, use_container_width=True)

# ================================================================================
# TAB 5: CĂUTARE TEXT
# ================================================================================
with tab5:
    st.header("🔍 Căutare Vinuri după Descriere")

    st.markdown("""
    Introduceți cuvinte cheie sau o descriere pentru a găsi vinurile similare.
    Sistemul folosește **TF-IDF** și **Cosine Similarity** pentru a găsi cele mai relevante rezultate.
    """)

    # Input pentru căutare
    search_query = st.text_input(
        "Descrieți vinul pe care îl căutați:",
        placeholder="Ex: fruity red wine with oak and tannins, cherry notes"
    )

    # Buton căutare
    if st.button("🔍 Caută Vinuri", type="primary"):
        if search_query:
            with st.spinner("Caut vinuri similare..."):
                results = search_wines(df_filtered, search_query, top_n=20)

            if len(results) > 0:
                st.success(f"Am găsit {len(results)} vinuri relevante!")

                # Afișare rezultate
                for idx, row in results.iterrows():
                    with st.expander(
                            f"⭐ {row['title']} - Rating: {row['points']} | Preț: ${row['price']:.2f} | Similaritate: {row['similarity_score']:.3f}"
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Țară:** {row['country']}")
                            st.markdown(f"**Soi:** {row['variety']}")
                            st.markdown(f"**Cramă:** {row['winery']}")
                            if 'vintage' in row and not pd.isna(row['vintage']):
                                st.markdown(f"**An:** {int(row['vintage'])}")
                            if 'alcohol' in row and not pd.isna(row['alcohol']):
                                st.markdown(f"**Alcool:** {row['alcohol']:.1f}%")

                        with col2:
                            st.metric("Rating", f"{row['points']}")
                            st.metric("Preț", f"${row['price']:.2f}")
                            if 'price_quality_ratio' in row:
                                st.metric("Raport P/C", f"{row['price_quality_ratio']:.3f}")

                        st.markdown("**Descriere:**")
                        st.write(row['description'])
            else:
                st.warning("Nu am găsit vinuri relevante pentru această căutare. Încercați alți termeni.")
        else:
            st.info("Vă rugăm introduceți o descriere pentru a căuta.")

    st.markdown("---")

    # Analiză text - Word Cloud
    st.subheader("☁️ Word Cloud - Cuvinte Frecvente în Descrieri")

    if st.button("Generează Word Cloud"):
        with st.spinner("Generez word cloud..."):
            # Combină toate descrierile
            all_descriptions = ' '.join(df_filtered['description'].fillna('').astype(str))

            # Curăță textul
            all_descriptions = clean_text(all_descriptions)

            # Generează wordcloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                colormap='RdPu',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(all_descriptions)

            # Afișare
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)

    st.markdown("---")

    # Top cuvinte frecvente
    st.subheader("📊 Top Cuvinte Frecvente")

    if st.button("Analizează Cuvinte"):
        with st.spinner("Analizez textul..."):
            # Vectorizare TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )

            descriptions = df_filtered['description'].fillna('').apply(clean_text)
            tfidf_matrix = vectorizer.fit_transform(descriptions)

            # Calculează scoruri medii
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.mean(axis=0).A1

            # Creează DataFrame
            word_scores = pd.DataFrame({
                'Cuvânt/Expresie': feature_names,
                'Scor TF-IDF': tfidf_scores
            }).sort_values('Scor TF-IDF', ascending=False).head(30)

            # Grafic
            fig_words = px.bar(
                word_scores,
                x='Scor TF-IDF',
                y='Cuvânt/Expresie',
                orientation='h',
                title="Top 30 Cuvinte/Expresii după Scor TF-IDF",
                color='Scor TF-IDF',
                color_continuous_scale='Reds'
            )
            fig_words.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig_words, use_container_width=True)

    st.markdown("---")

    # Statistici lungime descriere
    st.subheader("📏 Statistici Lungime Descrieri")

    col1, col2, col3 = st.columns(3)

    df_filtered['desc_length'] = df_filtered['description'].fillna('').str.len()
    df_filtered['desc_words'] = df_filtered['description'].fillna('').str.split().str.len()

    with col1:
        avg_length = df_filtered['desc_length'].mean()
        st.metric("Lungime Medie (caractere)", f"{avg_length:.0f}")

    with col2:
        avg_words = df_filtered['desc_words'].mean()
        st.metric("Număr Mediu Cuvinte", f"{avg_words:.0f}")

    with col3:
        max_length = df_filtered['desc_length'].max()
        st.metric("Lungime Maximă", f"{max_length}")

    # Histogramă lungime descrieri
    fig_length = px.histogram(
        df_filtered,
        x='desc_length',
        nbins=50,
        title="Distribuția Lungimii Descrierilor",
        labels={'desc_length': 'Lungime (caractere)', 'count': 'Frecvență'},
        color_discrete_sequence=['#722f37']
    )
    fig_length.update_layout(height=400)
    st.plotly_chart(fig_length, use_container_width=True)

# ================================================================================
# FOOTER
# ================================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🍷 <strong>Wine Data Analysis Dashboard</strong> | Creat de Iatco Marcel</p>
        <p>Powered by Streamlit, Plotly & Python</p>
    </div>
""", unsafe_allow_html=True)



