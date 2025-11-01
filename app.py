import streamlit as st

st.set_page_config(
    page_title="Selamat Datang - Studio PCD",
    page_icon="✨",
    layout="centered"
)

# Sembunyikan navigasi sidebar di halaman ini
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Konten Halaman
st.markdown("<br><br><br>", unsafe_allow_html=True) # Spacer
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🔬</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Studio PCD Interaktif</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Filter, Restorasi, dan Analisis Citra dalam Satu Aplikasi.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Tombol Aksi Utama (st.link_button akan berfungsi sekarang)
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    st.link_button("Buka Studio Editor ➔", "/Studio_Editor", use_container_width=True)

st.markdown(
    """
    <style>
        /* Target tombol link button untuk membuatnya lebih besar */
        div[data-testid="stLinkButton"] a {
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)