import streamlit as st

st.set_page_config(
    page_title="Selamat Datang - Studio PCD",
    page_icon="✨",
    layout="centered"
)

# Konten Halaman
st.markdown("<br><br><br>", unsafe_allow_html=True) # Spacer
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🔬</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Studio PCD Interaktif</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Filter, Restorasi, dan Analisis Citra dalam Satu Aplikasi.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Tombol di Tengah ---
# Kita gunakan 3 kolom, dan letakkan tombol di kolom tengah (col2)
col1, col2, col3 = st.columns([1.5, 2, 1.5]) 
with col2:
    st.markdown(
        """
        <a href="Studio_Editor" target="_self" style="
            display: inline-block;
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            background-color: #007BFF; /* Warna aksen (biru) */
            border: none;
            border-radius: 0.5rem;
            text-decoration: none;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
        ">
            Buka Studio Editor ➔
        </a>
        """,
        unsafe_allow_html=True
    )