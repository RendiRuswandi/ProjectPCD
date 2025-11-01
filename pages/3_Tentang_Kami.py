import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Tentang Kami - Studio PCD",
    page_icon="👥",
    layout="centered"
)

st.title("👥 Tentang Kami")
st.divider()

st.markdown("""
Proyek **Studio PCD Interaktif** ini dibuat sebagai bagian dari pemenuhan tugas mata kuliah Pengolahan Citra Digital. 

Kami berfokus pada implementasi berbagai teknik PCD (Filtering, Restorasi, dan Enhancement) dalam sebuah aplikasi web yang interaktif, bermanfaat, dan mudah digunakan menggunakan Streamlit.
""")

st.divider()
st.header("Tim Pengembang")

# Gunakan kolom untuk menata profil anggota
col1, col2 = st.columns([1, 2])

with col1:
    # Anda bisa mengupload foto ke folder yang sama dan membukanya
    # try:
    #     profile_pic = Image.open("foto_tim.jpg")
    #     st.image(profile_pic, width=150, caption="Nama Anda")
    # except FileNotFoundError:
    #     st.markdown("**(Foto Anggota 1)**", unsafe_allow_html=True)
    
    st.markdown("### 👤", unsafe_allow_html=True, help="Placeholder Foto")


with col2:
    st.subheader("Rendi Ruswandi") # Ganti dengan nama Anda
    st.write("**Peran:** Full-Stack Developer, UI/UX Designer") # Ganti
    st.write("**NIM:** XXXXXXXX") # Ganti
    st.write("Kontak: [LinkedIn Anda](https://linkedin.com) | [GitHub Anda](https://github.com)") # Ganti

st.divider()

# Ulangi blok di atas untuk anggota tim lainnya
col3, col4 = st.columns([1, 2])
with col3:
    st.markdown("### 👤", unsafe_allow_html=True, help="Placeholder Foto")
with col4:
    st.subheader("Nama Anggota 2")
    st.write("**Peran:** Analis PCD, Dokumentasi")
    st.write("**NIM:** XXXXXXXX")
    st.write("Kontak: [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)")