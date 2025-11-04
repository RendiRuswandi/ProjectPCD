import streamlit as st

st.set_page_config(
    page_title="Penjelasan Teknik - Restorasi",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ Penjelasan Teknik Restorasi")
st.caption("Pelajari tentang teknik-teknik yang digunakan di Studio Restorasi.")
st.markdown("---") 

# --- Restorasi Otomatis ---
with st.expander("Restorasi Otomatis (Noise & Detail)", expanded=True):
    st.subheader("Median Blur")
    st.markdown("""
    **Apa itu?** Filter non-linear yang sangat efektif menghilangkan noise *salt-and-pepper* (bintik putih/hitam acak).
    **Bagaimana?** Mengganti setiap piksel dengan nilai **median** (nilai tengah) dari piksel tetangganya.
    **Manfaat:** Menghilangkan bintik debu atau noise digital yang tajam.
    """)
    
    st.subheader("Bilateral Filter")
    st.markdown("""
    **Apa itu?** Filter penghilang noise yang canggih.
    **Bagaimana?** Mirip Gaussian Blur, tapi mempertimbangkan dua hal: kedekatan spasial (jarak) dan kemiripan intensitas/warna.
    **Manfaat:** Mereduksi noise secara signifikan sambil **menjaga ketajaman tepi** gambar. Sangat bagus untuk menghaluskan kulit tanpa membuat blur mata.
    """)
    
    st.subheader("Brightness / Contrast")
    st.markdown("""
    **Apa itu?** Penyesuaian dasar untuk memperbaiki pencahayaan.
    **Bagaimana?** *Brightness* menggeser histogram. *Contrast* merentangkan/memampatkan histogram.
    **Manfaat:** Memperbaiki foto yang terlalu gelap, terlalu terang, atau "datar" (pudar).
    """)

    st.subheader("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    st.markdown("""
    **Apa itu?** Peningkatan kontras yang jauh lebih canggih.
    **Bagaimana?** Bekerja pada bagian-bagian kecil gambar (*tiles*) secara terpisah, bukan pada gambar keseluruhan.
    **Manfaat:** Menghasilkan detail yang sangat baik di area bayangan dan terang tanpa merusak gambar. Sangat bagus untuk foto pudar.
    """)

    st.subheader("Unsharp Masking")
    st.markdown("""
    **Apa itu?** Teknik penajaman yang lebih terkontrol.
    **Bagaimana?** Membuat versi blur dari gambar, menguranginya dari gambar asli (untuk mendapatkan "masker detail"), lalu menambahkan masker detail tersebut kembali ke gambar asli.
    **Manfaat:** Menegaskan detail yang sedikit blur dan membuat foto terlihat lebih "jelas".
    """)

# --- Restorasi Interaktif ---
with st.expander("Restorasi Interaktif (Manual)", expanded=False):
    st.subheader("Inpainting (Interaktif)")
    st.markdown("""
    **Apa itu?** Teknik restorasi untuk mengisi area gambar yang rusak atau hilang (atau yang Anda tandai).
    **Bagaimana?** Anda "mencoret" area yang rusak. Algoritma menganalisis piksel di sekitar "lubang" (masker) dan menyebarkan informasi (warna, tekstur) tersebut ke dalam area yang kosong.
    **Manfaat:** Menghilangkan goresan, noda, robekan, atau objek kecil yang tidak diinginkan secara "ajaib".
    """)

    st.subheader("Dodge & Burn (Interaktif)")
    st.markdown("""
    **Apa itu?** Teknik fotografi klasik untuk mencerahkan (Dodge) atau menggelapkan (Burn) area tertentu secara manual.
    **Bagaimana?** Anda "menggosok" kanvas. Aplikasi membuat versi gambar yang lebih terang/gelap dan menggabungkannya dengan gambar asli hanya di area yang Anda coret.
    **Manfaat:** Memperbaiki area yang terlalu terang (*overexposed*) atau terlalu gelap (*underexposed*) tanpa mengubah keseluruhan foto.
    """)