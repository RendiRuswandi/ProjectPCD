import streamlit as st

st.set_page_config(
    page_title="Penjelasan Teknik - PCD",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ Penjelasan Teknik PCD")
st.caption("Pelajari tentang teknik-teknik yang digunakan dalam aplikasi ini.")
st.markdown("---") # st.divider() diganti menjadi markdown

# --- Filtering ---
with st.expander("🎞️ Filtering (Gaussian Blur, Sharpen, Sepia, Cold/Warm)", expanded=True):
    st.subheader("Gaussian Blur")
    st.markdown("""
    **Apa itu?** Filter *low-pass* yang menghaluskan gambar dan mengurangi detail/noise.
    **Bagaimana?** Mengganti setiap piksel dengan rata-rata terbobot (distribusi Gaussian) dari piksel tetangganya.
    **Manfaat:** Menghilangkan noise halus.
    """)
    
    st.subheader("Sharpen")
    st.markdown("""
    **Apa itu?** Filter *high-pass* yang menonjolkan tepi dan detail halus.
    **Bagaimana?** Menggunakan kernel khusus (matriks 3x3) yang meningkatkan perbedaan antara piksel tetangga.
    **Manfaat:** Membuat gambar terlihat lebih tajam dan jelas.
    """)
    
    st.subheader("Sepia")
    st.markdown("""
    **Apa itu?** Filter warna yang memberikan tampilan foto lama (kekuningan/kecoklatan).
    **Bagaimana?** Menerapkan matriks transformasi linear ke channel warna R, G, B.
    **Manfaat:** Estetika foto retro atau antik.
    """)
    
    st.subheader("Koreksi Warna (Cold/Warm)")
    st.markdown("""
    **Apa itu?** Menyesuaikan "suhu" gambar agar terlihat lebih dingin (kebiruan) atau lebih hangat (kekuningan).
    **Bagaimana?** Menambah/mengurangi nilai channel Biru (B) dan Merah (R) menggunakan Lookup Table (LUT).
    **Manfaat:** Memperbaiki *white balance* atau memberi *mood* tertentu pada foto.
    """)

# --- Restorasi ---
with st.expander("🛠️ Restorasi (Median, Bilateral, Inpainting)", expanded=False):
    st.subheader("Median Blur")
    st.markdown("""
    **Apa itu?** Filter non-linear yang sangat efektif menghilangkan noise *salt-and-pepper* (bintik putih/hitam acak).
    **Bagaimana?** Mengganti setiap piksel dengan nilai **median** (nilai tengah) dari piksel tetangganya.
    **Manfaat:** Menghilangkan noise ekstrem sambil menjaga tepi lebih baik daripada Gaussian Blur.
    """)
    
    st.subheader("Bilateral Filter")
    st.markdown("""
    **Apa itu?** Filter penghilang noise yang canggih.
    **Bagaimana?** Mirip Gaussian Blur, tapi mempertimbangkan dua hal: kedekatan spasial (jarak) dan kemiripan intensitas/warna.
    **Manfaat:** Mereduksi noise secara signifikan sambil **menjaga ketajaman tepi** gambar.
    """)
    
    st.subheader("Inpainting (Interaktif)")
    st.markdown("""
    **Apa itu?** Teknik restorasi untuk mengisi area gambar yang rusak atau hilang (atau yang Anda tandai).
    **Bagaimana?** Algoritma menganalisis piksel di sekitar "lubang" (masker) dan menyebarkan informasi (warna, tekstur) tersebut ke dalam area yang kosong.
    **Manfaat:** Menghilangkan goresan, noda, atau objek kecil secara "ajaib".
    """)

# --- Enhancement ---
with st.expander("✨ Enhancement (CLAHE, Brightness/Contrast, Unsharp Mask)", expanded=False):
    st.subheader("Brightness / Contrast")
    st.markdown("""
    **Apa itu?** Penyesuaian dasar.
    **Bagaimana?** *Brightness* menggeser histogram. *Contrast* merentangkan/memampatkan histogram.
    **Manfaat:** Memperbaiki gambar yang terlalu gelap, terang, atau "datar".
    """)

    st.subheader("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    st.markdown("""
    **Apa itu?** Peningkatan kontras yang jauh lebih canggih.
    **Bagaimana?** Bekerja pada bagian-bagian kecil gambar (*tiles*) secara terpisah, bukan pada gambar keseluruhan.
    **Manfaat:** Menghasilkan detail yang sangat baik di area bayangan dan terang tanpa merusak gambar.
    """)

    st.subheader("Unsharp Masking")
    st.markdown("""
    **Apa itu?** Teknik penajaman yang lebih terkontrol.
    **Bagaimana?** Membuat versi blur dari gambar, menguranginya dari gambar asli (untuk mendapatkan "masker detail"), lalu menambahkan masker detail tersebut kembali ke gambar asli.
    **Manfaat:** Memberikan kontrol lebih besar atas penajaman.
    """)

# --- Transformasi (BARU) ---
with st.expander("🔄 Transformasi (Rotasi, Flip)", expanded=False):
    st.subheader("Rotasi")
    st.markdown("""
    **Apa itu?** Memutar gambar mengelilingi titik tengahnya.
    **Bagaimana?** Menerapkan matriks transformasi afinitas (affine transformation) ke setiap piksel.
    **Manfaat:** Meluruskan gambar yang miring.
    """)
    
    st.subheader("Flip (Cermin)")
    st.markdown("""
    **Apa itu?** Membalik gambar secara horizontal atau vertikal.
    **Bagaimana?** Membalik urutan piksel sepanjang sumbu X (Vertikal) atau sumbu Y (Horizontal).
    **Manfaat:** Koreksi orientasi (misal, foto selfie yang terbalik).
    """)

# --- Analisis ---
with st.expander("🎨 Analisis (Palet Warna)", expanded=False):
    st.subheader("Analisis Palet Warna (K-Means)")
    st.markdown("""
    **Apa itu?** Mengekstrak warna-warna yang paling dominan dalam sebuah gambar.
    **Bagaimana?** Menggunakan algoritma *machine learning* **K-Means Clustering** untuk mengelompokkan piksel berdasarkan warnanya.
    **Manfaat:** Memahami komposisi warna gambar.
    """)