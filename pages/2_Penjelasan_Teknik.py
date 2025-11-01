import streamlit as st

st.set_page_config(
    page_title="Penjelasan Teknik - PCD",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ Penjelasan Teknik PCD")
st.caption("Pelajari tentang teknik-teknik yang digunakan dalam aplikasi ini.")
st.divider()

# --- Filtering ---
with st.expander("🎞️ Filtering (Gaussian Blur, Sharpen)", expanded=True):
    st.subheader("Gaussian Blur")
    st.markdown("""
    **Apa itu?** Filter *low-pass* yang menghaluskan gambar dan mengurangi detail/noise.
    **Bagaimana?** Mengganti setiap piksel dengan rata-rata terbobot (distribusi Gaussian) dari piksel tetangganya. Semakin besar "Kekuatan Blur" (ukuran kernel), semakin kabur hasilnya.
    **Manfaat:** Menghilangkan noise halus, mempersiapkan gambar untuk deteksi tepi.
    """)
    
    st.subheader("Sharpen")
    st.markdown("""
    **Apa itu?** Filter *high-pass* yang menonjolkan tepi dan detail halus.
    **Bagaimana?** Menggunakan kernel khusus (matriks 3x3) yang meningkatkan perbedaan antara piksel tetangga.
    **Manfaat:** Membuat gambar terlihat lebih tajam dan jelas.
    """)

# --- Restorasi ---
with st.expander("🛠️ Restorasi (Median, Bilateral, Inpainting)", expanded=False):
    st.subheader("Median Blur")
    st.markdown("""
    **Apa itu?** Filter non-linear yang sangat efektif menghilangkan noise *salt-and-pepper* (bintik putih/hitam acak).
    **Bagaimana?** Mengganti setiap piksel dengan nilai **median** (nilai tengah) dari piksel tetangganya dalam jendela kernel.
    **Manfaat:** Menghilangkan noise ekstrem sambil menjaga tepi lebih baik daripada Gaussian Blur.
    """)
    
    st.subheader("Bilateral Filter")
    st.markdown("""
    **Apa itu?** Filter penghilang noise yang canggih.
    **Bagaimana?** Mirip Gaussian Blur, tapi mempertimbangkan dua hal: kedekatan spasial (jarak) dan kemiripan intensitas/warna. Piksel yang jaraknya dekat TAPI warnanya sangat berbeda (seperti di tepi) tidak akan dirata-ratakan.
    **Manfaat:** Mereduksi noise secara signifikan sambil **menjaga ketajaman tepi** gambar.
    """)
    
    st.subheader("Inpainting (Interaktif)")
    st.markdown("""
    **Apa itu?** Teknik restorasi untuk mengisi area gambar yang rusak atau hilang (atau yang Anda tandai).
    **Bagaimana?** Algoritma menganalisis piksel di sekitar "lubang" (masker) dan menyebarkan informasi (warna, tekstur) tersebut ke dalam area yang kosong.
    * **TELEA:** Metode berbasis Fast Marching Method.
    * **NS:** Metode berbasis Navier-Stokes (fluid dynamics).
    **Manfaat:** Menghilangkan goresan, noda, atau objek kecil secara "ajaib".
    """)

# --- Enhancement ---
with st.expander("✨ Enhancement (CLAHE, Brightness/Contrast, Unsharp Mask)", expanded=False):
    st.subheader("Brightness / Contrast")
    st.markdown("""
    **Apa itu?** Penyesuaian dasar.
    **Bagaimana?** *Brightness* menambahkan/mengurangi nilai konstan ke semua piksel (menggeser histogram). *Contrast* mengalikan nilai piksel (merentangkan/memampatkan histogram).
    **Manfaat:** Memperbaiki gambar yang terlalu gelap, terang, atau "datar".
    """)

    st.subheader("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    st.markdown("""
    **Apa itu?** Peningkatan kontras yang jauh lebih canggih daripada Histogram Equalization standar.
    **Bagaimana?** Bekerja pada bagian-bagian kecil gambar (*tiles*) secara terpisah, bukan pada gambar keseluruhan. *Clip Limit* membatasi seberapa kuat kontras ditingkatkan di setiap *tile*, mencegah noise diperkuat secara berlebihan.
    **Manfaat:** Menghasilkan detail yang sangat baik di area bayangan dan terang tanpa merusak gambar.
    """)

    st.subheader("Unsharp Masking")
    st.markdown("""
    **Apa itu?** Teknik penajaman yang lebih terkontrol.
    **Bagaimana?** Membuat versi blur dari gambar, menguranginya dari gambar asli (untuk mendapatkan "masker detail"), lalu menambahkan masker detail tersebut kembali ke gambar asli.
    **Manfaat:** Memberikan kontrol lebih besar atas penajaman (kekuatan/radius) dibanding filter Sharpen standar.
    """)

# --- Analisis ---
with st.expander("🎨 Analisis (Palet Warna)", expanded=False):
    st.subheader("Analisis Palet Warna (K-Means)")
    st.markdown("""
    **Apa itu?** Mengekstrak warna-warna yang paling dominan dalam sebuah gambar.
    **Bagaimana?** Menggunakan algoritma *machine learning* **K-Means Clustering**. Algoritma ini mengelompokkan semua piksel gambar ke dalam *K* kelompok (sesuai slider Anda) berdasarkan kesamaan warnanya di ruang warna RGB.
    **Manfaat:** Memahami komposisi warna gambar, berguna untuk desain, seni, atau analisis data.
    """)