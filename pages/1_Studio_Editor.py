import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import math

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Studio Restorasi Foto",
    page_icon="🛠️",
    layout="wide"
)

# --- 2. Fungsi Helper (Penting) ---
def pil_to_cv2(pil_image):
    """Konversi PIL Image (RGB) ke CV2 Image (BGR)."""
    try:
        img_input = pil_image
        if img_input.mode == 'RGBA':
            img_input = img_input.convert('RGB')
        return cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error konversi PIL ke CV2: {e}")
        return None

def cv2_to_pil(cv2_image):
    """Konversi CV2 Image (BGR) ke PIL Image (RGB)."""
    try:
        if cv2_image is None: return None
        if len(cv2_image.shape) == 2: return Image.fromarray(cv2_image).convert('RGB')
        if len(cv2_image.shape) == 3: return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return None
    except Exception as e:
        st.error(f"Error konversi CV2 ke PIL: {e}")
        return None

def get_image_download_button(img, filename_pil, operation_name):
    """Membuat tombol download untuk gambar PIL."""
    if img is None: return

    try:
        # Konversi PIL Image ke bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Dapatkan nama file asli (tanpa ekstensi)
        try:
            base_name = filename_pil.rsplit('.', 1)[0]
        except:
            base_name = "gambar"
        
        filename = f"hasil_{base_name}_{operation_name.lower().replace(' ', '_')}.png"
        
        st.download_button(
            label=f"Download Hasil {operation_name} ⬇️",
            data=byte_im,
            file_name=filename,
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error download link: {e}")

# --- 3. Fungsi Logika Restorasi (Inti) ---

# --- REDUKSI NOISE ---
def apply_median_blur(img, ksize_val):
    ksize = max(3, ksize_val if ksize_val % 2 != 0 else ksize_val + 1)
    try: return cv2.medianBlur(img, ksize)
    except: return img

def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    try: return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    except: return img

def apply_nl_means(img):
    try:
        # Parameter (h, hColor, templateWindowSize, searchWindowSize)
        # Ini lambat tapi sangat efektif
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    except: return img

# --- PENAJAMAN & DETAIL ---
def apply_unsharp_mask(img, sigma, strength):
    try:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    except: return img

def apply_clahe(img, clip_limit, grid_size):
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        if len(img.shape) == 2: return clahe.apply(img) # Untuk Grayscale (jika perlu)
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return img
    except: return img
    
def apply_brightness_contrast(img, brightness, contrast):
    alpha = 1.0 + (contrast / 100.0); alpha = max(0.1, alpha)
    beta = brightness
    try: return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    except: return img

# --- RESTORASI INTERAKTIF ---
def apply_inpainting(img, mask_gray, radius, method_flag):
    if mask_gray is None or np.sum(mask_gray) == 0:
        return img
    try:
        mask = mask_gray.astype(np.uint8)
        if len(mask.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        if img.shape[:2] != mask_binary.shape[:2]:
             mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.inpaint(img, mask_binary, radius, flags=method_flag)
    except: return img

# --- 4. Tampilan Utama (UI) ---
st.title("🛠️ Studio Restorasi & Enhancement Foto")

# --- Sidebar ---
with st.sidebar:
    st.title("PANEL KONTROL")
    uploaded_file = st.file_uploader("Upload Foto Anda di Sini", type=["jpg", "png", "jpeg"], key="uploader")
    
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar Asli (Preview)", use_column_width=True)
        
        # Simpan state gambar asli (penting untuk reset)
        st.session_state.original_pil = Image.open(uploaded_file)
        st.session_state.filename = uploaded_file.name
        
        # Inisialisasi gambar yang sedang diproses
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = st.session_state.original_pil.copy()
    
    if 'original_pil' in st.session_state:
        if st.button("Reset ke Asli", use_container_width=True, type="primary"):
            st.session_state.processed_image = st.session_state.original_pil.copy()
            st.success("Gambar telah direset ke asli.")

# --- Area Kerja Utama ---
if 'original_pil' not in st.session_state:
    st.info("Silakan upload gambar di sidebar untuk memulai.")
else:
    # Ambil gambar dari session state
    image_pil_orig = st.session_state.original_pil
    image_pil_processed = st.session_state.processed_image

    # Konversi ke CV2 untuk diproses
    image_cv_orig = pil_to_cv2(image_pil_orig)
    image_cv_processed = pil_to_cv2(image_pil_processed)

    # --- Tampilan Tab ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Noise & Blur (Reduksi Noise)", 
        "Kerusakan (Inpainting)", 
        "Cahaya & Detail (Enhancement)", 
        "Retouch Interaktif (Dodge & Burn)"
    ])

    # Inisialisasi gambar hasil
    final_pil_image = image_pil_processed # Defaultnya adalah gambar terakhir yg diproses
    operation_name = "Current"

    # --- TAB 1: REDUKSI NOISE ---
    with tab1:
        st.header("Reduksi Noise & Blur Ringan")
        st.markdown("Gunakan alat ini untuk membersihkan bintik atau blur pada foto.")
        noise_method = st.radio("Pilih Metode:", 
                                ("Tidak ada", "Median Blur (untuk Bintik)", "Bilateral Filter (Jaga Tepi)", "NL-Means (Kualitas Tinggi, Lambat)"), 
                                key="noise_method", horizontal=True)

        img_noise_reduced = image_cv_processed.copy()
        operation_name = "NoiseReduction"
        
        if noise_method == "Median Blur (untuk Bintik)":
            ksize_median = st.slider("Kekuatan Median", 3, 21, 5, step=2, key="median_ksize")
            img_noise_reduced = apply_median_blur(image_cv_processed, ksize_median)
        
        elif noise_method == "Bilateral Filter (Jaga Tepi)":
            st.info("Bagus untuk menghaluskan kulit sambil menjaga mata tetap tajam.")
            d_bilateral = st.slider("Diameter", 1, 15, 9, key="bilateral_d")
            sc_bilateral = st.slider("Sigma Color", 1, 150, 75, key="bilateral_sc")
            ss_bilateral = st.slider("Sigma Space", 1, 150, 75, key="bilateral_ss")
            img_noise_reduced = apply_bilateral_filter(image_cv_processed, d_bilateral, sc_bilateral, ss_bilateral)

        elif noise_method == "NL-Means (Kualitas Tinggi, Lambat)":
            st.warning("Metode ini mungkin memakan waktu beberapa detik.")
            if st.button("Jalankan NL-Means"):
                with st.spinner("Memproses NL-Means..."):
                    img_noise_reduced = apply_nl_means(image_cv_processed)
            else:
                img_noise_reduced = None # Jangan proses jika tombol belum ditekan
        
        else:
             img_noise_reduced = None # Tidak ada operasi
        
        if img_noise_reduced is not None:
            final_pil_image = cv2_to_pil(img_noise_reduced)


    # --- TAB 2: PERBAIKAN GORESAN (INPAINTING) ---
    with tab2:
        st.header("Perbaikan Goresan & Noda (Inpainting)")
        st.info("Coret area yang rusak (goresan, debu, noda) untuk menghilangkannya.")
        
        col1_i, col2_i = st.columns([1, 2]) # Kolom untuk kanvas dan pengaturan
        
        with col1_i:
            st.markdown("**Pengaturan Kuas Inpainting**")
            stroke_width_inp = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_inp")
            radius_inp = st.slider("Radius Inpainting (Kekuatan)", 1, 15, 3, key="inp_radius")
            method_str_inp = st.radio("Metode:", ("TELEA (Cepat)", "NS (Kualitas Tinggi)"), key="inp_method", horizontal=True)
            method_flag_inp = cv2.INPAINT_TELEA if method_str_inp == "TELEA (Cepat)" else cv2.INPAINT_NS

        with col2_i:
            # --- PERBAIKAN KANVAS HITAM ---
            # Versi baru (v0.10.0) bisa menerima PIL Image
            bg_pil_inp = image_pil_processed.copy()
            aspect_ratio = bg_pil_inp.height / bg_pil_inp.width
            CANVAS_WIDTH = 600
            CANVAS_HEIGHT = min(int(CANVAS_WIDTH * aspect_ratio), 600) 
            bg_pil_resized_inp = bg_pil_inp.resize((CANVAS_WIDTH, CANVAS_HEIGHT))

            canvas_result_inpainting = st_canvas(
                fill_color="rgba(255, 0, 0, 0.5)", # Coretan MERAH TRANSLUSEN
                stroke_width=stroke_width_inp,
                background_image=bg_pil_resized_inp, # <-- PERBAIKAN: Gunakan PIL Image
                update_streamlit=True,
                height=CANVAS_HEIGHT,
                width=CANVAS_WIDTH,
                drawing_mode="freedraw",
                key="canvas_inpainting",
            )
        
        img_inpainted = None
        mask_data_canvas = None
        operation_name = "Inpainting"
        
        if canvas_result_inpainting.image_data is not None:
            # Coretan ada di channel Alpha (indeks 3)
            mask_data_canvas = canvas_result_inpainting.image_data[:, :, 3] 

        if mask_data_canvas is not None and np.sum(mask_data_canvas > 0) > 0:
            with st.spinner("Menerapkan Inpainting..."):
                 mask_for_cv2 = ((mask_data_canvas > 0).astype(np.uint8) * 255)
                 mask_resized_to_orig = cv2.resize(mask_for_cv2, (image_cv_processed.shape[1], image_cv_processed.shape[0]), interpolation=cv2.INTER_NEAREST)
                 img_inpainted = apply_inpainting(image_cv_processed, mask_resized_to_orig, radius_inp, method_flag_inp)
                 final_pil_image = cv2_to_pil(img_inpainted)
        else:
            final_pil_image = image_pil_processed # Tampilkan gambar terakhir jika tidak ada coretan


    # --- TAB 3: PENAJAMAN & DETAIL ---
    with tab3:
        st.header("Penajaman & Detail (Enhancement)")
        st.markdown("Gunakan alat ini untuk memperjelas foto atau memperbaiki kontras.")

        img_enhanced = image_cv_processed.copy()
        
        st.subheader("1. Kecerahan & Kontras Global")
        b_enhance = st.slider("Brightness", -100, 100, 0, key="bc_brightness_enhance")
        c_enhance = st.slider("Contrast", -100, 100, 0, key="bc_contrast_enhance")
        
        st.subheader("2. Kontras Adaptif (CLAHE)")
        st.info("Sangat baik untuk memunculkan detail di area gelap/terang.")
        clip_enhance = st.slider("Clip Limit (Kekuatan)", 1.0, 10.0, 1.0, 0.5, key="clahe_clip_enhance")
        grid_enhance = st.slider("Tile Grid Size (Detail)", 2, 16, 8, key="clahe_grid_enhance")

        st.subheader("3. Penajaman (Unsharp Mask)")
        st.info("Mempertajam gambar dengan lebih terkontrol.")
        sigma_unsharp = st.slider("Sigma (Radius Blur)", 0.1, 5.0, 1.0, 0.1, key="unsharp_sigma_enhance")
        strength_unsharp = st.slider("Strength (Kekuatan)", 0.1, 3.0, 0.0, 0.1, key="unsharp_strength_enhance")
        
        # Terapkan berurutan
        if b_enhance != 0 or c_enhance != 0:
            img_enhanced = apply_brightness_contrast(img_enhanced, b_enhance, c_enhance)
            operation_name = "Brightness"
        
        if clip_enhance > 1.0:
            img_enhanced = apply_clahe(img_enhanced, clip_enhance, grid_enhance)
            operation_name = "CLAHE"
            
        if strength_unsharp > 0.0:
            img_enhanced = apply_unsharp_mask(img_enhanced, sigma_unsharp, strength_unsharp)
            operation_name = "Sharpening"
        
        final_pil_image = cv2_to_pil(img_enhanced)


    # --- TAB 4: RETOUCH INTERAKTIF (DODGE & BURN) ---
    with tab4:
        st.header("Retouch Interaktif (Dodge & Burn)")
        st.info("Pilih mode, lalu coret area yang ingin Anda cerahkan (Dodge) atau gelapkan (Burn).")
        
        # Pengaturan Kuas
        db_mode = st.radio("Pilih Mode Kuas:", ("Dodge (Mencerahkan)", "Burn (Menggelapkan)"), key="db_mode", horizontal=True)
        db_strength = st.slider("Kekuatan Kuas", 1, 50, 20, key="db_strength")
        
        col1_db, col2_db = st.columns([1, 2]) # Kolom untuk kanvas dan pengaturan

        with col1_db:
            st.markdown("**Pengaturan Kuas D&B**")
            stroke_width_db = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_db")
        
        with col2_db:
            # --- PERBAIKAN KANVAS HITAM ---
            bg_pil_db = image_pil_processed.copy()
            aspect_ratio_db = bg_pil_db.height / bg_pil_db.width
            CANVAS_WIDTH_DB = 600
            CANVAS_HEIGHT_DB = min(int(CANVAS_WIDTH_DB * aspect_ratio_db), 600) 
            bg_pil_resized_db = bg_pil_db.resize((CANVAS_WIDTH_DB, CANVAS_HEIGHT_DB))

            stroke_color_db = "rgba(255, 255, 255, 0.3)" if db_mode == "Dodge (Mencerahkan)" else "rgba(0, 0, 0, 0.3)"
            
            canvas_result_db = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width_db,
                stroke_color=stroke_color_db,
                background_image=bg_pil_resized_db, # <-- PERBAIKAN: Gunakan PIL Image
                update_streamlit=True,
                height=CANVAS_HEIGHT_DB,
                width=CANVAS_WIDTH_DB,
                drawing_mode="freedraw",
                key="canvas_db",
            )

        img_db_result = image_cv_processed.copy()
        mask_data_db = None
        operation_name = "DodgeBurn"

        if canvas_result_db.image_data is not None:
            # Ambil masker dari channel Alpha
            mask_data_db = canvas_result_db.image_data[:, :, 3] 

        if mask_data_db is not None and np.sum(mask_data_db > 0) > 0:
            with st.spinner("Menerapkan Dodge/Burn..."):
                strength = db_strength if db_mode == "Dodge (Mencerahkan)" else -db_strength
                image_filtered = apply_brightness_contrast(image_cv_processed, strength, 0)
                
                mask_for_cv2 = ((mask_data_db > 0).astype(np.uint8) * 255)
                mask_resized = cv2.resize(mask_for_cv2, (image_cv_processed.shape[1], image_cv_processed.shape[0]))
                
                mask_3channel = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR) > 0
                
                img_db_result = np.where(mask_3channel, image_filtered, image_cv_processed)
                final_pil_image = cv2_to_pil(img_db_result)
        else:
            final_pil_image = image_pil_processed # Tampilkan gambar terakhir jika tidak ada coretan


    # --- 5. Tampilkan Hasil & Tombol Aksi ---
    st.divider()
    st.header("Perbandingan Hasil")
    
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(image_pil_orig, use_container_width=True)
    
    with col_proc:
        st.markdown(f"**Hasil Proses: {operation_name}**")
        st.image(final_pil_image, use_container_width=True)

    # Tombol Aksi di Bawah Gambar
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        # Tombol untuk menyimpan hasil proses ke state
        if st.button("Terapkan Perubahan Ini", use_container_width=True, type="primary"):
            st.session_state.processed_image = final_pil_image
            st.success("Perubahan diterapkan! Anda bisa lanjut ke alat lain.")
    with col_act2:
        # Tombol download
        get_image_download_button(final_pil_image, st.session_state.filename, operation_name)