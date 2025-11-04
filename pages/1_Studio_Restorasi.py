import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw # Import ImageDraw untuk kasus tertentu (walau tidak dipakai saat ini)
from streamlit_drawable_canvas import st_canvas
import io
import math

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Studio Restorasi",
    page_icon="🛠️",
    layout="wide"
)

# --- 2. Fungsi Helper (Penting) ---
def pil_to_cv2(pil_image):
    """Konversi PIL Image ke CV2 (BGR)."""
    try:
        img_input = pil_image
        if img_input.mode == 'RGBA':
            img_input = img_input.convert('RGB')
        return cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error konversi PIL ke CV2: {e}")
        return None

def cv2_to_pil(cv2_image):
    """Konversi CV2 (BGR) ke PIL Image."""
    try:
        if cv2_image is None: return None
        if len(cv2_image.shape) == 2: return Image.fromarray(cv2_image).convert('RGB')
        if len(cv2_image.shape) == 3: return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return None
    except Exception as e:
        st.error(f"Error konversi CV2 ke PIL: {e}")
        return None

def get_image_download_button(img_pil, filename_base, operation_name):
    """Membuat tombol download untuk gambar hasil proses."""
    if img_pil is None: return
    try:
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        try:
            base_name = filename_base.rsplit('.', 1)[0]
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
def apply_median_blur(img_cv, ksize_val):
    ksize = max(3, ksize_val if ksize_val % 2 != 0 else ksize_val + 1)
    try: return cv2.medianBlur(img_cv, ksize)
    except: return img_cv

def apply_bilateral_filter(img_cv, d, sigma_color, sigma_space):
    try: return cv2.bilateralFilter(img_cv, d, sigma_color, sigma_space)
    except: return img_cv

def apply_unsharp_mask(img_cv, sigma, strength):
    try:
        blurred = cv2.GaussianBlur(img_cv, (0, 0), sigma)
        sharpened = cv2.addWeighted(img_cv, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    except: return img_cv

def apply_clahe(img_cv, clip_limit, grid_size):
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        if len(img_cv.shape) == 3:
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return img_cv
    except: return img_cv
    
def apply_brightness_contrast(img_cv, brightness, contrast):
    alpha = 1.0 + (contrast / 100.0)
    alpha = max(0.1, alpha) 
    beta = brightness
    try: return cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
    except: return img_cv

def apply_inpainting(img_cv, mask_gray, radius, method_flag):
    """Fungsi inpainting, membutuhkan mask abu-abu yang sudah di-resize."""
    if mask_gray is None or np.sum(mask_gray) == 0:
        return img_cv
    try:
        mask = mask_gray.astype(np.uint8)
        _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(img_cv, mask_binary, radius, flags=method_flag)
    except Exception as e: 
        st.error(f"Error Inpainting: {e}")
        return img_cv

# --- FUNGSI UMUM KANVAS (FIXED) ---
def run_canvas(pil_image, key, stroke_width, stroke_color, fill_color="rgba(0, 0, 0, 0)"):
    """Menjalankan kanvas drawable dengan gambar sebagai latar belakang yang ditumpuk."""
    # Tentukan ukuran
    aspect_ratio = pil_image.height / pil_image.width
    CANVAS_WIDTH = 600
    CANVAS_HEIGHT = min(int(CANVAS_WIDTH * aspect_ratio), 600)
    
    # Resize gambar untuk latar belakang
    bg_pil_resized = pil_image.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
    
    # Paksa RGBA (penting untuk st_canvas)
    if bg_pil_resized.mode != 'RGBA':
        bg_pil_resized = bg_pil_resized.convert('RGBA')

    # --- Kunci: Tumpukan Gambar dan Kanvas dengan CSS ---
    st.markdown(f"""
        <style>
        .canvas-stack-container {{
            position: relative;
            width: {CANVAS_WIDTH}px;
            height: {CANVAS_HEIGHT}px;
        }}
        .canvas-stack-container .stImage > div > img {{
            position: absolute;
            top: 0;
            left: 0;
            width: {CANVAS_WIDTH}px !important;
            height: {CANVAS_HEIGHT}px !important;
            object-fit: contain; 
        }}
        .canvas-stack-container div[data-testid="stCanvas"] {{
            position: absolute; 
            top: 0;
            left: 0;
            z-index: 10;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="canvas-stack-container">', unsafe_allow_html=True)
    
    # Lapisan 1: Gambar (terlihat)
    st.image(bg_pil_resized, width=CANVAS_WIDTH)
    
    # Lapisan 2: Kanvas (transparan)
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width, # Diambil dari parameter
        stroke_color=stroke_color,
        background_color="rgba(0, 0, 0, 0)", # Transparan
        background_image=None, 
        update_streamlit=True,
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        key=key,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return canvas_result

# --- 4. Tampilan Utama (UI) ---
st.title("🛠️ Studio Restorasi & Enhancement Foto")

# --- Sidebar ---
with st.sidebar:
    st.title("PANEL KONTROL")
    uploaded_file = st.file_uploader("Upload Foto Anda di Sini", type=["jpg", "png", "jpeg"], key="uploader")
    
    if uploaded_file:
        st.session_state.original_pil = Image.open(uploaded_file)
        st.session_state.filename = uploaded_file.name
        if 'processed_image' not in st.session_state or st.session_state.get('last_uploaded_name') != uploaded_file.name:
            st.session_state.processed_image = st.session_state.original_pil.copy()
            st.session_state.last_uploaded_name = uploaded_file.name
        st.image(st.session_state.original_pil, caption="Gambar Asli (Preview)", use_column_width=True)
    
    if 'original_pil' in st.session_state:
        if st.button("Reset ke Asli", key="reset_button_main"):
            st.session_state.processed_image = st.session_state.original_pil.copy()
            st.success("Gambar telah direset ke asli.")

# --- Area Kerja Utama ---
if 'original_pil' not in st.session_state:
    st.info("Silakan upload gambar di sidebar untuk memulai.")
else:
    image_pil_orig = st.session_state.original_pil
    image_pil_processed = st.session_state.processed_image
    image_cv_processed = pil_to_cv2(image_pil_processed)
    
    if image_cv_processed is None:
        st.error("Gagal memproses gambar. Mohon upload ulang file.")
        
    feature_tab = st.radio(
        "Pilih Alat Restorasi:",
        ("Auto (Noise & Detail)", "Manual (Inpainting)", "Manual (Dodge & Burn)"),
        key="feature_tab_selector",
        horizontal=True
    )
    st.markdown("---")

    final_pil_image = image_pil_processed
    operation_name = "Current"

    # --- TAB 1: AUTO (NOISE & DETAIL) ---
    if feature_tab == "Auto (Noise & Detail)":
        st.header("Restorasi & Enhancement Otomatis")
        col_noise, col_enhance = st.columns(2)
        with col_noise:
            st.subheader("Reduksi Noise")
            noise_method = st.radio("Pilih Metode:", ("Tidak ada", "Median (Bintik)", "Bilateral (Jaga Tepi)"), key="noise_method")
            img_noise_reduced = image_cv_processed.copy()
            
            if noise_method == "Median (Bintik)":
                ksize_median = st.slider("Kekuatan Median", 3, 21, 5, step=2, key="median_ksize")
                img_noise_reduced = apply_median_blur(img_noise_reduced, ksize_median)
            elif noise_method == "Bilateral (Jaga Tepi)":
                d_bilateral = st.slider("Diameter", 1, 15, 9, key="bilateral_d")
                sc_bilateral = st.slider("Sigma Color", 1, 150, 75, key="bilateral_sc")
                img_noise_reduced = apply_bilateral_filter(img_noise_reduced, d_bilateral, sc_bilateral, sc_bilateral)
                
        with col_enhance:
            st.subheader("Peningkatan Detail & Cahaya")
            b_enhance = st.slider("Brightness", -100, 100, 0, key="bc_brightness_enhance")
            c_enhance = st.slider("Contrast", -100, 100, 0, key="bc_contrast_enhance")
            clip_enhance = st.slider("Clip Limit (CLAHE)", 1.0, 10.0, 1.0, 0.5, key="clahe_clip_enhance")
            strength_unsharp = st.slider("Strength (Unsharp Mask)", 0.0, 3.0, 0.0, 0.1, key="unsharp_strength_enhance")

        img_auto_result = img_noise_reduced
        operation_name = noise_method
        
        if b_enhance != 0 or c_enhance != 0:
            img_auto_result = apply_brightness_contrast(img_auto_result, b_enhance, c_enhance)
            operation_name += "+Brightness"
        
        if clip_enhance > 1.0:
            img_auto_result = apply_clahe(img_auto_result, clip_enhance, 8)
            operation_name += "+CLAHE"
            
        if strength_unsharp > 0.0:
            img_auto_result = apply_unsharp_mask(img_auto_result, 1.0, strength_unsharp)
            operation_name += "+Sharpen"
            
        final_pil_image = cv2_to_pil(img_auto_result)

    # --- TAB 2: MANUAL (INPAINTING) ---
    elif feature_tab == "Manual (Inpainting)":
        st.header("Perbaikan Goresan & Noda (Inpainting)")
        col1_i, col2_i = st.columns([1, 2])
        
        with col1_i:
            st.markdown("**Pengaturan Kuas Inpainting**")
            stroke_width = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_inp") # FIX: Definisi stroke_width
            st.markdown("**Pengaturan Inpainting**")
            radius_inp = st.slider("Radius Perbaikan", 1, 15, 3, key="inp_radius")
            method_str_inp = st.radio("Metode:", ("TELEA (Cepat)", "NS (Kualitas Tinggi)"), key="inp_method")
            method_flag_inp = cv2.INPAINT_TELEA if method_str_inp == "TELEA (Cepat)" else cv2.INPAINT_NS
            
        with col2_i:
            st.markdown("**Kanvas Masking** (Gambar di sini)")
            canvas_result = run_canvas(
                image_pil_processed, 
                key="canvas_inpainting", 
                stroke_width=stroke_width, # FIX: Meneruskan stroke_width
                stroke_color="rgba(255, 0, 0, 0.7)" # Kuas Merah
            )
        
        operation_name = "Inpainting"
        if canvas_result.image_data is not None:
            mask_data_canvas = canvas_result.image_data[:, :, 3] # Ambil Alpha channel
            if np.sum(mask_data_canvas > 0) > 0:
                with st.spinner("Menerapkan Inpainting..."):
                    mask_for_cv2 = ((mask_data_canvas > 0).astype(np.uint8) * 255)
                    mask_resized = cv2.resize(
                        mask_for_cv2, 
                        (image_cv_processed.shape[1], image_cv_processed.shape[0]), 
                        interpolation=cv2.INTER_NEAREST 
                    )
                    img_inpainted = apply_inpainting(image_cv_processed, mask_resized, radius_inp, method_flag_inp)
                    final_pil_image = cv2_to_pil(img_inpainted)

    # --- TAB 3: MANUAL (DODGE & BURN) ---
    elif feature_tab == "Manual (Dodge & Burn)":
        st.header("Retouch Interaktif (Dodge & Burn)")
        col1_db, col2_db = st.columns([1, 2])
        
        with col1_db:
            st.markdown("**Pengaturan Kuas D&B**")
            db_mode = st.radio("Pilih Mode Kuas:", ("Dodge (Mencerahkan)", "Burn (Menggelapkan)"), key="db_mode")
            db_strength = st.slider("Kekuatan Kuas", 1, 50, 20, key="db_strength")
            stroke_width = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_db") # FIX: Definisi stroke_width
            
            stroke_color = "rgba(255, 255, 255, 0.3)" if db_mode == "Dodge (Mencerahkan)" else "rgba(0, 0, 0, 0.3)"
            
        with col2_db:
            st.markdown("**Kanvas Dodge & Burn** (Gambar di sini)")
            canvas_result = run_canvas(
                image_pil_processed, 
                key="canvas_db", 
                stroke_width=stroke_width, # FIX: Meneruskan stroke_width
                stroke_color=stroke_color
            )

        operation_name = "DodgeBurn"
        if canvas_result.image_data is not None:
            mask_data_canvas = canvas_result.image_data[:, :, 3] # Ambil Alpha channel
            if np.sum(mask_data_canvas > 0) > 0:
                with st.spinner("Menerapkan Dodge/Burn..."):
                    strength = db_strength 
                    
                    if db_mode == "Dodge (Mencerahkan)":
                         image_filtered = apply_brightness_contrast(image_cv_processed, strength, 0)
                    else: # Burn (Menggelapkan)
                         image_filtered = apply_brightness_contrast(image_cv_processed, -strength, 0)
                    
                    mask_for_cv2 = ((mask_data_canvas > 0).astype(np.uint8) * 255)
                    mask_resized = cv2.resize(
                        mask_for_cv2, 
                        (image_cv_processed.shape[1], image_cv_processed.shape[0]), 
                        interpolation=cv2.INTER_NEAREST 
                    )
                    
                    mask_3channel = (cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR) > 0)
                    img_db_result = np.where(mask_3channel, image_filtered, image_cv_processed)
                    
                    final_pil_image = cv2_to_pil(img_db_result.astype(np.uint8))


    # --- 5. Tampilkan Hasil & Tombol Aksi ---
    st.markdown("---")
    st.header("Perbandingan Hasil")
    
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(image_pil_orig, use_column_width=True)
    with col_proc:
        st.markdown(f"**Hasil Proses: {operation_name}**")
        st.image(final_pil_image, use_column_width=True)

    st.markdown("---")
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        # Pengecekan penting: pastikan final_pil_image benar-benar berbeda dari state sebelumnya
        if final_pil_image != st.session_state.processed_image:
             if st.button("Terapkan Perubahan Ini"):
                st.session_state.processed_image = final_pil_image
                st.success("Perubahan diterapkan! Anda bisa lanjut ke alat lain.")
        else:
            st.info("Belum ada perubahan yang perlu diterapkan.")

    with col_act2:
        get_image_download_button(final_pil_image, st.session_state.filename, operation_name)