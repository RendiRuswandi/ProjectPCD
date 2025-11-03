import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from streamlit_drawable_canvas import st_canvas
import io
import math
import base64

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Studio Editor - PCD",
    page_icon="🔬",
    layout="wide"
)

# Judul Utama
st.title("🔬 Studio Editor PCD")
st.caption("Gunakan Panel Kontrol di sidebar untuk mengunggah gambar dan mulai mengedit.")

# --- Fungsi Helper ---

def pil_to_cv2(pil_image):
    """Mengkonversi PIL Image ke OpenCV BGR Format."""
    try:
        img_input = pil_image
        if img_input.mode == 'RGBA':
            img_input = img_input.convert('RGB')
        # PIL (RGB) -> CV2 (BGR)
        return cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error konversi PIL ke CV2: {e}")
        return None

def cv2_to_pil(cv2_image):
    """Mengkonversi OpenCV BGR/Grayscale ke PIL Image."""
    try:
        if cv2_image is None: return None
        # Grayscale (1 channel)
        if len(cv2_image.shape) == 2: 
            return Image.fromarray(cv2_image).convert('RGB')
        # BGR (3 channels)
        if len(cv2_image.shape) == 3: 
            return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return None
    except Exception as e:
        st.error(f"Error konversi CV2 ke PIL: {e}")
        return None

def get_image_download_button(img_cv2, filename_base, operation_name):
    """Membuat tombol download untuk gambar hasil."""
    if img_cv2 is None: return
    try:
        # Coba ambil ekstensi file asli, default ke png
        try:
            base_name, ext = filename_base.rsplit('.', 1)
            ext = ext.lower() if ext.lower() in ['png', 'jpg', 'jpeg'] else 'png'
            filename = f"hasil_{operation_name.lower().replace(' ', '_')}.{ext}"
            mime = f"image/{ext}"
            
            # Encode gambar (untuk JPEG bisa atur kualitas)
            if ext in ['jpg', 'jpeg']:
                is_success, buffer = cv2.imencode(f".{ext}", img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                is_success, buffer = cv2.imencode(".png", img_cv2)
        except ValueError:
            # Fallback jika nama file tidak memiliki ekstensi
            filename = f"hasil_{operation_name.lower().replace(' ', '_')}.png"
            mime = "image/png"
            is_success, buffer = cv2.imencode(".png", img_cv2)

        if not is_success:
            st.error("Gagal encode gambar download.")
            return
            
        io_buf = io.BytesIO(buffer)
        st.download_button(label=f"Download Hasil {operation_name} ⬇️", data=io_buf, file_name=filename, mime=mime)
    except Exception as e:
        st.error(f"Error download link: {e}")

# --- Fungsi PCD ---

def apply_inpainting(img, mask_gray, radius, method_flag):
    """Menerapkan operasi Inpainting (mengisi lubang pada gambar)."""
    if mask_gray is None or np.sum(mask_gray) == 0:
        st.info("Masker inpainting kosong. Gambar area pada kanvas untuk memulai.")
        return img
    try:
        mask = mask_gray.astype(np.uint8)
        if len(mask.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # Pastikan ukuran mask sama dengan gambar asli (untuk hasil yang akurat)
        if img.shape[:2] != mask_binary.shape[:2]:
            mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.inpaint(img, mask_binary, radius, flags=method_flag)
        else:
            st.warning("Inpainting hanya support gambar BGR 3-channel.")
            return img
    except Exception as e: 
        st.error(f"Error Inpainting: {e}")
        return img

def apply_brightness_contrast(img, brightness, contrast):
    """Mengatur Brightness dan Contrast."""
    alpha = 1.0 + (contrast / 100.0)
    alpha = max(0.1, alpha) # Mencegah alpha terlalu kecil
    beta = brightness
    try: 
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    except Exception as e: 
        st.error(f"Error Brightness/Contrast: {e}")
        return img

def apply_gaussian_blur(img, ksize):
    """Menerapkan Gaussian Blur."""
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_sharpen(img):
    """Menerapkan filter Sharpen sederhana."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(img, -1, kernel)

def apply_sepia(img):
    """Menerapkan efek Sepia."""
    img_sepia = img.copy()
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype=np.float64)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    img_sepia = cv2.transform(img_sepia, kernel)
    img_sepia[np.where(img_sepia > 255)] = 255 # Batasi nilai di 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)

def apply_cold_warm(img, temp_val):
    """Mengatur suhu warna (Cold/Warm)."""
    temp_val = np.clip(temp_val, -100, 100)
    img_out = img.copy().astype(np.float32) / 255.0
    if temp_val > 0: # Hangat (Merah naik, Biru turun)
        img_out[:, :, 2] = np.clip(img_out[:, :, 2] + temp_val / 300.0, 0, 1) # Red (Index 2 di BGR)
        img_out[:, :, 0] = np.clip(img_out[:, :, 0] - temp_val / 300.0, 0, 1) # Blue (Index 0 di BGR)
    elif temp_val < 0: # Dingin (Biru naik, Merah turun)
        img_out[:, :, 0] = np.clip(img_out[:, :, 0] + abs(temp_val) / 300.0, 0, 1)
        img_out[:, :, 2] = np.clip(img_out[:, :, 2] - abs(temp_val) / 300.0, 0, 1)
    return (img_out * 255).astype(np.uint8)

def apply_median_blur(img, ksize):
    """Menerapkan Median Blur (efektif untuk Salt & Pepper Noise)."""
    if ksize % 2 == 0: ksize += 1
    return cv2.medianBlur(img, ksize)

def apply_bilateral_filter(img, d, sc, ss):
    """Menerapkan Bilateral Filter (mereduksi noise sambil menjaga tepi)."""
    return cv2.bilateralFilter(img, d, sc, ss)

def apply_clahe(img, clip, grid):
    """Menerapkan Contrast Limited Adaptive Histogram Equalization."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_unsharp_mask(img, sigma, strength):
    """Menerapkan Unsharp Masking (penajaman)."""
    # Gaussian blur image
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # Weighted sum: Original + (Original - Blurred) * Strength
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)

def apply_rotation(img, angle):
    """Menerapkan Rotasi pada gambar."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    # Menghitung batas baru agar gambar tidak terpotong
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - (w / 2)
    M[1, 2] += (nH / 2) - (h / 2)
    return cv2.warpAffine(img, M, (nW, nH))

def apply_flip(img, flip_code):
    """Menerapkan Flip (Cermin)."""
    return cv2.flip(img, flip_code)

def analyze_color_palette(img, k):
    """Menganalisis palet warna dominan menggunakan K-Means Clustering."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(img_reshaped)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Hitung persentase
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = dominant_colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Konversi ke format Hex
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in sorted_colors]
    return hex_colors, sorted_counts

def get_histogram(img):
    """Menghitung data histogram untuk berbagai channel."""
    hist_data = {}
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_data['Grayscale'] = hist_gray
    
    # RGB
    colors_rgb = ('b', 'g', 'r')
    hist_rgb = {}
    for i, col in enumerate(colors_rgb):
        hist_rgb[col] = cv2.calcHist([img], [i], None, [256], [0, 256])
    hist_data['RGB'] = hist_rgb
    
    # HSV (Hue)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_data['HSV'] = {'H': hist_h}
    
    return hist_data

# --- Sidebar (Hanya untuk upload) ---
with st.sidebar:
    st.title("PANEL KONTROL")
    uploaded_file = st.file_uploader("Upload Gambar Anda di Sini", type=["jpg", "png", "jpeg"], key="uploader")

    if uploaded_file:
        try:
            st.session_state['filename_for_download'] = uploaded_file.name
            
            # Memuat dan mengkonversi gambar
            img_pil_loaded = Image.open(uploaded_file)
            st.session_state['image_pil_orig'] = img_pil_loaded
            st.session_state['image_cv_bgr'] = pil_to_cv2(img_pil_loaded)
            
            # Menghitung ukuran dan menyimpan PIL yang sudah di-resize untuk kanvas (max 600px lebar)
            bg_pil = st.session_state['image_pil_orig'].copy()
            aspect_ratio = bg_pil.height / bg_pil.width
            CANVAS_WIDTH = 600
            CANVAS_HEIGHT = min(int(CANVAS_WIDTH * aspect_ratio), 600) 
            
            st.session_state['canvas_width'] = CANVAS_WIDTH
            st.session_state['canvas_height'] = CANVAS_HEIGHT
            
            bg_pil_resized = bg_pil.resize((CANVAS_WIDTH, CANVAS_HEIGHT))
            st.session_state['canvas_image_resized'] = bg_pil_resized
            
            st.image(st.session_state['image_pil_orig'], caption="Gambar Asli (Preview)", use_column_width=True)
            
        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            # Reset state jika gagal
            uploaded_file = None
            st.session_state['image_pil_orig'] = None
            st.session_state['image_cv_bgr'] = None
            
    if st.button("Reset Gambar Asli", key="reset_button"):
        # Reset semua state terkait gambar
        keys_to_reset = ['image_pil_orig', 'image_cv_bgr', 'filename_for_download', 'canvas_image_resized', 'canvas_width', 'canvas_height']
        for key in keys_to_reset:
             if key in st.session_state: del st.session_state[key]
        st.session_state['filename_for_download'] = "untitled.png" # Set default
        st.experimental_rerun() # Muat ulang halaman untuk reset uploader

# --- Inisialisasi State Awal dan Referensi ---
if 'image_pil_orig' not in st.session_state:
    st.session_state['image_pil_orig'] = None
if 'image_cv_bgr' not in st.session_state:
    st.session_state['image_cv_bgr'] = None
if 'filename_for_download' not in st.session_state:
    st.session_state['filename_for_download'] = "untitled.png" 
if 'canvas_image_resized' not in st.session_state:
    st.session_state['canvas_image_resized'] = None
if 'canvas_width' not in st.session_state:
    st.session_state['canvas_width'] = 600
if 'canvas_height' not in st.session_state:
    st.session_state['canvas_height'] = 400

# Referensi mudah dari session state
image_pil_orig = st.session_state.get('image_pil_orig')
image_cv_bgr = st.session_state.get('image_cv_bgr')
filename_for_download = st.session_state.get('filename_for_download') 

# --- Area Konten Utama ---

# --- Navigasi di HALAMAN UTAMA ---
feature_tab = st.radio(
    "Pilih Kategori Fitur:",
    ("🎞️ Filtering", "🛠️ Restorasi", "✨ Enhancement", "🔄 Transformasi", "🎨 Analisis"),
    key="feature_tab_selector",
    horizontal=True 
)
st.markdown("---") 

if image_cv_bgr is None:
    st.info("Silakan upload gambar di sidebar untuk memulai.")
else:
    # --- Tampilan 1: Filtering ---
    if feature_tab == "🎞️ Filtering":
        st.header("🎞️ Filtering Gambar")
        st.subheader("Pengaturan Filter")
        filter_type = st.radio("Pilih Filter:", 
                               ("Tidak ada", "Gaussian Blur", "Sharpen", "Sepia", "Koreksi Warna (Cold/Warm)"), 
                               key="filter_radio")
        
        img_filtered = image_cv_bgr.copy()
        
        if filter_type == "Gaussian Blur":
            ksize_blur = st.slider("Kekuatan Blur (Kernel Size)", 1, 15, 3, step=2, key="blur_ksize_filter")
            img_filtered = apply_gaussian_blur(image_cv_bgr, ksize_blur) 
        elif filter_type == "Sharpen":
            img_filtered = apply_sharpen(image_cv_bgr) 
        elif filter_type == "Sepia":
            img_filtered = apply_sepia(image_cv_bgr)
        elif filter_type == "Koreksi Warna (Cold/Warm)":
            temp_val = st.slider("Suhu Warna (-100 = Dingin, 100 = Hangat)", -100, 100, 0, 5, key="temp_slider")
            img_filtered = apply_cold_warm(image_cv_bgr, temp_val)
            
        col1_f, col2_f = st.columns(2)
        with col1_f:
            st.markdown("**Original**")
            st.image(image_pil_orig, caption="Gambar Asli", use_column_width=True)
            
        with col2_f:
            st.markdown(f"**Hasil: {filter_type}**")
            img_display_result = cv2_to_pil(img_filtered)
            if img_display_result:
                st.image(img_display_result, caption=f"Hasil: {filter_type}", use_column_width=True)
                get_image_download_button(img_filtered, filename_for_download, filter_type)
            else:
                st.warning("Gagal memproses gambar untuk ditampilkan.")


    # --- Tampilan 2: Restorasi ---
    elif feature_tab == "🛠️ Restorasi":
        st.header("🛠️ Restorasi Citra")
        
        restore_mode = st.radio("Pilih Mode Restorasi:", 
                                ["Reduksi Noise", "(Unik) Inpainting Interaktif", "🖌️ Dodge & Burn Interaktif"], 
                                key="restore_mode_radio")
        
        if restore_mode == "Reduksi Noise":
            st.subheader("Reduksi Noise")
            restore_type = st.radio("Pilih Metode:", ("Tidak ada", "Median Blur", "Bilateral Filter"), key="restore_radio")
            
            img_restored = image_cv_bgr.copy()
            
            if restore_type == "Median Blur":
                ksize_median = st.slider("Kekuatan Median (Kernel Size)", 3, 21, 5, step=2, key="median_ksize_restore")
                img_restored = apply_median_blur(image_cv_bgr, ksize_median)
            elif restore_type == "Bilateral Filter":
                st.info("Bilateral Filter mengurangi noise sambil menjaga tepi tetap tajam.")
                d_bilateral = st.slider("Diameter (d)", 1, 15, 9, key="bilateral_d_restore")
                sc_bilateral = st.slider("Sigma Color (Sensitivitas Warna)", 1, 150, 75, key="bilateral_sc_restore")
                ss_bilateral = st.slider("Sigma Space (Sensitivitas Ruang)", 1, 150, 75, key="bilateral_ss_restore")
                img_restored = apply_bilateral_filter(image_cv_bgr, d_bilateral, sc_bilateral, ss_bilateral)
                
            col1_r, col2_r = st.columns(2)
            with col1_r:
                st.markdown("**Original**")
                st.image(image_pil_orig, caption="Gambar Asli", use_column_width=True)
                
            with col2_r:
                st.markdown(f"**Hasil: {restore_type}**")
                img_display_result = cv2_to_pil(img_restored)
                if img_display_result:
                    st.image(img_display_result, caption=f"Hasil: {restore_type}", use_column_width=True)
                    get_image_download_button(img_restored, filename_for_download, restore_type)
                else:
                    st.warning("Gagal memproses gambar untuk ditampilkan.")

        # --- Inpainting Interaktif ---
        elif restore_mode == "(Unik) Inpainting Interaktif":
            st.subheader("Inpainting Interaktif (Hapus Area)")
            st.info("Gunakan tools di bawah untuk menggambar masker (coretan) pada area yang ingin Anda hilangkan/perbaiki.")
            
            col1_i, col2_i = st.columns(2)
            canvas_result_inpainting = None 

            if st.session_state['canvas_image_resized'] is not None:
                with col1_i:
                    st.markdown("**Kanvas Masking** (Gambar di sini)")
                    stroke_width_inp = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_inp")
                    
                    bg_image_pil = st.session_state['canvas_image_resized']
                    CANVAS_WIDTH = st.session_state['canvas_width']
                    CANVAS_HEIGHT = st.session_state['canvas_height']
                    
                    canvas_result_inpainting = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.5)", 
                        stroke_width=stroke_width_inp,
                        stroke_color="#000000", # Warna kuas tidak penting, hanya alpha yang akan diambil
                        background_image=bg_image_pil, 
                        update_streamlit=True,
                        height=CANVAS_HEIGHT, 
                        width=CANVAS_WIDTH,   
                        drawing_mode="freedraw",
                        key="canvas_inpainting",
                    )
            else: 
                st.warning("Gagal memuat latar belakang kanvas. Coba unggah ulang gambar.")


            with col2_i:
                st.markdown("**Hasil Inpainting**")
                radius_inp = st.slider("Radius Inpainting", 1, 15, 3, key="inp_radius")
                method_str_inp = st.radio("Metode:", ("TELEA", "NS"), key="inp_method")
                method_flag_inp = cv2.INPAINT_TELEA if method_str_inp == "TELEA" else cv2.INPAINT_NS
                
                img_inpainted = None
                mask_data_canvas = None
                
                if canvas_result_inpainting and canvas_result_inpainting.image_data is not None:
                    # Ambil channel Alpha (indeks 3) dari data gambar kanvas
                    mask_data_canvas = canvas_result_inpainting.image_data[:, :, 3] 
                
                if mask_data_canvas is not None and np.sum(mask_data_canvas > 0) > 0:
                    with st.spinner("Menerapkan Inpainting..."):
                        # Konversi data mask ke format yang bisa diproses CV2 (hitam putih)
                        mask_for_cv2 = ((mask_data_canvas > 0).astype(np.uint8) * 255)
                        # Resize mask agar ukurannya sama dengan gambar asli
                        mask_resized_to_orig = cv2.resize(mask_for_cv2, (image_cv_bgr.shape[1], image_cv_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_inpainted = apply_inpainting(image_cv_bgr, mask_resized_to_orig, radius_inp, method_flag_inp)
                        
                    st.image(cv2_to_pil(img_inpainted), caption="Hasil Inpainting", use_column_width=True)
                    get_image_download_button(img_inpainted, filename_for_download, "Inpainting")
                else:
                    st.image(image_pil_orig, caption="Gambar Asli (Belum ada masker)", use_column_width=True)

        # --- Dodge & Burn Interaktif ---
        elif restore_mode == "🖌️ Dodge & Burn Interaktif":
            st.subheader("Dodge & Burn Interaktif")
            st.info("Pilih mode, lalu coret area yang ingin Anda cerahkan (Dodge) atau gelapkan (Burn).")
            
            col_db_mode, col_db_strength = st.columns(2)
            with col_db_mode:
                db_mode = st.radio("Pilih Mode Kuas:", ("Dodge (Mencerahkan)", "Burn (Menggelapkan)"), key="db_mode")
            with col_db_strength:
                db_strength = st.slider("Kekuatan Kuas", 1, 50, 20, key="db_strength")
            
            col1_db, col2_db = st.columns(2)
            canvas_result_db = None 

            if st.session_state['canvas_image_resized'] is not None:
                with col1_db:
                    st.markdown("**Kanvas Dodge & Burn** (Gambar di sini)")
                    stroke_width_db = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_db")
                    
                    background_image_db = st.session_state['canvas_image_resized']
                    CANVAS_WIDTH_DB = st.session_state['canvas_width']
                    CANVAS_HEIGHT_DB = st.session_state['canvas_height']
                    
                    stroke_color_db = "rgba(0, 0, 0, 0)" # Warna tidak penting, hanya untuk memaksa canvas_data_db
                    
                    canvas_result_db = st_canvas(
                        fill_color="rgba(0, 0, 0, 0)", 
                        stroke_width=stroke_width_db,
                        stroke_color=stroke_color_db, 
                        background_image=background_image_db, 
                        update_streamlit=True,
                        height=CANVAS_HEIGHT_DB, 
                        width=CANVAS_WIDTH_DB,   
                        drawing_mode="freedraw",
                        key="canvas_db",
                    )
            else:
                st.warning("Gagal memuat latar belakang kanvas. Coba unggah ulang gambar.")


            with col2_db:
                st.markdown("**Hasil Dodge & Burn**")
                
                img_db_result = image_cv_bgr.copy()
                mask_data_db = None

                if canvas_result_db and canvas_result_db.image_data is not None:
                    # Mengambil data gambar dari kanvas (coretan)
                    mask_data_db_rgba = canvas_result_db.image_data
                
                    # Hanya proses jika ada coretan
                    if np.sum(mask_data_db_rgba[:, :, 3] > 0) > 0:
                        with st.spinner("Menerapkan Dodge/Burn..."):
                            
                            # 1. Hitung nilai kecerahan yang akan diterapkan (sekitar 0-255)
                            strength_b = db_strength * 2.5 
                            brightness_value = strength_b if db_mode == "Dodge (Mencerahkan)" else -strength_b
                            
                            # 2. Buat gambar yang sudah diberi efek (tanpa masking)
                            image_filtered = apply_brightness_contrast(image_cv_bgr, brightness_value, 0)
                            
                            # 3. Ambil alpha mask dari coretan
                            mask_for_cv2 = mask_data_db_rgba[:, :, 3]
                            # Resize mask agar ukurannya sama dengan gambar asli
                            mask_resized = cv2.resize(mask_for_cv2, (image_cv_bgr.shape[1], image_cv_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                            
                            # 4. Blur mask untuk transisi yang halus
                            mask_resized_blurred = cv2.GaussianBlur(mask_resized, (0, 0), sigmaX=db_strength/5.0)
                            
                            # 5. Normalisasi dan buat mask 3-channel
                            mask_float = mask_resized_blurred.astype(np.float32) / 255.0
                            mask_3channel = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
                            
                            # 6. Gabungkan gambar asli dan gambar terfilter menggunakan mask (Blending)
                            img_db_result = (image_filtered.astype(np.float32) * mask_3channel) + \
                                            (image_cv_bgr.astype(np.float32) * (1 - mask_3channel))
                            img_db_result = np.clip(img_db_result, 0, 255).astype(np.uint8)

                        st.image(cv2_to_pil(img_db_result), caption="Hasil Dodge & Burn", use_column_width=True)
                        get_image_download_button(img_db_result, filename_for_download, "DodgeBurn")
                    else:
                        st.image(image_pil_orig, caption="Gambar Asli (Belum ada coretan)", use_column_width=True)
                else:
                    st.image(image_pil_orig, caption="Gambar Asli (Belum ada coretan)", use_column_width=True)


    # --- Tampilan 3: Enhancement ---
    elif feature_tab == "✨ Enhancement":
        st.header("✨ Enhancement Citra")
        st.subheader("Pengaturan Enhancement")
        enhance_type = st.radio("Pilih Metode:", ("Tidak ada", "Brightness / Contrast", "CLAHE", "Unsharp Masking"), key="enhance_radio")

        img_enhanced = image_cv_bgr.copy()

        if enhance_type == "Brightness / Contrast":
            b_enhance = st.slider("Brightness (Nilai Penambahan)", -100, 100, 0, key="bc_brightness_enhance")
            c_enhance = st.slider("Contrast (Nilai Perkalian)", -100, 100, 0, key="bc_contrast_enhance")
            img_enhanced = apply_brightness_contrast(image_cv_bgr, b_enhance, c_enhance)
        elif enhance_type == "CLAHE":
            st.info("CLAHE meningkatkan kontras lokal tanpa memperkuat noise.")
            clip_enhance = st.slider("Clip Limit", 1.0, 10.0, 2.5, 0.5, key="clahe_clip_enhance")
            grid_enhance = st.slider("Tile Grid Size", 2, 16, 8, key="clahe_grid_enhance")
            img_enhanced = apply_clahe(image_cv_bgr, clip_enhance, grid_enhance)
        elif enhance_type == "Unsharp Masking":
            st.info("Unsharp Masking menajamkan gambar berdasarkan detail yang di-blur.")
            sigma_unsharp = st.slider("Sigma (Radius Blur)", 0.1, 5.0, 1.0, 0.1, key="unsharp_sigma_enhance")
            strength_unsharp = st.slider("Strength (Kekuatan Efek)", 0.1, 3.0, 1.5, 0.1, key="unsharp_strength_enhance")
            img_enhanced = apply_unsharp_mask(image_cv_bgr, sigma_unsharp, strength_unsharp)

        col1_e, col2_e = st.columns(2)
        with col1_e:
            st.markdown("**Original**")
            st.image(image_pil_orig, caption="Gambar Asli", use_column_width=True)
            
        with col2_e:
            st.markdown(f"**Hasil: {enhance_type}**")
            img_display_result = cv2_to_pil(img_enhanced)
            if img_display_result:
                st.image(img_display_result, caption=f"Hasil: {enhance_type}", use_column_width=True)
                get_image_download_button(img_enhanced, filename_for_download, enhance_type)
            else:
                st.warning("Gagal memproses gambar untuk ditampilkan.")

    # --- Tampilan 4: Transformasi ---
    elif feature_tab == "🔄 Transformasi":
        st.header("🔄 Transformasi Gambar")
        
        st.subheader("Rotasi (Putar Gambar)")
        angle = st.slider("Sudut Rotasi (Searah Jarum Jam)", -180, 180, 0, 1, key="rotation_angle")
        
        st.subheader("Flip (Cermin)")
        flip_type = st.radio("Pilih Tipe Flip:", 
                              ("Tidak ada", "Horizontal (Kiri/Kanan)", "Vertikal (Atas/Bawah)"), 
                              key="flip_radio") 
        
        img_transformed = image_cv_bgr.copy()
        operation_name = "Transformasi"
        
        # Rotasi
        if angle != 0:
            img_transformed = apply_rotation(img_transformed, angle)
            
        final_img = img_transformed 
        
        # Flip
        if flip_type == "Horizontal (Kiri/Kanan)":
            final_img = apply_flip(img_transformed, 1) 
        elif flip_type == "Vertikal (Atas/Bawah)":
            final_img = apply_flip(img_transformed, 0) 
            
        # Update nama operasi untuk download
        if angle != 0 and flip_type != "Tidak ada":
            operation_name = f"Rotasi{angle} & {flip_type.split(' ')[0]}"
        elif angle != 0:
            operation_name = f"Rotasi{angle}"
        elif flip_type != "Tidak ada":
            operation_name = flip_type.split(' ')[0]
        else:
            operation_name = "Original" # Jika tidak ada transformasi

        
        col1_t, col2_t = st.columns(2)
        with col1_t:
            st.markdown("**Original**")
            st.image(image_pil_orig, caption="Gambar Asli", use_column_width=True)
        
        with col2_t:
            st.markdown(f"**Hasil {operation_name}**")
            
            img_display_result = cv2_to_pil(final_img) 
            
            if img_display_result:
                st.image(img_display_result, caption=f"Hasil: {operation_name}", use_column_width=True)
                if operation_name != "Original":
                    get_image_download_button(final_img, filename_for_download, operation_name)
            else:
                st.warning("Gagal memproses gambar untuk ditampilkan.")

    # --- Tampilan 5: Analisis (Fitur Unik) ---
    elif feature_tab == "🎨 Analisis":
        st.header("🎨 Analisis Citra")
        st.info("Fitur ini menganalisis gambar asli Anda tanpa mengubahnya.")
        
        st.subheader("Analisis Palet Warna")
        k_colors_analyze = st.slider("Jumlah Warna Dominan (K)", 2, 10, 5, key="k_colors_analyze")
        
        with st.spinner("Menganalisis palet..."):
            dom_colors_res, counts_res = analyze_color_palette(image_cv_bgr, k_colors_analyze)
        
        if dom_colors_res:
            cols_color_res = st.columns(len(dom_colors_res)) 
            total_pixels_res = sum(counts_res) if counts_res is not None else 1
            st.markdown('<style>div.stDownloadButton {text-align: center;}</style>', unsafe_allow_html=True)
            
            for i, color_hex_res in enumerate(dom_colors_res):
                with cols_color_res[i]:
                    # Display color box using Markdown HTML
                    st.markdown(
                        f'<div style="background-color:{color_hex_res}; width:100%; height:50px; border: 1px solid grey; margin:auto;"></div>',
                        unsafe_allow_html=True
                    )
                    st.code(color_hex_res, language='text')
                    if counts_res is not None and i < len(counts_res):
                        percentage = (counts_res[i] / total_pixels_res) * 100
                        st.caption(f"Populasi: **{percentage:.1f}%**")
        else:
            st.warning("Gagal menganalisis palet.")

        st.markdown("---") 

        st.subheader("Analisis Histogram")
        hist_channel_select = st.selectbox("Pilih Channel:", ('Grayscale', 'RGB', 'HSV'), key="hist_channel_analyze")
        
        with st.spinner("Menghitung histogram..."):
            hist_data_res = get_histogram(image_cv_bgr)
        
        if hist_data_res:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            plt.style.use('dark_background') # Membuat plot lebih enak dilihat di Streamlit
            ax_hist.set_xlabel("Nilai Piksel")
            ax_hist.set_ylabel("Frekuensi (# Piksel)")

            if hist_channel_select == 'Grayscale':
                ax_hist.set_title("Histogram Grayscale")
                ax_hist.plot(hist_data_res['Grayscale'], color='gray')
                ax_hist.set_xlim([0, 256])
            elif hist_channel_select == 'RGB':
                ax_hist.set_title("Histogram RGB")
                colors_rgb = ('b', 'g', 'r')
                for i, col in enumerate(colors_rgb):
                    ax_hist.plot(hist_data_res['RGB'][col], color=col, alpha=0.7)
                    ax_hist.set_xlim([0, 256])
            elif hist_channel_select == 'HSV':
                ax_hist.set_title("Histogram HSV (Hue)")
                ax_hist.plot(hist_data_res['HSV']['H'], color='red')
                ax_hist.set_xlim([0, 180])
            
            st.pyplot(fig_hist)
        else:
            st.warning("Gagal menghitung histogram.")