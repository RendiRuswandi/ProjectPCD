import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from streamlit_drawable_canvas import st_canvas
import io
import math

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Studio Editor - PCD",
    page_icon="🔬",
    layout="wide"
)

# --- Fungsi Helper ---
def pil_to_cv2(pil_image):
    try:
        img_input = pil_image
        if img_input.mode == 'RGBA':
            img_input = img_input.convert('RGB')
        return cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error konversi PIL ke CV2: {e}")
        return None

def cv2_to_pil(cv2_image):
    try:
        if cv2_image is None: return None
        if len(cv2_image.shape) == 2: return Image.fromarray(cv2_image).convert('RGB')
        if len(cv2_image.shape) == 3: return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return None
    except Exception as e:
        st.error(f"Error konversi CV2 ke PIL: {e}")
        return None

def get_image_download_button(img_cv2, filename_base, operation_name):
    if img_cv2 is None: return
    try:
        try:
            base_name, ext = filename_base.rsplit('.', 1)
            if ext.lower() not in ['png', 'jpg', 'jpeg']: ext = 'png'
            filename = f"hasil_{operation_name.lower().replace(' ', '_')}.{ext}"
            mime = f"image/{ext}"
            if ext.lower() in ['jpg', 'jpeg']:
                is_success, buffer = cv2.imencode(f".{ext}", img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                is_success, buffer = cv2.imencode(".png", img_cv2)
        except ValueError:
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

# Filtering
def apply_gaussian_blur(img, ksize_val):
    ksize = max(1, (ksize_val * 2) + 1)
    try: return cv2.GaussianBlur(img, (ksize, ksize), 0)
    except Exception as e: st.error(f"Error Gaussian Blur: {e}"); return img

def apply_sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    try: return cv2.filter2D(img, -1, kernel)
    except Exception as e: st.error(f"Error Sharpen: {e}"); return img

# Restorasi
def apply_median_blur(img, ksize_val):
    ksize = max(3, ksize_val if ksize_val % 2 != 0 else ksize_val + 1)
    try: return cv2.medianBlur(img, ksize)
    except Exception as e: st.error(f"Error Median Blur: {e}"); return img

def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    try: return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    except Exception as e: st.error(f"Error Bilateral Filter: {e}"); return img

def apply_inpainting(img, mask_gray, radius, method_flag):
    if mask_gray is None or np.sum(mask_gray) == 0:
        st.info("Masker inpainting kosong. Gambar area pada kanvas untuk memulai.")
        return img
    try:
        mask = mask_gray.astype(np.uint8)
        if len(mask.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        
        if img.shape[:2] != mask_binary.shape[:2]:
             st.warning(f"Menyesuaikan ukuran masker dari {mask_binary.shape} ke {img.shape[:2]}...")
             mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if len(img.shape) == 3 and img.shape[2] == 3: # BGR
            return cv2.inpaint(img, mask_binary, radius, flags=method_flag)
        else:
             st.warning("Inpainting hanya support gambar BGR 3-channel.")
             return img
    except Exception as e: st.error(f"Error Inpainting: {e}"); return img

# Enhancement
def apply_clahe(img, clip_limit, grid_size):
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        if len(img.shape) == 2: return clahe.apply(img)
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return img
    except Exception as e: st.error(f"Error CLAHE: {e}"); return img

def apply_brightness_contrast(img, brightness, contrast):
    alpha = 1.0 + (contrast / 100.0); alpha = max(0.1, alpha)
    beta = brightness
    try: return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    except Exception as e: st.error(f"Error Brightness/Contrast: {e}"); return img

def apply_unsharp_mask(img, sigma, strength):
    try:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    except Exception as e: st.error(f"Error Unsharp Mask: {e}"); return img

# Analisis
def analyze_color_palette(img, num_colors):
    try:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale_percent = 50 
        if image_rgb.shape[0] > 600 or image_rgb.shape[1] > 600:
             width = int(image_rgb.shape[1] * scale_percent / 100)
             height = int(image_rgb.shape[0] * scale_percent / 100)
             dim = (width, height)
             image_rgb_small = cv2.resize(image_rgb, dim, interpolation = cv2.INTER_AREA)
        else:
             image_rgb_small = image_rgb

        pixels = image_rgb_small.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42) # n_init=10
        kmeans.fit(pixels)
        
        dominant_colors_rgb = kmeans.cluster_centers_.astype(int)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors_rgb = dominant_colors_rgb[sorted_indices]
        counts = counts[sorted_indices]
        dominant_colors_hex = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in dominant_colors_rgb]
        return dominant_colors_hex, counts
    except Exception as e: st.error(f"Error Analisis Palet: {e}"); return [], []

def get_histogram(img):
    try:
        hist_data = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_data['Grayscale'] = cv2.calcHist([gray], [0], None, [256], [0, 256])
        color_rgb = ('b', 'g', 'r')
        hist_data['RGB'] = {}
        for i, col in enumerate(color_rgb):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist_data['RGB'][col] = hist
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_data['HSV'] = {}
        hist_data['HSV']['H'] = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_data['HSV']['S'] = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_data['HSV']['V'] = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        return hist_data
    except Exception as e: st.error(f"Error Histogram: {e}"); return {}


# --- UI STREAMLIT ---
st.title("🔬 Studio Editor PCD")
st.caption("Gunakan Panel Kontrol di sidebar untuk mengunggah gambar dan mulai mengedit.")

# --- Sidebar ---
image_pil_orig = None
image_cv_bgr = None
filename_for_download = "untitled.png" 

# --- Navigasi dipindahkan ke Sidebar (menggantikan st.tabs) ---
feature_tab = st.sidebar.radio(
    "Pilih Kategori Fitur:",
    ("🎞️ Filtering", "🛠️ Restorasi", "✨ Enhancement", "🎨 Analisis"),
    key="feature_tab_selector"
)

with st.sidebar:
    st.title("PANEL KONTROL")
    uploaded_file = st.file_uploader("Upload Gambar Anda di Sini", type=["jpg", "png", "jpeg"], key="uploader")

    if uploaded_file:
        try:
            filename_for_download = uploaded_file.name
            image_pil_orig = Image.open(uploaded_file)
            image_cv_bgr = pil_to_cv2(image_pil_orig)
            st.image(image_pil_orig, caption="Gambar Asli (Preview)", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            uploaded_file = None
            
    if st.button("Reset Gambar Asli", key="reset_button", disabled=(uploaded_file is None)):
        st.info("Fitur reset masih dalam pengembangan. Silakan upload ulang gambar.")

# --- Area Konten Utama ---
if uploaded_file is None or image_cv_bgr is None:
    st.info("Silakan upload gambar di sidebar untuk memulai.")
else:
    # --- Gunakan if/elif berdasarkan st.sidebar.radio ---

    # --- Tampilan 1: Filtering ---
    if feature_tab == "🎞️ Filtering":
        st.header("🎞️ Filtering Gambar")
        st.subheader("Pengaturan Filter")
        filter_type = st.radio("Pilih Filter:", ("Tidak ada", "Gaussian Blur", "Sharpen"), key="filter_radio")
        
        img_filtered = image_cv_bgr.copy()
        
        if filter_type == "Gaussian Blur":
            ksize_blur = st.slider("Kekuatan Blur", 1, 15, 3, key="blur_ksize_filter")
            img_filtered = apply_gaussian_blur(image_cv_bgr, ksize_blur)
        elif filter_type == "Sharpen":
            img_filtered = apply_sharpen(image_cv_bgr)
            
        col1_f, col2_f = st.columns(2)
        with col1_f:
            fig_orig, ax_orig = plt.subplots()
            ax_orig.imshow(image_pil_orig) 
            ax_orig.set_title("Original")
            ax_orig.axis('off')
            st.pyplot(fig_orig) # Hapus use_column_width
            
        with col2_f:
            fig_res, ax_res = plt.subplots()
            ax_res.set_title(f"Hasil: {filter_type}")
            ax_res.axis('off')
            img_display_result = cv2_to_pil(img_filtered)
            if img_display_result:
                if len(img_filtered.shape) == 2:
                    ax_res.imshow(np.array(img_display_result), cmap='gray')
                else:
                    ax_res.imshow(np.array(img_display_result))
                st.pyplot(fig_res) # Hapus use_column_width
                get_image_download_button(img_filtered, filename_for_download, filter_type)
            else:
                st.warning("Gagal memproses gambar untuk ditampilkan.")

    # --- Tampilan 2: Restorasi ---
    elif feature_tab == "🛠️ Restorasi":
        st.header("🛠️ Restorasi Citra")
        
        restore_mode = st.radio("Pilih Mode Restorasi:", ["Reduksi Noise", "(Unik) Inpainting Interaktif"], key="restore_mode_radio")
        
        if restore_mode == "Reduksi Noise":
            st.subheader("Reduksi Noise")
            restore_type = st.radio("Pilih Metode:", ("Tidak ada", "Median Blur", "Bilateral Filter"), key="restore_radio")
            
            img_restored = image_cv_bgr.copy()
            
            if restore_type == "Median Blur":
                ksize_median = st.slider("Kekuatan Median", 3, 21, 5, step=2, key="median_ksize_restore")
                img_restored = apply_median_blur(image_cv_bgr, ksize_median)
            elif restore_type == "Bilateral Filter":
                st.info("Bilateral Filter mengurangi noise sambil menjaga tepi tetap tajam.")
                d_bilateral = st.slider("Diameter (d)", 1, 15, 9, key="bilateral_d_restore")
                sc_bilateral = st.slider("Sigma Color", 1, 150, 75, key="bilateral_sc_restore")
                ss_bilateral = st.slider("Sigma Space", 1, 150, 75, key="bilateral_ss_restore")
                img_restored = apply_bilateral_filter(image_cv_bgr, d_bilateral, sc_bilateral, ss_bilateral)
                
            col1_r, col2_r = st.columns(2)
            with col1_r:
                fig_orig, ax_orig = plt.subplots()
                ax_orig.imshow(image_pil_orig) 
                ax_orig.set_title("Original")
                ax_orig.axis('off')
                st.pyplot(fig_orig) # Hapus use_column_width
                
            with col2_r:
                fig_res, ax_res = plt.subplots()
                ax_res.set_title(f"Hasil: {restore_type}")
                ax_res.axis('off')
                img_display_result = cv2_to_pil(img_restored)
                if img_display_result:
                    if len(img_restored.shape) == 2:
                        ax_res.imshow(np.array(img_display_result), cmap='gray')
                    else:
                        ax_res.imshow(np.array(img_display_result))
                    st.pyplot(fig_res) # Hapus use_column_width
                    get_image_download_button(img_restored, filename_for_download, restore_type)
                else:
                    st.warning("Gagal memproses gambar untuk ditampilkan.")

        elif restore_mode == "(Unik) Inpainting Interaktif":
            st.subheader("Inpainting Interaktif (Hapus Area)")
            st.info("Gunakan tools di bawah untuk menggambar masker (coretan) pada area yang ingin Anda hilangkan/perbaiki.")
            
            col1_i, col2_i = st.columns(2)
            
            with col1_i:
                st.markdown("**Kanvas Masking** (Gambar di sini)")
                stroke_width_inp = st.slider("Ukuran Kuas", 1, 50, 15, key="stroke_inp")
                bg_pil = cv2_to_pil(image_cv_bgr) 
                
                aspect_ratio = bg_pil.height / bg_pil.width
                CANVAS_WIDTH = 600
                CANVAS_HEIGHT = min(int(CANVAS_WIDTH * aspect_ratio), 600) 
                
                if bg_pil:
                    bg_pil_resized = bg_pil.resize((CANVAS_WIDTH, CANVAS_HEIGHT))
                else:
                    bg_pil_resized = None

                canvas_result_inpainting = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=stroke_width_inp,
                    stroke_color="#FF0000", 
                    background_image=bg_pil_resized, 
                    update_streamlit=True,
                    height=CANVAS_HEIGHT,
                    width=CANVAS_WIDTH,
                    drawing_mode="freedraw",
                    key="canvas_inpainting",
                )

            with col2_i:
                st.markdown("**Hasil Inpainting**")
                radius_inp = st.slider("Radius Inpainting", 1, 15, 3, key="inp_radius")
                method_str_inp = st.radio("Metode:", ("TELEA", "NS"), key="inp_method")
                method_flag_inp = cv2.INPAINT_TELEA if method_str_inp == "TELEA" else cv2.INPAINT_NS
                
                img_inpainted = None
                mask_data_canvas = None
                
                if canvas_result_inpainting.image_data is not None:
                    mask_data_canvas = canvas_result_inpainting.image_data[:, :, 0] 
                
                if mask_data_canvas is not None and np.sum(mask_data_canvas > 0) > 0:
                    with st.spinner("Menerapkan Inpainting..."):
                         mask_resized_to_orig = cv2.resize(mask_data_canvas, (image_cv_bgr.shape[1], image_cv_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                         img_inpainted = apply_inpainting(image_cv_bgr, mask_resized_to_orig, radius_inp, method_flag_inp)
                    
                    st.image(cv2_to_pil(img_inpainted), caption="Hasil Inpainting", use_column_width=True)
                    get_image_download_button(img_inpainted, filename_for_download, "Inpainting")
                else:
                    st.image(image_pil_orig, caption="Gambar Asli (Belum ada masker)", use_column_width=True)

    # --- Tampilan 3: Enhancement ---
    elif feature_tab == "✨ Enhancement":
        st.header("✨ Enhancement Citra")
        st.subheader("Pengaturan Enhancement")
        enhance_type = st.radio("Pilih Metode:", ("Tidak ada", "Brightness / Contrast", "CLAHE", "Unsharp Masking"), key="enhance_radio")

        img_enhanced = image_cv_bgr.copy()

        if enhance_type == "Brightness / Contrast":
            b_enhance = st.slider("Brightness", -100, 100, 0, key="bc_brightness_enhance")
            c_enhance = st.slider("Contrast", -100, 100, 0, key="bc_contrast_enhance")
            img_enhanced = apply_brightness_contrast(image_cv_bgr, b_enhance, c_enhance)
        elif enhance_type == "CLAHE":
            st.info("CLAHE meningkatkan kontras lokal tanpa memperkuat noise.")
            clip_enhance = st.slider("Clip Limit", 1.0, 10.0, 2.5, 0.5, key="clahe_clip_enhance")
            grid_enhance = st.slider("Tile Grid Size", 2, 16, 8, key="clahe_grid_enhance")
            img_enhanced = apply_clahe(image_cv_bgr, clip_enhance, grid_enhance)
        elif enhance_type == "Unsharp Masking":
            st.info("Unsharp Masking menajamkan gambar berdasarkan detail yang di-blur.")
            sigma_unsharp = st.slider("Sigma (Radius Blur)", 0.1, 5.0, 1.0, 0.1, key="unsharp_sigma_enhance")
            strength_unsharp = st.slider("Strength (Kekuatan)", 0.1, 3.0, 1.5, 0.1, key="unsharp_strength_enhance")
            img_enhanced = apply_unsharp_mask(image_cv_bgr, sigma_unsharp, strength_unsharp)

        col1_e, col2_e = st.columns(2)
        with col1_e:
            fig_orig, ax_orig = plt.subplots()
            ax_orig.imshow(image_pil_orig) 
            ax_orig.set_title("Original")
            ax_orig.axis('off')
            st.pyplot(fig_orig) # Hapus use_column_width
            
        with col2_e:
            fig_res, ax_res = plt.subplots()
            ax_res.set_title(f"Hasil: {enhance_type}")
            ax_res.axis('off')
            img_display_result = cv2_to_pil(img_enhanced)
            if img_display_result:
                if len(img_enhanced.shape) == 2:
                    ax_res.imshow(np.array(img_display_result), cmap='gray')
                else:
                    ax_res.imshow(np.array(img_display_result))
                st.pyplot(fig_res) # Hapus use_column_width
                get_image_download_button(img_enhanced, filename_for_download, enhance_type)
            else:
                st.warning("Gagal memproses gambar untuk ditampilkan.")


    # --- Tampilan 4: Analisis (Fitur Unik) ---
    elif feature_tab == "🎨 Analisis":
        st.header("🎨 Analisis Citra")
        st.info("Fitur ini menganalisis gambar asli Anda tanpa mengubahnya.")
        
        # --- PERBAIKAN: Hapus kolom luar (col1_a, col2_a) ---
        
        st.subheader("Analisis Palet Warna")
        k_colors_analyze = st.slider("Jumlah Warna (K)", 2, 10, 5, key="k_colors_analyze")
        
        with st.spinner("Menganalisis palet..."):
            dom_colors_res, counts_res = analyze_color_palette(image_cv_bgr, k_colors_analyze)
        
        if dom_colors_res:
            # Ini sekarang adalah satu-satunya 'st.columns' dan tidak nested
            cols_color_res = st.columns(len(dom_colors_res)) 
            total_pixels_res = sum(counts_res) if counts_res is not None else 1
            for i, color_hex_res in enumerate(dom_colors_res):
                with cols_color_res[i]:
                    st.markdown(
                        f'<div style="background-color:{color_hex_res}; width:100%; height:50px; border: 1px solid grey; margin:auto;"></div>',
                        unsafe_allow_html=True
                    )
                    st.code(color_hex_res)
                    if counts_res is not None and i < len(counts_res):
                        percentage = (counts_res[i] / total_pixels_res) * 100
                        st.caption(f"{percentage:.1f}%")
        else:
            st.warning("Gagal menganalisis palet.")

        st.markdown("---") # Pemisah visual

        st.subheader("Analisis Histogram")
        hist_channel_select = st.selectbox("Pilih Channel:", ('Grayscale', 'RGB', 'HSV'), key="hist_channel_analyze")
        
        with st.spinner("Menghitung histogram..."):
            hist_data_res = get_histogram(image_cv_bgr)
        
        if hist_data_res:
            fig_hist, ax_hist = plt.subplots()
            plt.style.use('dark_background') 
            ax_hist.set_xlabel("Bins")
            ax_hist.set_ylabel("# Piksel")

            if hist_channel_select == 'Grayscale':
                ax_hist.set_title("Histogram Grayscale")
                ax_hist.plot(hist_data_res['Grayscale'], color='gray')
                ax_hist.set_xlim([0, 256])
            elif hist_channel_select == 'RGB':
                ax_hist.set_title("Histogram RGB")
                colors_rgb = ('b', 'g', 'r')
                for i, col in enumerate(colors_rgb):
                    ax_hist.plot(hist_data_res['RGB'][col], color=col)
                    ax_hist.set_xlim([0, 256])
            elif hist_channel_select == 'HSV':
                ax_hist.set_title("Histogram HSV (Hue)")
                ax_hist.plot(hist_data_res['HSV']['H'], color='r')
                ax_hist.set_xlim([0, 180])
            
            # PERBAIKAN: Hapus use_column_width
            st.pyplot(fig_hist)
        else:
            st.warning("Gagal menghitung histogram.")