# START OF FILE streamlit_app.py

import streamlit as st
from pathlib import Path
import tempfile
import ForensikVideo as fv
import sys
import io
import traceback
from datetime import datetime
import pandas as pd
from typing import Optional, Union

# --- KONFIGURASI HALAMAN DAN GAYA ---
st.set_page_config(
    page_title="VIFA-Pro | Dashboard Forensik Video",
    layout="wide"
)

# Kustomisasi CSS untuk tema biru-putih profesional
st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; }
    h1 { color: #0B3D91; font-weight: bold; }
    h2, h3 { color: #0056b3; }
    .stButton>button { border-radius: 8px; border: 1px solid #0c6dd6; background-color: #0c6dd6; color: white; transition: all 0.2s; }
    .stButton>button:hover { border-color: #004494; background-color: #0056b3; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔎 VIFA-Pro: Sistem Deteksi Forensik Video Profesional")

# --- KONTROL PANEL DI SIDEBAR ---
with st.sidebar:
    st.header("Panel Kontrol Analisis")
    uploaded_video = st.file_uploader(
        "Unggah Video Bukti", type=["mp4", "avi", "mov", "mkv"]
    )
    baseline_video = st.file_uploader(
        "Unggah Video Baseline (Opsional)", type=["mp4", "avi", "mov", "mkv"]
    )
    fps = st.number_input("Frame Extraction FPS", min_value=1, max_value=30, value=15, step=1)
    run = st.button("🚀 Jalankan Analisis Forensik", use_container_width=True)

# --- FUNGSI BANTUAN ---
def load_image_as_bytes(path_str: Optional[Union[str, Path]]) -> Optional[bytes]:
    """Membaca file gambar dari path dan mengembalikannya sebagai byte."""
    if path_str and Path(path_str).exists():
        try:
            with open(path_str, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None

# --- LOGIKA UTAMA SAAT TOMBOL DITEKAN ---
if run:
    if uploaded_video is None:
        st.error("⚠️ Mohon unggah video bukti terlebih dahulu di sidebar.")
    else:
        # Gunakan TemporaryDirectory yang membersihkan otomatis
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sus_path = tmpdir_path / uploaded_video.name
            with open(sus_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            baseline_path = None
            if baseline_video is not None:
                baseline_path = tmpdir_path / baseline_video.name
                with open(baseline_path, "wb") as f:
                    f.write(baseline_video.getbuffer())

            result = None
            baseline_result = None
            
            # Menjalankan pipeline dengan progress bar
            try:
                with st.spinner("Tahap 1: Pra-pemrosesan..."):
                    result = fv.run_tahap_1_pra_pemrosesan(sus_path, tmpdir_path, int(fps))
                if not result:
                    st.error("Gagal pada Tahap 1. Analisis dihentikan."); st.stop()
                
                with st.spinner("Memproses baseline & Tahap 2: Analisis Temporal..."):
                    if baseline_path:
                        baseline_result = fv.run_tahap_1_pra_pemrosesan(baseline_path, tmpdir_path, int(fps))
                        if baseline_result: fv.run_tahap_2_analisis_temporal(baseline_result)
                    fv.run_tahap_2_analisis_temporal(result, baseline_result)

                with st.spinner("Tahap 3: Sintesis Bukti & Investigasi Mendalam..."):
                    fv.run_tahap_3_sintesis_bukti(result, tmpdir_path)

                with st.spinner("Tahap 4 & 5: Visualisasi dan Pelaporan..."):
                    fv.run_tahap_4_visualisasi_dan_penilaian(result, tmpdir_path)
                    fv.run_tahap_5_pelaporan_dan_validasi(result, tmpdir_path, baseline_result)
            except Exception:
                 st.error("Terjadi error saat analisis."); st.code(traceback.format_exc()); st.stop()


            # --- LANGKAH KRUSIAL: BACA SEMUA ARTEFAK KE MEMORI (BYTES) ---
            # Ini dilakukan sebelum blok 'with' selesai dan menghapus direktori temporer
            with st.spinner("Mengemas hasil untuk ditampilkan..."):
                # Konversi plot menjadi byte
                for key in result.plots:
                    result.plots[key] = load_image_as_bytes(result.plots[key])
                # Konversi path di lokalisasi menjadi byte
                for loc in result.localizations:
                    loc['image_bytes'] = load_image_as_bytes(loc['image'])
                    loc['ela_path_bytes'] = load_image_as_bytes(loc['ela_path'])
                    loc['sift_path_bytes'] = load_image_as_bytes(loc['sift_path'])
                # Konversi path laporan PDF
                if result.pdf_report_path:
                    result.pdf_report_data = load_image_as_bytes(result.pdf_report_path)

            st.success("✅ Analisis 5 Tahap Forensik Berhasil Diselesaikan!")

            # --- TAMPILAN HASIL PROFESIONAL DENGAN TAB ---
            tab_titles = [
                "📄 **Tahap 1: Akuisisi**", "📊 **Tahap 2: Analisis Temporal**", 
                "🔬 **Tahap 3: Investigasi**", "📈 **Tahap 4: Penilaian**", "📥 **Tahap 5: Laporan**"
            ]
            tabs = st.tabs(tab_titles)

            with tabs[0]: # Tahap 1
                st.header("Hasil Tahap 1: Akuisisi & Ekstraksi Fitur Dasar")
                st.info("Tujuan: Mengamankan dan mempersiapkan barang bukti digital.", icon="🛡️")
                col1, col2 = st.columns(2);
                with col1:
                    st.subheader("Integritas Bukti (SHA-256)"); st.code(result.preservation_hash, language="bash")
                with col2:
                    st.subheader("Ekstraksi Frame"); st.metric("Total Frame Dianalisis", result.summary.get('total_frames', 'N/A'))
                st.subheader("Metadata Video"); video_stream=next((s for s in result.metadata.get('streams', []) if s.get('codec_type') == 'video'), None)
                if video_stream:
                    c1,c2,c3,c4 = st.columns(4); c1.metric("Resolusi", f"{video_stream.get('width')}x{video_stream.get('height')}"); c2.metric("Codec", video_stream.get('codec_name', 'N/A').upper()); c3.metric("Durasi", f"{float(video_stream.get('duration', 0)):.2f} s"); c4.metric("Frame Rate", f"{eval(video_stream.get('r_frame_rate', '0/1')):.2f} FPS")

            with tabs[1]: # Tahap 2
                st.header("Hasil Tahap 2: Analisis Anomali Temporal & Komparatif")
                st.write("Menganalisis hubungan antar frame berurutan untuk mendeteksi 'patahan' atau inkonsistensi yang ditinggalkan manipulasi.")
                st.subheader("Distribusi Metrik Temporal");
                if result.plots.get('statistic'):
                    st.image(result.plots['statistic'], caption="Kiri: Skor SSIM. Kanan: Aliran Optik.")
                    st.write("- **SSIM:** Sebaran rendah bisa indikasi re-encoding.\n- **Aliran Optik:** Lonjakan dapat menandakan *cut*.")
                if baseline_result:
                    st.subheader("Analisis Komparatif"); st.info(f"Ditemukan **{len([loc for loc in result.localizations if loc['event'] == 'anomaly_insertion'])} peristiwa penyisipan**.", icon="🔎")

            with tabs[2]: # Tahap 3
                st.header("Hasil Tahap 3: Investigasi Detail Anomali")
                st.write("Inti analisis: mengkorelasikan temuan dan melakukan investigasi mendalam (ELA, SIFT) pada kandidat terkuat.")
                if not result.localizations: st.success("🎉 **Tidak Ditemukan Anomali Signifikan.**")
                else:
                    st.warning(f"🚨 Ditemukan **{len(result.localizations)} peristiwa anomali**:", icon="🚨")
                    for i, loc in enumerate(result.localizations):
                        event_type = loc['event'].replace('anomaly_', '').capitalize()
                        with st.expander(f"**Peristiwa #{i+1}: {event_type}** @ {loc['start_ts']:.2f} detik (Keyakinan: {loc.get('confidence', 'N/A')})", expanded= i == 0):
                            c1,c2=st.columns([1,2])
                            with c1:
                                if loc.get('image_bytes'): st.image(loc['image_bytes'], caption=f"Frame #{loc.get('start_frame')}")
                                if loc.get('ela_path_bytes'): st.image(loc['ela_path_bytes'], caption="Analisis ELA")
                            with c2:
                                st.markdown(f"**Rentang:** Frame {loc.get('start_frame')} - {loc.get('end_frame')}")
                                st.markdown(f"**Alasan Deteksi:** `{loc.get('reasons', 'N/A')}`")
                                st.markdown("**Metrik Teknis:**"); st.json(loc.get('metrics', {}))
                            if loc.get('sift_path_bytes'): st.image(loc.get('sift_path_bytes'), caption="Bukti SIFT")

            with tabs[3]: # Tahap 4
                st.header("Hasil Tahap 4: Visualisasi & Penilaian Integritas")
                st.write("Meringkas temuan dalam bentuk visual dan skor kuantitatif untuk mempermudah pengambilan keputusan.")
                c1,c2=st.columns(2)
                with c1:
                    st.subheader("Skor Integritas"); integrity_score, integrity_desc = fv.generate_integrity_score(result.summary); st.metric(label="Skor Integritas Video", value=f"{integrity_score}/100", delta=integrity_desc, delta_color="inverse" if integrity_score < 70 else "normal")
                with c2:
                    st.subheader("Statistik"); st.metric("Frame Anomali", f"{result.summary.get('total_anomaly', 'N/A')}"); st.metric("Persentase Anomali", f"{result.summary.get('pct_anomaly', 'N/A')}%")
                st.subheader("Peta Anomali Temporal")
                if result.plots.get('temporal'):
                    st.image(result.plots['temporal'], caption="Visualisasi lokasi dan jenis anomali.", use_column_width=True)

            with tabs[4]: # Tahap 5
                st.header("Hasil Tahap 5: Penyusunan Laporan & Validasi Forensik")
                st.write("Tahap akhir menghasilkan laporan PDF yang dapat diaudit dan berfungsi sebagai validasi proses.")
                st.info("Unduh laporan PDF lengkap untuk dokumentasi atau sebagai lampiran bukti digital.", icon="📄")
                if 'pdf_report_data' in result and result.pdf_report_data:
                    st.download_button(label="📥 Unduh Laporan PDF Lengkap", data=result.pdf_report_data, file_name=result.pdf_report_path.name, mime="application/pdf", use_container_width=True)
                else: st.error("File laporan PDF tidak dapat dibuat.")
                st.subheader("Validasi Proses"); st.table(pd.DataFrame.from_dict({"File Bukti": Path(result.video_path).name, "Hash SHA-256": result.preservation_hash, "Waktu Analisis": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}, orient='index', columns=['Detail']))