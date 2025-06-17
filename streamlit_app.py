import streamlit as st
from pathlib import Path
import tempfile
import ForensikVideo as fv

st.set_page_config(page_title="VIFA-Pro Video Forensics")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #0c2d70; }
    .stButton>button { background-color: #0c6dd6; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VIFA-Pro: Sistem Forensik Video")

with st.sidebar:
    st.header("Input Media")
    uploaded_video = st.file_uploader(
        "Video Bukti", type=["mp4", "avi", "mov", "mkv"]
    )
    baseline_video = st.file_uploader(
        "Video Baseline (Opsional)", type=["mp4", "avi", "mov", "mkv"]
    )
    fps = st.number_input("Frame Extraction FPS", min_value=1, value=15, step=1)
    run = st.button("Jalankan Analisis")

if run:
    if uploaded_video is None:
        st.error("Mohon unggah video bukti terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sus_path = tmpdir / uploaded_video.name
            with open(sus_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            baseline_path = None
            if baseline_video is not None:
                baseline_path = tmpdir / baseline_video.name
                with open(baseline_path, "wb") as f:
                    f.write(baseline_video.getbuffer())

            progress = st.progress(0.0)

            result = fv.run_tahap_1_pra_pemrosesan(sus_path, tmpdir, int(fps))
            progress.progress(0.2)
            baseline_result = None
            if baseline_path:
                baseline_result = fv.run_tahap_1_pra_pemrosesan(baseline_path, tmpdir, int(fps))
                if baseline_result:
                    fv.run_tahap_2_analisis_temporal(baseline_result)
            progress.progress(0.4)

            fv.run_tahap_2_analisis_temporal(result, baseline_result)
            progress.progress(0.6)
            fv.run_tahap_3_sintesis_bukti(result, tmpdir)
            progress.progress(0.8)
            fv.run_tahap_4_visualisasi_dan_penilaian(result, tmpdir)
            fv.run_tahap_5_pelaporan_dan_validasi(result, tmpdir, baseline_result)
            progress.progress(1.0)

            st.success("Analisis selesai.")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Tahap 1",
                "Tahap 2",
                "Tahap 3",
                "Tahap 4",
                "Tahap 5",
            ])

            with tab1:
                st.header("Pra-pemrosesan & Ekstraksi Fitur Dasar")
                st.write("Hash SHA-256:", result.preservation_hash)
                st.json(result.metadata)

            with tab2:
                st.header("Analisis Anomali Temporal & Komparatif")
                st.write("Total Frame:", len(result.frames))

            with tab3:
                st.header("Sintesis Bukti & Investigasi Mendalam")
                total_anom = sum(1 for f in result.frames if f.type.startswith("anomaly"))
                st.write("Total Anomali:", total_anom)

            with tab4:
                st.header("Visualisasi & Penilaian Integritas")
                if result.plots.get("temporal"):
                    st.image(str(result.plots["temporal"]), caption="Timeline Anomali")
                if result.plots.get("statistic"):

            with tab5:
                st.header("Penyusunan Laporan & Validasi Forensik")
                if result.pdf_report_path and result.pdf_report_path.exists():
                    with open(result.pdf_report_path, "rb") as f:
                        st.download_button(
                            "Unduh Laporan PDF",
                            data=f,
                            file_name=result.pdf_report_path.name,
                        )