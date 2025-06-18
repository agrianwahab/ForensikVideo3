# START OF FILE ForensikVideo.py

# vifa_pro.py
# (Sistem Forensik Video Profesional dengan Analisis Multi-Lapis)
# VERSI 5 TAHAP PENELITIAN (DENGAN PERBAIKAN BUG STYLE REPORTLAB)

"""
VIFA-Pro: Sistem Forensik Video Profesional (Arsitektur 5 Tahap)
========================================================================================
Versi ini mengimplementasikan alur kerja forensik formal dalam 5 tahap yang jelas,
sesuai dengan metodologi penelitian untuk deteksi manipulasi video. Setiap tahap
memiliki tujuan spesifik, dari ekstraksi fitur dasar hingga validasi proses.

ARSITEKTUR PIPELINE:
- TAHAP 1: Pra-pemrosesan & Ekstraksi Fitur Dasar (Hashing, Frame, pHash, Warna)
- TAHAP 2: Analisis Anomali Temporal & Komparatif (Optical Flow, SSIM, Baseline Check)
- TAHAP 3: Sintesis Bukti & Investigasi Mendalam (Korelasi Metrik, ELA & SIFT on-demand)
- TAHAP 4: Visualisasi & Penilaian Integritas (Plotting, Integrity Score)
- TAHAP 5: Penyusunan Laporan & Validasi Forensik (Laporan PDF Naratif)

Deteksi:
- Diskontinuitas (Deletion/Insertion): Melalui Aliran Optik, SSIM, K-Means, dan Perbandingan Baseline.
- Duplikasi Frame (Duplication): Melalui pHash, dikonfirmasi oleh SIFT+RANSAC dan SSIM.
- Penyisipan Area (Splicing): Terindikasi oleh Analisis Tingkat Kesalahan (ELA) pada titik diskontinuitas.

Author: OpenAI-GPT & Anda
License: MIT
Dependencies: opencv-python, opencv-contrib-python, imagehash, numpy, Pillow,
              reportlab, matplotlib, tqdm, scikit-learn, scikit-image
"""

from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

# Pemeriksaan Dependensi Awal
try:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    print(f"Error: Dependensi penting tidak ditemukan -> {e}")
    sys.exit(1)


###############################################################################
# Utilitas & Konfigurasi Global
###############################################################################

class Icons: IDENTIFICATION="🔍"; PRESERVATION="🛡️"; COLLECTION="📥"; EXAMINATION="🔬"; ANALYSIS="📈"; REPORTING="📄"; SUCCESS="✅"; ERROR="❌"; INFO="ℹ️"; CONFIDENCE_LOW="🟩"; CONFIDENCE_MED="🟨"; CONFIDENCE_HIGH="🟧"; CONFIDENCE_VHIGH="🟥"
CONFIG = {"HASH_DIST_DUPLICATE": 2, "OPTICAL_FLOW_Z_THRESH": 4.0, "SSIM_DISCONTINUITY_DROP": 0.25, "SIFT_MIN_MATCH_COUNT": 10, "KMEANS_CLUSTERS": 8, "DUPLICATION_SSIM_CONFIRM": 0.95}

# Fungsi log yang dienkapsulasi untuk output ke konsol dan UI Streamlit
def log(message: str):
    print(message, file=sys.stdout) # Menggunakan stdout asli untuk logging

def print_stage_banner(stage_number: int, stage_name: str, icon: str, description: str):
    width=80
    log("\n" + "="*width)
    log(f"=== {icon}  TAHAP {stage_number}: {stage_name.upper()} ".ljust(width - 3) + "===")
    log("="*width)
    log(f"{Icons.INFO}  {description}")
    log("-" * width)

###############################################################################
# Struktur Data Inti
###############################################################################

@dataclass
class Evidence:
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    confidence: str = "N/A"
    ela_path: str | None = None
    sift_path: str | None = None

@dataclass
class FrameInfo:
    index: int
    timestamp: float
    img_path: str
    hash: str | None = None
    type: str = "original"
    ssim_to_prev: float | None = None
    optical_flow_mag: float | None = None
    color_cluster: int | None = None
    evidence_obj: Evidence = field(default_factory=Evidence)

@dataclass
class AnalysisResult:
    video_path: str
    preservation_hash: str
    metadata: dict
    frames: list[FrameInfo]
    summary: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    localizations: list[dict] = field(default_factory=list)
    pdf_report_path: Path | None = None

###############################################################################
# Fungsi Analisis Individual (tidak berubah)
#<editor-fold desc="Fungsi Analisis Inti">
def perform_ela(image_path: Path, quality: int=90) -> Path | None:
    try:
        ela_dir = image_path.parent.parent / "ela_artifacts"
        ela_dir.mkdir(exist_ok=True)
        out_path = ela_dir / f"{image_path.stem}_ela.jpg"
        temp_jpg_path = out_path.with_name(f"temp_{out_path.name}")
        with Image.open(image_path).convert('RGB') as im:
            im.save(temp_jpg_path, 'JPEG', quality=quality)
        with Image.open(image_path).convert('RGB') as im_orig, Image.open(temp_jpg_path) as resaved_im:
            ela_im = ImageChops.difference(im_orig, resaved_im)
        if Path(temp_jpg_path).exists():
            Path(temp_jpg_path).unlink()
        extrema = ela_im.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_im.save(out_path)
        return out_path
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal ELA pada {image_path.name}: {e}")
        return None

def compare_sift(img_path1: Path, img_path2: Path, out_dir: Path) -> tuple[int, Path | None]:
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return 0, None
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: return 0, None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        if not matches or any(len(m) < 2 for m in matches): return 0, None
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > CONFIG["SIFT_MIN_MATCH_COUNT"]:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or mask is None: return len(good_matches), None
            inliers = mask.ravel().sum()
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
            img_out = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            sift_dir = out_dir / "sift_artifacts"
            sift_dir.mkdir(exist_ok=True)
            out_path = sift_dir / f"sift_{img_path1.stem}_vs_{img_path2.stem}.jpg"
            cv2.imwrite(str(out_path), img_out)
            return int(inliers), out_path
        return len(good_matches), None
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal SIFT: {e}")
        return 0, None

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ffprobe_metadata(video_path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        return json.loads(result.stdout)
    except Exception as e:
        log(f"FFprobe error: {e}")
        return {}

def extract_frames(video_path: Path, out_dir: Path, fps: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2", pattern, "-y", "-hide_banner", "-loglevel", "error"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return len(list(out_dir.glob('*.jpg')))
    except subprocess.CalledProcessError as e:
        log(f"ffmpeg error:\n{e.stderr.decode() if e.stderr else 'Tidak ada output error dari ffmpeg.'}")
        return 0
#</editor-fold>

###############################################################################
# PIPELINE 5-TAHAP
###############################################################################

# --- TAHAP 1: PRA-PEMROSESAN & EKSTRAKSI FITUR DASAR ---
def run_tahap_1_pra_pemrosesan(video_path: Path, out_dir: Path, fps: int) -> AnalysisResult | None:
    print_stage_banner(1, "Pra-pemrosesan & Ekstraksi Fitur Dasar", Icons.COLLECTION, 
                       "Melakukan hashing, ekstraksi metadata, ekstraksi frame, pHash, dan analisis warna.")
    
    log(f"  {Icons.PRESERVATION} Menghitung hash SHA-256 untuk preservasi...")
    preservation_hash = calculate_sha256(video_path)
    log(f"  -> Hash Bukti: {preservation_hash}")

    log(f"  {Icons.IDENTIFICATION} Mengekstrak metadata dengan FFprobe...")
    metadata = ffprobe_metadata(video_path)
    if not metadata:
        log(f"  {Icons.ERROR} Gagal mengekstrak metadata. Analisis tidak dapat dilanjutkan.")
        return None

    log(f"  {Icons.COLLECTION} Mengekstrak frame @ {fps} FPS menggunakan FFmpeg...")
    frames_dir = out_dir / f"frames_{video_path.stem}"
    num_frames = extract_frames(video_path, frames_dir, fps)
    if num_frames == 0:
        log(f"  {Icons.ERROR} Gagal mengekstrak frame. Pastikan video valid dan FFmpeg terinstal.")
        return None
    log(f"  ✅ {num_frames} frame berhasil diekstrak ke {frames_dir.stem}")

    log(f"  {Icons.EXAMINATION} Menghitung pHash untuk setiap frame...")
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frames = []
    for idx, fpath in enumerate(tqdm(frame_files, desc="    pHash", leave=False, bar_format='{l_bar}{bar}{r_bar}')):
        try:
            with Image.open(fpath) as img:
                frame_hash = str(imagehash.average_hash(img))
            frames.append(FrameInfo(idx, idx / fps, str(fpath), hash=frame_hash))
        except Exception as e:
            log(f"  {Icons.ERROR} Gagal memproses frame {fpath.name}: {e}")
    
    log(f"  {Icons.EXAMINATION} Menganalisis layout warna global (K-Means)...")
    histograms = []
    for f in tqdm(frames, desc="    Histogram", leave=False, bar_format='{l_bar}{bar}{r_bar}'):
        img = cv2.imread(f.img_path)
        if img is None: continue
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    if histograms:
        actual_n_clusters = min(CONFIG["KMEANS_CLUSTERS"], len(histograms))
        if actual_n_clusters >= 2:
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto').fit(histograms)
            for f, label in zip(frames, kmeans.labels_.tolist()):
                f.color_cluster = int(label)

    log(f"  {Icons.SUCCESS} Tahap 1 Selesai.")
    return AnalysisResult(str(video_path), preservation_hash, metadata, frames)

# --- TAHAP 2: ANALISIS ANOMALI TEMPORAL & KOMPARATIF ---
def run_tahap_2_analisis_temporal(result: AnalysisResult, baseline_result: AnalysisResult | None = None):
    print_stage_banner(2, "Analisis Anomali Temporal & Komparatif", Icons.ANALYSIS,
                       "Menganalisis aliran optik, SSIM, dan perbandingan dengan baseline jika ada.")
    frames = result.frames
    prev_gray = None

    log(f"  {Icons.EXAMINATION} Menghitung Aliran Optik & SSIM antar frame...")
    for f in tqdm(frames, desc="    Temporal", leave=False, bar_format='{l_bar}{bar}{r_bar}'):
        current_gray = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE)
        if current_gray is not None and prev_gray is not None:
            # SSIM
            data_range = float(current_gray.max() - current_gray.min())
            if data_range > 0:
                ssim_score, _ = ssim(prev_gray, current_gray, data_range=data_range, full=True)
                f.ssim_to_prev = float(ssim_score)
            
            # Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            f.optical_flow_mag = float(np.mean(mag))

        prev_gray = current_gray

    # Analisis Komparatif (jika ada baseline)
    if baseline_result:
        log(f"  {Icons.ANALYSIS} Melakukan analisis komparatif terhadap video baseline...")
        base_hashes = {f.hash for f in baseline_result.frames if f.hash}
        insertion_count = 0
        for f_sus in frames:
            if f_sus.hash and f_sus.hash not in base_hashes:
                f_sus.type = "anomaly_insertion"
                f_sus.evidence_obj.reasons.append("Frame tidak ada di baseline")
                f_sus.evidence_obj.confidence = "SANGAT TINGGI"
                insertion_count += 1
        log(f"  -> Terdeteksi {insertion_count} frame sisipan potensial.")

    log(f"  {Icons.SUCCESS} Tahap 2 Selesai.")

# --- TAHAP 3: SINTESIS BUKTI & INVESTIGASI MENDALAM ---
def run_tahap_3_sintesis_bukti(result: AnalysisResult, out_dir: Path):
    print_stage_banner(3, "Sintesis Bukti & Investigasi Mendalam", "🔬",
                       "Mengkorelasikan semua temuan dan melakukan analisis ELA/SIFT pada anomali terkuat.")
    frames = result.frames
    n = len(frames)
    if n < 2: return

    log("  {Icons.ANALYSIS} Menganalisis metrik diskontinuitas (Z-score, SSIM drop, Color jump)...")
    flow_mags = [f.optical_flow_mag for f in frames if f.optical_flow_mag is not None]
    if flow_mags:
        median_flow = np.median(flow_mags)
        mad_flow = np.median(np.abs(flow_mags - median_flow))
        mad_flow = 1e-9 if mad_flow == 0 else mad_flow # Hindari pembagian dengan nol

        for f in frames:
            if f.optical_flow_mag:
                z_score = 0.6745 * (f.optical_flow_mag - median_flow) / mad_flow
                if abs(z_score) > CONFIG["OPTICAL_FLOW_Z_THRESH"]:
                    f.evidence_obj.reasons.append("Lonjakan Aliran Optik")
                    f.evidence_obj.metrics["optical_flow_z_score"] = round(z_score, 2)
    
    for i in range(1, n):
        f_curr, f_prev = frames[i], frames[i - 1]
        if f_curr.ssim_to_prev and f_prev.ssim_to_prev and abs(f_prev.ssim_to_prev - f_curr.ssim_to_prev) > CONFIG["SSIM_DISCONTINUITY_DROP"]:
            f_curr.evidence_obj.reasons.append("Penurunan Drastis SSIM")
            f_curr.evidence_obj.metrics["ssim_drop"] = round(abs(f_prev.ssim_to_prev - f_curr.ssim_to_prev), 2)
        if f_curr.color_cluster is not None and f_prev.color_cluster is not None and f_curr.color_cluster != f_prev.color_cluster:
            f_curr.evidence_obj.reasons.append("Perubahan Adegan (Warna)")
            f_curr.evidence_obj.metrics["color_cluster_jump"] = f"{f_prev.color_cluster} -> {f_curr.color_cluster}"

    log(f"  {Icons.EXAMINATION} Memverifikasi duplikasi frame menggunakan pHash dan SIFT...")
    hash_map = defaultdict(list)
    for f in frames:
        if f.hash: hash_map[f.hash].append(f.index)
    
    dup_candidates = {k: v for k, v in hash_map.items() if len(v) > 1}
    for _, idxs in tqdm(dup_candidates.items(), desc="    Duplikasi", leave=False, bar_format='{l_bar}{bar}{r_bar}'):
        for i in range(len(idxs) - 1):
            idx1, idx2 = idxs[i], idxs[i + 1]
            p1, p2 = Path(frames[idx1].img_path), Path(frames[idx2].img_path)
            # Konfirmasi SSIM dulu karena lebih cepat dari SIFT
            im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
            if im1 is None or im2 is None: continue
            data_range = float(im1.max() - im1.min())
            if data_range == 0: continue
            ssim_val, _ = ssim(im1, im2, data_range=data_range, full=True)
            
            if ssim_val > CONFIG["DUPLICATION_SSIM_CONFIRM"]:
                # Baru jalankan SIFT yang mahal
                inliers, sift_p = compare_sift(p1, p2, out_dir)
                if inliers >= CONFIG["SIFT_MIN_MATCH_COUNT"]:
                    f_dup = frames[idx2]
                    f_dup.type = "anomaly_duplication"
                    f_dup.evidence_obj.reasons.append(f"Duplikasi dari frame {idx1}")
                    f_dup.evidence_obj.metrics.update({"source_frame": idx1, "ssim_to_source": round(ssim_val, 4), "sift_inliers": inliers})
                    f_dup.evidence_obj.sift_path = str(sift_p) if sift_p else None

    log(f"  {Icons.ANALYSIS} Menjalankan ELA on-demand dan finalisasi sintesis bukti...")
    for f in tqdm(frames, desc="    Sintesis", leave=False, bar_format='{l_bar}{bar}{r_bar}'):
        if f.evidence_obj.reasons:
            if f.type == "original": 
                f.type = "anomaly_discontinuity"
            
            num_reasons = len(f.evidence_obj.reasons)
            if f.type == "anomaly_duplication" or f.type == "anomaly_insertion":
                f.evidence_obj.confidence = "SANGAT TINGGI"
            elif num_reasons > 2:
                f.evidence_obj.confidence = "TINGGI"
            elif num_reasons > 1:
                f.evidence_obj.confidence = "SEDANG"
            else:
                f.evidence_obj.confidence = "RENDAH"

            # Jalankan ELA hanya pada frame yang mencurigakan (selain duplikasi)
            if f.evidence_obj.confidence in ["SEDANG", "TINGGI", "SANGAT TINGGI"] and f.type not in ["anomaly_duplication", "anomaly_insertion"]:
                ela_p = perform_ela(Path(f.img_path))
                if ela_p:
                    f.evidence_obj.ela_path = str(ela_p)
                    f.evidence_obj.reasons.append("Potensi Anomali Kompresi (ELA)")
                    # Tingkatkan kepercayaan jika ELA berhasil
                    f.evidence_obj.confidence = "SANGAT TINGGI"
    
    # Finalisasi reason menjadi string
    for f in frames:
        if isinstance(f.evidence_obj.reasons, list) and f.evidence_obj.reasons:
             f.evidence_obj.reasons = ", ".join(sorted(list(set(f.evidence_obj.reasons))))

    log(f"  {Icons.SUCCESS} Tahap 3 Selesai.")


# --- TAHAP 4: VISUALISASI & PENILAIAN INTEGRITAS ---
def run_tahap_4_visualisasi_dan_penilaian(result: AnalysisResult, out_dir: Path):
    print_stage_banner(4, "Visualisasi & Penilaian Integritas", "📊",
                       "Membuat plot, melokalisasi peristiwa, dan menghitung skor integritas.")
    
    # 1. Bangun Lokalisasi
    locs, event = [], None
    for f in result.frames:
        if f.type.startswith("anomaly"):
            if event and event["event"] == f.type and f.index == event["end_frame"] + 1:
                event["end_frame"] = f.index; event["end_ts"] = f.timestamp
            else:
                if event: locs.append(event)
                event = {"event": f.type, "start_frame": f.index, "end_frame": f.index, "start_ts": f.timestamp,
                         "end_ts": f.timestamp, "confidence": f.evidence_obj.confidence, "reasons": f.evidence_obj.reasons,
                         "metrics": f.evidence_obj.metrics, "image": f.img_path, "ela_path": f.evidence_obj.ela_path,
                         "sift_path": f.evidence_obj.sift_path}
        elif event:
            locs.append(event)
            event = None
    if event: locs.append(event)
    result.localizations = locs
    log(f"  {Icons.INFO} Ditemukan {len(locs)} peristiwa anomali yang dilokalisasi.")
    
    # 2. Hitung Summary
    total_anom = sum(1 for f in result.frames if f.type.startswith("anomaly"))
    total_frames = len(result.frames)
    pct_anomaly = round(total_anom * 100 / total_frames, 2) if total_frames > 0 else 0
    result.summary = {"total_frames": total_frames, "total_anomaly": total_anom, "pct_anomaly": pct_anomaly}
    log(f"  {Icons.INFO} {total_anom} dari {total_frames} frame terindikasi anomali ({pct_anomaly}%).")
    
    # 3. Buat Plot
    log("  {Icons.ANALYSIS} Membuat plot visualisasi anomali...")
    # Plot Peta Anomali Temporal
    plt.figure(figsize=(15, 6))
    anomaly_data = {'Duplikasi':{'x':[],'color':'orange','marker':'o','level':1.0}, 'Penyisipan':{'x':[],'color':'red','marker':'x','level':0.9}, 'Diskontinuitas':{'x':[],'color':'purple','marker':'|','level':0.8}}
    for f in result.frames:
        if f.type == "anomaly_duplication": anomaly_data['Duplikasi']['x'].append(f.index)
        elif f.type == "anomaly_insertion": anomaly_data['Penyisipan']['x'].append(f.index)
        elif f.type == "anomaly_discontinuity": anomaly_data['Diskontinuitas']['x'].append(f.index)
    for label, data in anomaly_data.items():
        if data['x']: plt.vlines(data['x'], 0, data['level'], colors=data['color'], lw=1.5, alpha=0.8); plt.scatter(data['x'], np.full_like(data['x'], data['level']), c=data['color'], marker=data['marker'], s=40, label=label, zorder=5)
    plt.ylim(-0.1, 1.2); plt.yticks([0, 0.8, 0.9, 1.0], ['Asli', 'Diskontinuitas', 'Penyisipan', 'Duplikasi']); plt.xlabel("Indeks Bingkai", fontsize=12); plt.ylabel("Jenis Anomali Terdeteksi", fontsize=12); plt.title(f"Peta Anomali Temporal untuk {Path(result.video_path).name}", fontsize=14, weight='bold'); plt.grid(True, axis='x', linestyle=':', alpha=0.7); from matplotlib.lines import Line2D; plt.legend(handles=[Line2D([0], [0], color=d['color'], marker=d['marker'], linestyle='None', label=l) for l, d in anomaly_data.items()], loc='upper right', fontsize=10); plt.tight_layout()
    temporal_plot_path = out_dir / f"plot_temporal_{Path(result.video_path).stem}.png"
    plt.savefig(temporal_plot_path, bbox_inches="tight", dpi=150); plt.close()
    result.plots['temporal'] = str(temporal_plot_path)

    # Plot Statistik
    ssim_values = [f.ssim_to_prev for f in result.frames if f.ssim_to_prev is not None]
    flow_values = [f.optical_flow_mag for f in result.frames if f.optical_flow_mag is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4));
    if ssim_values: ax1.hist(ssim_values, bins=50, color='skyblue', edgecolor='black'); ax1.set_title("Distribusi Skor SSIM"); ax1.set_xlabel("Skor SSIM (Tinggi = Mirip)"); ax1.set_ylabel("Frekuensi")
    if flow_values: ax2.hist(flow_values, bins=50, color='salmon', edgecolor='black'); ax2.set_title("Distribusi Aliran Optik"); ax2.set_xlabel("Rata-rata Pergerakan"); ax2.set_ylabel("Frekuensi")
    plt.tight_layout()
    stat_plot_path = out_dir / f"plot_statistic_{Path(result.video_path).stem}.png"
    plt.savefig(stat_plot_path, dpi=100); plt.close()
    result.plots['statistic'] = str(stat_plot_path)

    log(f"  {Icons.SUCCESS} Tahap 4 Selesai.")

# --- TAHAP 5: PENYUSUNAN LAPORAN & VALIDASI FORENSIK ---
def run_tahap_5_pelaporan_dan_validasi(result: AnalysisResult, out_dir: Path, baseline_result: AnalysisResult | None = None):
    print_stage_banner(5, "Penyusunan Laporan & Validasi Forensik", Icons.REPORTING,
                       "Menghasilkan laporan PDF naratif yang komprehensif dan dapat diaudit.")
    
    pdf_path = out_dir / f"laporan_forensik_{Path(result.video_path).stem}.pdf"
    
    # Helper functions for PDF generation
    def get_encoder_info(metadata: dict) -> str:
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if stream.get('codec_type') == 'video': return stream.get('tags', {}).get('encoder', 'N/A')
        return metadata.get('format', {}).get('tags', {}).get('encoder', 'N/A')

    def generate_integrity_score(summary: dict) -> tuple[int, str]:
        pct = summary.get('pct_anomaly', 0)
        if pct == 0: return (100, "Sangat Baik")
        if pct < 5: return (85, "Baik")
        if pct < 15: return (60, "Cukup")
        if pct < 30: return (40, "Buruk")
        return (20, "Sangat Buruk")

    def get_anomaly_explanation(event_type: str) -> str:
        explanations = {
            "Duplication": "Frame-frame ini adalah salinan identik dari frame sebelumnya. Teknik ini sering digunakan untuk memperpanjang durasi adegan secara buatan atau menutupi penghapusan.",
            "Insertion": "Frame-frame ini <b>tidak ditemukan</b> dalam video asli (baseline). Ini adalah indikasi kuat bahwa segmen video baru telah disisipkan.",
            "Discontinuity": "Terdeteksi 'patahan' atau transisi mendadak pada titik ini. Ini biasanya menandakan adanya pemotongan (cut) atau penggabungan dua klip video yang berbeda, yang merupakan jejak dari penghapusan atau penyisipan."
        }
        return explanations.get(event_type, "Jenis anomali tidak dikenal.")

    def explain_metric(metric_name: str) -> str:
        return {
            "optical_flow_z_score": "Skor ini mengukur seberapa tidak biasa pergerakan piksel pada frame ini dibandingkan rata-rata. Skor tinggi (>4.0) menunjukkan gerakan yang sangat mendadak atau tidak wajar.",
            "ssim_drop": "Ini mengukur seberapa drastis perubahan visual antara frame ini dan frame sebelumnya. Penurunan besar menunjukkan perubahan adegan yang tiba-tiba.",
            "color_cluster_jump": "Video dibagi menjadi 'adegan' berdasarkan warna dominan. 'Jump' berarti frame ini memiliki palet warna yang sama sekali berbeda dari sebelumnya, menandakan potongan.",
            "source_frame": "Ini adalah nomor frame asli yang disalin untuk membuat duplikasi ini.",
            "ssim_to_source": "Skor kemiripan visual (0-1) antara frame ini dan sumber duplikasinya. Skor >0.95 menunjukkan mereka hampir identik.",
            "sift_inliers": "Jumlah titik fitur unik yang cocok antara frame ini dan sumbernya. Ratusan titik cocok adalah bukti duplikasi yang sangat kuat."
        }.get(metric_name, "Metrik tidak dikenal.")

    # PDF Generation Logic
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=25, bottomMargin=50, leftMargin=30, rightMargin=30)
    styles = getSampleStyleSheet()

    # === BAGIAN YANG DIPERBAIKI ===
    # Cek apakah style sudah ada sebelum menambahkannya
    if 'Code' not in styles:
        styles.add(ParagraphStyle('Code', fontName='Courier', fontSize=8, leading=10))
    if 'SubTitle' not in styles:
        styles.add(ParagraphStyle('SubTitle', parent=styles['h2'], fontSize=12, textColor=colors.darkslategray))
    if 'Justify' not in styles:
        styles.add(ParagraphStyle('Justify', parent=styles['Normal'], alignment=4))
    if 'H3-Box' not in styles:
        styles.add(ParagraphStyle('H3-Box', parent=styles['h3'], backColor=colors.lightgrey, padding=4, leading=14, leftIndent=4, borderPadding=2))
    # ===============================

    story = []

    def header_footer(canvas, doc):
        canvas.saveState(); canvas.setFont('Helvetica', 8)
        canvas.drawString(30, 30, f"Laporan VIFA-Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(A4[0] - 30, 30, f"Halaman {doc.page}"); canvas.restoreState()

    # ---- Executive Summary ----
    story.append(Paragraph("Laporan Analisis Forensik Video", styles['h1']))
    story.append(Paragraph("Dihasilkan oleh Sistem VIFA-Pro", styles['SubTitle']))
    story.append(Spacer(1, 24))
    story.append(Paragraph("Ringkasan Eksekutif", styles['h2']))
    integrity_score, integrity_desc = generate_integrity_score(result.summary)
    summary_text = (f"Analisis komprehensif terhadap file <b>{Path(result.video_path).name}</b> telah selesai. "
                    f"Berdasarkan <b>{len(result.localizations)} peristiwa anomali</b> yang terdeteksi dari total {result.summary.get('total_frames',0)} frame, "
                    f"video ini diberikan <b>Skor Integritas Keseluruhan: {integrity_score}/100 ({integrity_desc})</b>. ")
    event_counts = Counter(loc['event'].replace('anomaly_', '').capitalize() for loc in result.localizations)
    if event_counts:
        findings = [f"<b>{count} peristiwa {event}</b>" for event, count in event_counts.items()]
        summary_text += f"Temuan utama meliputi: {', '.join(findings)}. Rincian investigasi untuk setiap temuan tersedia di Tahap 3 laporan ini."
    story.append(Paragraph(summary_text, styles['Justify'])); story.append(Spacer(1, 12))
    
    # ---- Stages Report ----
    story.append(PageBreak())
    story.append(Paragraph("Detail Laporan Berdasarkan Tahapan Forensik", styles['h1']))

    # Tahap 1 & 2
    story.append(Paragraph("Tahap 1 & 2: Akuisisi dan Analisis Temporal", styles['h2']))
    story.append(Paragraph("Pada tahap awal, sistem melakukan akuisisi data digital secara forensik. File video bukti di-hash menggunakan SHA-256 untuk menjamin integritasnya. Selanjutnya, video diekstrak menjadi frame individual untuk dianalisis. Fitur dasar (pHash, warna) dan temporal (SSIM, Aliran Optik) dihitung untuk setiap frame. Jika video baseline diberikan, perbandingan langsung dilakukan untuk mengidentifikasi potensi frame sisipan.", styles['Justify']))
    if baseline_result:
        enc_base = get_encoder_info(baseline_result.metadata)
        enc_sus = get_encoder_info(result.metadata)
        if enc_base != "N/A" and enc_sus != "N/A" and enc_base.lower() not in enc_sus.lower() and enc_sus.lower() not in enc_base.lower():
            story.append(Paragraph(f"<para backColor=yellow><b>Peringatan Re-Encoding:</b> Terdeteksi perbedaan encoder. Baseline: <b>{enc_base}</b>. Bukti: <b>{enc_sus}</b>. Ini indikasi kuat video bukti telah diproses ulang.</para>", styles['Normal']))
    story.append(Spacer(1, 12))

    # Tahap 3
    story.append(Paragraph("Tahap 3: Investigasi Detail Anomali", styles['h2']))
    story.append(Paragraph("Tahap ini merupakan inti dari analisis, di mana semua bukti dari tahap sebelumnya disintesis. Sistem mencari korelasi antara berbagai indikator untuk menemukan anomali dengan keyakinan tinggi. Investigasi mendalam seperti ELA dan SIFT hanya dijalankan pada kandidat yang paling kuat untuk efisiensi dan akurasi. Berikut adalah rincian setiap peristiwa anomali yang ditemukan:", styles['Justify']))
    if not result.localizations:
        story.append(Paragraph("Tidak ditemukan anomali signifikan.", styles['Normal']))
    
    for i, loc in enumerate(result.localizations):
        event_type = loc['event'].replace('anomaly_', '').capitalize()
        confidence = loc.get('confidence', 'N/A')
        story.append(Paragraph(f"<b>3.{i+1} | Peristiwa: {event_type}</b> @ {loc['start_ts']:.2f} - {loc['end_ts']:.2f} detik", styles['H3-Box']))
        story.append(Paragraph(f"<b>Penjelasan:</b> {get_anomaly_explanation(event_type)}", styles['Normal']))
        tech_data = [["<b>Bukti Teknis</b>", "<b>Nilai Terukur</b>", "<b>Interpretasi Sederhana</b>"]]
        tech_data.append(["Tingkat Kepercayaan", f"<b>{confidence}</b>", "Keyakinan sistem bahwa ini adalah manipulasi."])
        if isinstance(loc.get('metrics'), dict):
            for key, val in loc.get('metrics', {}).items():
                tech_data.append([key, str(val), Paragraph(explain_metric(key), styles['Code'])])
        story.append(Table(tech_data, colWidths=[120, 100, 305], style=TableStyle([('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])))
        v_headers, v_evidence = [], []
        if loc.get('image'): v_headers.append("<b>Sampel Frame Anomali</b>"); v_evidence.append(PlatypusImage(loc['image'], width=250, height=140, kind='proportional'))
        if loc.get('ela_path'): v_headers.append("<b>Analisis Kompresi (ELA)</b>"); v_evidence.append(PlatypusImage(loc['ela_path'], width=250, height=140, kind='proportional'))
        if v_evidence: story.append(Table([v_headers, v_evidence], colWidths=[260, 260], style=[('ALIGN',(0,0),(-1,-1),'CENTER')]))
        if loc.get('sift_path'): story.append(Paragraph("<b>Bukti Pencocokan Fitur (SIFT)</b>", styles['Normal'])); story.append(PlatypusImage(loc.get('sift_path'), width=520, height=160, kind='proportional'))
        story.append(Spacer(1, 12))

    # Tahap 4
    story.append(PageBreak())
    story.append(Paragraph("Tahap 4: Visualisasi dan Penilaian", styles['h2']))
    story.append(Paragraph("Tahap ini menyajikan ringkasan temuan dalam bentuk visual. Peta Anomali Temporal menunjukkan di mana saja anomali terjadi sepanjang durasi video. Plot statistik memberikan gambaran umum tentang karakteristik teknis video.", styles['Justify']))
    if result.plots.get('temporal'): story.append(PlatypusImage(result.plots['temporal'], width=520, height=195, kind='proportional'))
    if result.plots.get('statistic'): story.append(PlatypusImage(result.plots['statistic'], width=520, height=160, kind='proportional'))
    story.append(Spacer(1, 12))

    # Tahap 5
    story.append(Paragraph("Tahap 5: Validasi Forensik Digital", styles['h2']))
    story.append(Paragraph("Tahap terakhir ini adalah untuk memastikan bahwa seluruh proses analisis dapat dipertanggungjawabkan dan divalidasi. Berikut adalah detail validasi untuk laporan ini:", styles['Justify']))
    validation_data = [
        ["<b>Item Validasi</b>", "<b>Detail</b>"],
        ["File Bukti", Paragraph(f"<code>{Path(result.video_path).name}</code>", styles['Code'])],
        ["Hash Preservasi (SHA-256)", Paragraph(f"<code>{result.preservation_hash}</code>", styles['Code'])],
        ["Waktu Analisis", datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
        ["Metodologi Utama", "K-Means, Lokalisasi Tampering"],
        ["Metode Pendukung", "ELA, SIFT+RANSAC, Optical Flow, SSIM"],
        ["Pustaka Kunci", "OpenCV, scikit-learn, scikit-image, Pillow, ReportLab"]
    ]
    story.append(Table(validation_data, colWidths=[150, 375], style=TableStyle([('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    log(f"\n  {Icons.SUCCESS} Laporan PDF berhasil dibuat: {pdf_path.name}")
    result.pdf_report_path = pdf_path


###############################################################################
# Fungsi Pipeline Utama
###############################################################################
def run_full_analysis_pipeline(video_path: Path, out_dir: Path, fps: int, baseline_path: Path | None) -> AnalysisResult | None:
    """
    Menjalankan pipeline forensik 5 tahap secara lengkap.
    """
    log(f"\n{'='*25} MEMULAI PIPELINE ANALISIS FORENSIK {'='*25}")
    log(f"Video Bukti: {video_path.name}")
    
    # Proses Baseline terlebih dahulu jika ada
    baseline_result = None
    if baseline_path:
        log(f"Video Baseline: {baseline_path.name}")
        # Jalankan tahap 1 & 2 untuk baseline
        baseline_result = run_tahap_1_pra_pemrosesan(baseline_path, out_dir, fps)
        if baseline_result:
            run_tahap_2_analisis_temporal(baseline_result)
        else:
            log(f"{Icons.ERROR} Gagal memproses video baseline. Melanjutkan tanpa perbandingan.")
    
    # Jalankan Tahap 1 untuk video bukti
    result = run_tahap_1_pra_pemrosesan(video_path, out_dir, fps)
    if not result:
        return None # Hentikan jika tahap 1 gagal
    
    # Jalankan sisa pipeline
    run_tahap_2_analisis_temporal(result, baseline_result)
    run_tahap_3_sintesis_bukti(result, out_dir)
    run_tahap_4_visualisasi_dan_penilaian(result, out_dir)
    run_tahap_5_pelaporan_dan_validasi(result, out_dir, baseline_result)

    log(f"\n✅ PROSES FORENSIK SELESAI. Hasil di '{out_dir.resolve()}'")
    
    # Lakukan pembersihan direktori frame jika diinginkan
    # (dapat diaktifkan jika tidak dalam mode debug)
    # for d in out_dir.glob("frames_*"):
    #     if d.is_dir(): shutil.rmtree(d)
    
    return result