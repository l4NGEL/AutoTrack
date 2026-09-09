"""AutoTrack canli arayuzu (Gradio).

Ayni takip motorunu (interface/pipeline.py -- main.py'ye dokunmadan, onun
kullandigi ayni siniflari sarmalar) uc farkli girisle calistirir:

  1. Video yukle -- dosya tamamen islenir, overlay'li video + ozet doner.
  2. PC kamerasi -- tarayicidan gelen webcam karesi canli islenir.
  3. IP kamera   -- sunucu tarafinda bir RTSP/HTTP akisina baglanip canli islenir.

Kullanim:
    python interface/app.py
(varsayilan olarak http://127.0.0.1:7860 adresinde acilir)
"""
from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr

from pipeline import TrackingPipeline, read_source_fps

RUNS_DIR = Path(__file__).parent / "_runs"
RUNS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_FRAMES_DEFAULT = 600  # ~30sn @20fps -- demo amacli ust sinir, arayuzden ayarlanabilir


def _ffmpeg_to_h264(src_path: Path, dst_path: Path) -> Path:
    """main.py'nin de yorumunda belirttigi gibi: bu ortamdaki OpenCV, kendi
    yazdigi mp4v dosyasini geri okuyamiyor, ve tarayicilar mp4v'yi guvenilir
    oynatmiyor -- H.264'e cevirmek gerekiyor. ffmpeg yoksa/basarisiz olursa
    ham dosyaya duser (oynatma o zaman tarayiciya kalir)."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst_path)],
            check=True, capture_output=True, timeout=300,
        )
        src_path.unlink(missing_ok=True)
        return dst_path
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return src_path


def _reid_note(pipeline: TrackingPipeline) -> str:
    if pipeline.reid_checkpoint_missing:
        return (" UYARI: egitilmis Re-ID checkpoint'i bulunamadi, ImageNet "
                "backbone'a dusuldu -- kimlik eslesmesi bu haliyle zayif olur.")
    return ""


# ---------------------------------------------------------------- Video yukle

def process_video(video_path, person_only, max_frames, progress=gr.Progress()):
    if not video_path:
        return None, "Once bir video dosyasi secin."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Video acilamadi: {video_path}"

    fps = read_source_fps(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_limit = int(max_frames) if max_frames else MAX_UPLOAD_FRAMES_DEFAULT
    progress_denom = min(total_frames, frame_limit) if total_frames > 0 else frame_limit

    pipeline = TrackingPipeline(fps=fps, person_only=person_only)

    stem = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    raw_path = RUNS_DIR / f"{stem}_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    processed = 0
    t0 = time.time()
    try:
        while processed < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame = pipeline.process_frame(frame)
            writer.write(frame)
            processed += 1
            progress(min(processed / progress_denom, 1.0), desc=f"{processed}/{progress_denom} kare islendi")
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raw_path.unlink(missing_ok=True)
        return None, "Videodan kare okunamadi (bozuk dosya veya desteklenmeyen kodek olabilir)."

    final_path = _ffmpeg_to_h264(raw_path, RUNS_DIR / f"{stem}.mp4")
    elapsed = time.time() - t0

    summary = (
        f"{processed} kare islendi -- {pipeline.unique_track_count} farkli track ID, "
        f"ortalama {processed / elapsed:.1f} FPS (islem hizi, kaynak video hizi degil)."
        f"{_reid_note(pipeline)}"
    )
    return str(final_path), summary


# ---------------------------------------------------------------- PC kamerasi

def process_webcam_frame(frame_rgb, pipeline_state, person_only):
    if frame_rgb is None:
        return None, pipeline_state
    if pipeline_state is None:
        pipeline_state = TrackingPipeline(fps=30.0, person_only=person_only)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = pipeline_state.process_frame(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, pipeline_state


def reset_webcam_state():
    return None


# ------------------------------------------------------------------ IP kamera

_stream_running: dict[str, bool] = {}
STREAM_TIMEOUT_MS = 8000  # bkz. asagidaki not


def start_ip_stream(url, person_only, session_id):
    if not url or not url.strip():
        yield None, "Once bir akis URL'si girin (orn. rtsp://... veya http://.../video)."
        return

    yield None, "Baglaniliyor..."
    # Varsayilan (parametresiz) cv2.VideoCapture, yanlis/erisilemez bir RTSP
    # adresinde acilana kadar (dogrulama: ~90 saniye) sessizce bekliyor --
    # arayuzde bu, tikladiktan sonra dakikalarca hicbir geri bildirim
    # olmadan "donmus" gibi gorunur. CAP_PROP_OPEN_TIMEOUT_MSEC/READ_TIMEOUT_MSEC
    # ile bunu makul bir sureye (8sn) sikistiriyoruz -- FFmpeg backend'i
    # GEREKTIRIR (bu iki ozellik sadece onda calisir), bu da zaten RTSP/HTTP
    # akislari icin dogru backend.
    cap = cv2.VideoCapture()
    cap.open(
        url.strip(), cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
         cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS],
    )
    if not cap.isOpened():
        yield None, f"Akisa baglanilamadi ({STREAM_TIMEOUT_MS/1000:.0f}sn icinde yanit yok): {url}"
        return

    _stream_running[session_id] = True
    fps = read_source_fps(cap)
    pipeline = TrackingPipeline(fps=fps, person_only=person_only)
    try:
        while _stream_running.get(session_id, False):
            ok, frame = cap.read()
            if not ok:
                yield None, "Akis sonlandi veya kare okunamadi."
                return
            annotated = pipeline.process_frame(frame)
            status = f"Canli -- suana kadar {pipeline.unique_track_count} farkli ID.{_reid_note(pipeline)}"
            yield cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), status
    finally:
        cap.release()
    yield None, "Durduruldu."


def stop_ip_stream(session_id):
    _stream_running[session_id] = False
    return "Durduruluyor..."


def _new_session_id():
    return uuid.uuid4().hex


# --------------------------------------------------------------------- Arayuz

with gr.Blocks(title="AutoTrack") as demo:
    gr.Markdown(
        "# AutoTrack\n"
        "YOLOv8 tespit + Kalman hareket + OSNet Re-ID + adaptif Hungarian eslesme "
        "ile gercek zamanli coklu hedef takip. Asagidaki uc mod da ayni takip "
        "motorunu (`interface/pipeline.py`) kullanir."
    )

    with gr.Tabs():
        with gr.Tab("Video Yukle"):
            gr.Markdown("Bir video dosyasi yukleyin -- tum boru hatti (tespit + takip + Re-ID + oncelik) uzerinde calisir.")
            video_in = gr.Video(label="Video dosyasi")
            with gr.Row():
                video_person_only = gr.Checkbox(label="Sadece 'person' sinifini takip et", value=True)
                video_max_frames = gr.Slider(
                    minimum=100, maximum=2000, step=100, value=MAX_UPLOAD_FRAMES_DEFAULT,
                    label="Islenecek maksimum kare",
                )
            analyze_btn = gr.Button("Analiz et", variant="primary")
            video_out = gr.Video(label="Islenmis cikti")
            video_summary = gr.Textbox(label="Ozet", interactive=False)

            analyze_btn.click(
                fn=process_video,
                inputs=[video_in, video_person_only, video_max_frames],
                outputs=[video_out, video_summary],
            )

        with gr.Tab("PC Kamerasi"):
            gr.Markdown("Tarayici kamera izni isteyecek. Her kare geldigi anda canli islenir.")
            webcam_person_only = gr.Checkbox(label="Sadece 'person' sinifini takip et", value=True)
            with gr.Row():
                webcam_in = gr.Image(sources=["webcam"], streaming=True, label="Kamera")
                webcam_out = gr.Image(label="Islenmis cikti")
            webcam_state = gr.State(None)
            webcam_reset_btn = gr.Button("Takibi sifirla (yeni ID'lerle basla)")

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in, webcam_state, webcam_person_only],
                outputs=[webcam_out, webcam_state],
            )
            webcam_reset_btn.click(fn=reset_webcam_state, outputs=webcam_state)

        with gr.Tab("IP Kamera"):
            gr.Markdown(
                "Sunucu tarafinda bir RTSP/HTTP akisina dogrudan baglanir (tarayici "
                "kamera izni GEREKMEZ). Ornekler: `rtsp://kullanici:sifre@192.168.1.50:554/stream1`, "
                "veya bir telefon IP-kamera uygulamasinin verdigi `http://192.168.1.60:8080/video`."
            )
            ip_url = gr.Textbox(label="Akis URL'si")
            ip_person_only = gr.Checkbox(label="Sadece 'person' sinifini takip et", value=True)
            with gr.Row():
                ip_start_btn = gr.Button("Baglan ve baslat", variant="primary")
                ip_stop_btn = gr.Button("Durdur")
            ip_out = gr.Image(label="Canli islenmis akis")
            ip_status = gr.Textbox(label="Durum", interactive=False)
            ip_session_id = gr.State(_new_session_id)

            ip_start_btn.click(
                fn=start_ip_stream,
                inputs=[ip_url, ip_person_only, ip_session_id],
                outputs=[ip_out, ip_status],
            )
            ip_stop_btn.click(fn=stop_ip_stream, inputs=ip_session_id, outputs=ip_status)


def _prewarm():
    print("AutoTrack: modeller yukleniyor (YOLOv8n + OSNet Re-ID)...")
    TrackingPipeline(person_only=True)
    print("AutoTrack: modeller hazir.")


if __name__ == "__main__":
    _prewarm()
    demo.queue().launch()
