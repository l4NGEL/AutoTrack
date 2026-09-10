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

# Takip icin anlamli COCO siniflarinin bir alt kumesi (COCO'nun 80 sinifinin
# tamami secilebilir olsaydi liste kullanissiz uzun olurdu). Re-ID sadece
# "Kisi" icin gecerli; digerleri IOU + Kalman hareketiyle takip edilir.
TRACKABLE_CLASSES: dict[str, int] = {
    "Kisi": 0,
    "Bisiklet": 1,
    "Araba": 2,
    "Motosiklet": 3,
    "Otobus": 5,
    "Tren": 6,
    "Kamyon": 7,
    "Tekne": 8,
    "Kus": 14,
    "Kedi": 15,
    "Kopek": 16,
    "At": 17,
    "Sirt cantasi": 24,
    "Semsiye": 25,
    "El cantasi": 26,
    "Valiz": 28,
    "Spor topu": 32,
}
DEFAULT_CLASS_LABELS = ["Kisi"]


def _labels_to_class_ids(labels) -> tuple[int, ...] | None:
    """Arayuzden secilen etiketleri COCO id demetine cevirir. Bos secim =>
    None => TUM 80 COCO sinifi takip edilir."""
    if not labels:
        return None
    return tuple(sorted(TRACKABLE_CLASSES[l] for l in labels if l in TRACKABLE_CLASSES)) or None


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

def process_video(video_path, class_labels, max_frames, progress=gr.Progress()):
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

    pipeline = TrackingPipeline(fps=fps, classes=_labels_to_class_ids(class_labels))

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
    scope = "tum siniflar" if pipeline.classes is None else f"{len(pipeline.classes)} sinif"

    summary = (
        f"{processed} kare islendi ({scope}) -- {pipeline.unique_track_count} farkli track ID, "
        f"ortalama {processed / elapsed:.1f} FPS (islem hizi, kaynak video hizi degil)."
        f"{_reid_note(pipeline)}"
    )
    return str(final_path), summary


# ---------------------------------------------------------------- PC kamerasi

def process_webcam_frame(frame_rgb, pipeline_state, class_labels):
    if frame_rgb is None:
        return None, pipeline_state
    if pipeline_state is None:
        pipeline_state = TrackingPipeline(fps=30.0, classes=_labels_to_class_ids(class_labels))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = pipeline_state.process_frame(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, pipeline_state


def reset_webcam_state():
    return None


# ------------------------------------------------------------------ IP kamera

_stream_running: dict[str, bool] = {}
STREAM_TIMEOUT_MS = 8000  # bkz. asagidaki not


def start_ip_stream(url, class_labels, session_id):
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
    pipeline = TrackingPipeline(fps=fps, classes=_labels_to_class_ids(class_labels))
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


# --------------------------------------------------------------------- Tema/stil

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f4f5f0",
    body_background_fill_dark="#12140f",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1b1d17",
    block_border_width="1px",
    block_radius="12px",
    # Etiketler duz metin olsun -- varsayilan Soft temasi bunlari dolu
    # (primary renkli) rozetlere ceviriyor, asiri gorunuyor.
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_border_width="0px",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_label_text_weight="500",
    block_label_text_size="*text_sm",
    button_large_radius="10px",
    button_small_radius="8px",
    button_primary_background_fill="#2f6e4d",
    button_primary_background_fill_hover="#26583e",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_dark="*neutral_800",
)

CSS = """
.gradio-container { max-width: 1060px !important; margin: 0 auto !important; }
#at-header {
  position: relative; padding: 24px 30px 20px; margin: 4px 0 10px;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary); border-radius: 14px;
}
#at-header::before, #at-header::after {
  content: ""; position: absolute; width: 14px; height: 14px; border: 2px solid #3f8a63;
}
#at-header::before { top: 9px; left: 9px; border-right: 0; border-bottom: 0; }
#at-header::after { bottom: 9px; right: 9px; border-left: 0; border-top: 0; }
#at-header h1 {
  margin: 0 0 7px; font-size: 1.55rem; font-weight: 600; letter-spacing: -0.015em;
}
#at-header .at-sub { margin: 0; font-size: 0.88rem; line-height: 1.6; opacity: 0.7; }
#at-header .at-sub code {
  font-size: 0.82em; padding: 1px 5px; border-radius: 5px;
  background: var(--neutral-100); border: 1px solid var(--border-color-primary);
}
.dark #at-header .at-sub code { background: var(--neutral-800); }
/* etiketleri iyice sakinlestir */
.block .label-wrap span, span[data-testid="block-info"], .gradio-container label span:first-child {
  font-weight: 500 !important;
}
.at-mono textarea {
  font-family: var(--font-mono) !important; font-size: 0.83rem !important; line-height: 1.55 !important;
}
.tabs button[role="tab"] { font-weight: 600; font-size: 0.92rem; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div id="at-header">
  <h1>AutoTrack</h1>
  <p class="at-sub">
    YOLOv8 tespit &nbsp;·&nbsp; Kalman hareket &nbsp;·&nbsp; OSNet Re-ID &nbsp;·&nbsp;
    adaptif Hungarian eslesme &nbsp;—&nbsp; coklu hedef takip.<br/>
    Uc modun ucu de ayni motoru (<code>interface/pipeline.py</code>) kullanir;
    <code>main.py</code>'ye dokunulmaz.
  </p>
</div>
"""


def _class_picker(elem_id: str):
    return gr.Dropdown(
        choices=list(TRACKABLE_CLASSES.keys()),
        value=DEFAULT_CLASS_LABELS,
        multiselect=True,
        label="Takip edilecek siniflar",
        info="Bos birakilirsa 80 COCO sinifinin tamami. Re-ID sadece 'Kisi' icin; "
             "digerleri IOU + Kalman ile takip edilir.",
        elem_id=elem_id,
    )


# --------------------------------------------------------------------- Arayuz

with gr.Blocks(title="AutoTrack", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("Video yukle"):
            with gr.Row(equal_height=True):
                video_in = gr.Video(label="Kaynak video", height=340)
                video_out = gr.Video(label="Islenmis cikti", interactive=False, height=340)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    video_classes = _class_picker("video-classes")
                with gr.Column(scale=2):
                    video_max_frames = gr.Slider(
                        minimum=100, maximum=2000, step=100, value=MAX_UPLOAD_FRAMES_DEFAULT,
                        label="Islenecek maksimum kare",
                    )
            analyze_btn = gr.Button("Analiz et", variant="primary", size="lg")
            video_summary = gr.Textbox(
                label="Ozet", interactive=False, lines=2, elem_classes="at-mono",
            )

            analyze_btn.click(
                fn=process_video,
                inputs=[video_in, video_classes, video_max_frames],
                outputs=[video_out, video_summary],
            )

        with gr.Tab("PC kamerasi"):
            gr.Markdown("Tarayici kamera izni isteyecek. Her kare geldigi anda canli islenir.")
            webcam_classes = _class_picker("webcam-classes")
            with gr.Row(equal_height=True):
                webcam_in = gr.Image(sources=["webcam"], streaming=True, label="Kamera", height=380)
                webcam_out = gr.Image(label="Islenmis cikti", height=380)
            webcam_reset_btn = gr.Button("Takibi sifirla (yeni ID'lerle basla)")
            webcam_state = gr.State(None)

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in, webcam_state, webcam_classes],
                outputs=[webcam_out, webcam_state],
            )
            webcam_reset_btn.click(fn=reset_webcam_state, outputs=webcam_state)

        with gr.Tab("IP kamera"):
            gr.Markdown(
                "Sunucu tarafinda bir RTSP/HTTP akisina dogrudan baglanir (tarayici "
                "kamera izni GEREKMEZ). Ornek: `rtsp://kullanici:sifre@192.168.1.50:554/stream1` "
                "veya bir telefon IP-kamera uygulamasinin verdigi `http://192.168.1.60:8080/video`."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    ip_url = gr.Textbox(label="Akis URL'si", placeholder="rtsp://...")
                    ip_classes = _class_picker("ip-classes")
                    with gr.Row():
                        ip_start_btn = gr.Button("Baglan ve baslat", variant="primary")
                        ip_stop_btn = gr.Button("Durdur")
                    ip_status = gr.Textbox(label="Durum", interactive=False, elem_classes="at-mono")
                with gr.Column(scale=1):
                    ip_out = gr.Image(label="Canli islenmis akis", height=420)
            ip_session_id = gr.State(_new_session_id)

            ip_start_btn.click(
                fn=start_ip_stream,
                inputs=[ip_url, ip_classes, ip_session_id],
                outputs=[ip_out, ip_status],
            )
            ip_stop_btn.click(fn=stop_ip_stream, inputs=ip_session_id, outputs=ip_status)


def _prewarm():
    print("AutoTrack: modeller yukleniyor (YOLOv8n + OSNet Re-ID)...")
    TrackingPipeline(classes=(0,))
    print("AutoTrack: modeller hazir.")


if __name__ == "__main__":
    _prewarm()
    demo.queue().launch()
