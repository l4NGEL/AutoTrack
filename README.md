# AutoTrack

Gercek zamanli coklu hedef tespit, takip ve otonom onceliklendirme sistemi.

Bir video akisinda (drone kamerasi, sabit kamera, simulasyon) birden fazla nesneyi
tespit eder, kare kare takip eder ve tanimli kurallara gore hangi hedefin
oncelikli oldugunu **acikca gerekcelendirerek** belirler. Algila -> takip et ->
karar ver zincirinin acik kaynak, portfoy olcekli implementasyonu.

## Mimari

```
Video akisi -> [Tespit: YOLOv8] -> [Takip: ByteTrack (IOU tabanli)]
            -> [Karar: Kural tabanli oncelik motoru] -> [Overlay + Log]
```

| Katman | Dosya | Aciklama |
|---|---|---|
| Tespit | `detection/yolo_detector.py` | YOLOv8 ile nesne tespiti |
| Takip | `tracking/bytetrack_wrapper.py` | IOU eslesmeli kalici ID takibi |
| Karar | `decision/priority_engine.py`, `decision/zone_rules.py` | Hiz + bolge + bekleme suresine gore aciklanabilir oncelik skoru |
| Gorsellestirme | `viz/overlay_renderer.py` | Kutu, ID, skor, hassas bolge overlay |
| Metrik | `metrics/performance_logger.py` | FPS ve karar gecmisi JSONL log |

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

```bash
python main.py --config config.yaml
```

`config.yaml` icinde video kaynagini (webcam icin `0`, dosya icin yol), hassas
bolge poligonlarini ve oncelik agirliklarini duzenleyebilirsiniz.

## Oncelik skoru nasil hesaplanir

Her hedef icin:

```
skor = w_speed * hiz_skoru + w_zone * bolge_skoru + w_dwell * bekleme_skoru
```

Sistem her karar icin gerekce uretir (orn. `"hassas bolgede: hassas_bolge_1"`,
`"yuksek hiz (0.82)"`) — kara kutu degil, yorumlanabilir bir kural motoru.

## Yol haritasi

- [x] Proje iskeleti
- [ ] YOLOv8 ile temel tespit (webcam/kayitli video)
- [ ] ByteTrack entegrasyonu — kararli ID takibi
- [ ] Karar katmani — hassas bolge + oncelik skoru ayarlama
- [ ] Demo video, metrik raporu, GitHub yayini

## Veri seti onerisi

Drone acisina ozel senaryolar icin [VisDrone](http://aiskyeye.com/) veri seti
ile fine-tune onerilir; genel nesne siniflari icin COCO on-egitimli agirliklar
yeterlidir.
