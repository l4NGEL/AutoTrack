# AutoTrack

Gercek zamanli coklu hedef tespit, takip ve otonom onceliklendirme sistemi.

Bir video akisinda (drone kamerasi, sabit kamera, simulasyon) birden fazla nesneyi
tespit eder, kare kare takip eder ve tanimli kurallara gore hangi hedefin
oncelikli oldugunu **acikca gerekcelendirerek** belirler. Algila -> takip et ->
karar ver zincirinin acik kaynak, portfoy olcekli implementasyonu.

## Mimari

Iki izleyici modu var, ayni Hungarian-atama koduna giren ortak bir arayuz
uzerinden: `tracker.type: iou` (baseline) ve `tracker.type: adaptive`
(varsayilan -- Kalman hareket + OSNet Re-ID + adaptif agirlik).

```
                                   +-- IOU skoru (kutu ortusmesi)
Video akisi -> [YOLOv8 Tespit] -> [Adaptif Birlesim] -- Hareket skoru (Kalman tahmini)
                     |                    |            +-- Re-ID skoru (OSNet cosine benzerligi, sadece 'person')
                     |                    v
                     |          [Hungarian Atama + Track Guveni]
                     |                    |
                     +--------------------+---> [Karar: Kural tabanli oncelik motoru] -> [Overlay + Log]
```

| Katman | Dosya | Aciklama |
|---|---|---|
| Tespit | `detection/yolo_detector.py` | YOLOv8 ile nesne tespiti |
| Takip (baseline) | `tracking/bytetrack_wrapper.py` | Salt IOU + Hungarian eslesmeli takip |
| Hareket modeli | `tracking/kalman_motion.py` | Sabit hizli Kalman filtresi (filterpy), her track icin konum tahmini |
| Gorunum modeli | `reid/feature_extractor.py` | OSNet (torchreid) ile 512-d, L2-normalize kimlik embedding'i -- sadece `person` sinifi |
| Takip (adaptif) | `tracking/adaptive_tracker.py` | IOU + hareket + Re-ID'yi adaptif agirliklarla birlestiren asil katki -- bkz. asagida |
| Karar | `decision/priority_engine.py`, `decision/zone_rules.py` | Hiz + bolge + bekleme suresine gore aciklanabilir oncelik skoru |
| Gorsellestirme | `viz/overlay_renderer.py` | Kutu, ID, skor, guven durumu, hassas bolge overlay |
| Metrik | `metrics/performance_logger.py` | FPS, karar ve track-guven gecmisi JSONL log |
| Degerlendirme | `evaluation/benchmark_fps.py` | Gercek gecikme/FPS olcumu (bkz. "Olculen sonuclar") |

### Neden salt ByteTrack degil?

`tracking/bytetrack_wrapper.py` adiyla anilsa da aslinda basitlestirilmis bir
IOU+Hungarian eslesmesidir (gercek ByteTrack'in iki asamali eslesme mantigi
yok) ve tek basina iki bilinen soruna acik: (1) hedefler kesistiginde kutu
geometrisi kimligi ayirt edemez, (2) hizli hareket eden hedeflerde ardisik
kareler arasi kutu ortusmesi dusup ID degisimine (ID switch) yol acabilir.
`tracking/adaptive_tracker.py` bu ikisini, sabit agirlik yerine **her track
icin dinamik olarak** hesaplanan bir birlesim skoruyla cozmeye calisir:

```
S = alpha * S_iou + beta * S_motion + gamma * S_reid      (alpha+beta+gamma=1)
```

`compute_adaptive_weights()` (bkz. `tracking/adaptive_tracker.py`) taban
agirliklari (varsayilan 0.4/0.3/0.3) su sinyallere gore kaydirir:

| Sinyal | Tetiklendiginde | Etki |
|---|---|---|
| Ortusme (occlusion) | Track'in tahmini kutusu baska bir track'inkiyle IOU>0.15 | gamma (Re-ID) artar, alpha (IOU) azalir -- geometri ayirt edemedigi anda gorunume guven |
| Hizli hareket | Kalman hiz vektoru buyuklugu, kare kosegenine oranla esigi asiyor | beta (hareket) artar, alpha (IOU) azalir -- hizli hedefte kutu ortusmesi guvenilmez |
| Dusuk kirpim kalitesi | Tespit guveni < 0.55 veya kutu alani referansin < %15'i | gamma (Re-ID) azalir -- kucuk/belirsiz kirpimdan cikan embedding guvenilmez |

`person` disindaki siniflar icin gamma her zaman 0'dir (OSNet insan Re-ID
verisiyle egitilir, arac/hayvan gibi siniflar icin embedding uzayi gecersizdir).

Bu formulun her bir bileseni `tests/test_adaptive_tracker.py` icinde birim
testlerle dogrulanmistir (ortusme -> gamma artisi, hiz -> beta artisi, dusuk
kalite -> gamma azalisi, capraz gecen iki hedefte ID'lerin Re-ID sayesinde
korunmasi).

### Track guveni

Her track icin (`AdaptiveTrack.track_confidence`):

```
track_confidence = 0.4 * tespit_guveni_EMA + 0.3 * hareket_guveni + 0.3 * reid_guveni
```

(Re-ID gecersiz siniflarda son iki bilesene 0.6/0.4 olarak yeniden dagitilir.)
`hareket_guveni`, Kalman tahmini ile gercek tespit arasindaki normalize
innovation'dan; `reid_guveni`, eslesen embedding'in track'in EMA-guncellenen
kimlik imzasiyla cosine benzerliginden gelir. Skor uc kovaya ayrilir:
`stable` (>=0.75), `uncertain` (0.5-0.75), `lost_candidate` (<0.5) -- HUD
overlay'de `[bucket]` etiketi olarak, JSONL logda `track_confidence` alani
olarak gorunur.

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

```bash
python main.py --config config.yaml
```

`config.yaml` icinde video kaynagini (webcam icin `0`, dosya icin yol), hassas
bolge poligonlarini, izleyici tipini (`tracker.type: iou` / `adaptive`) ve
Re-ID ayarlarini duzenleyebilirsiniz.

## Testler

```bash
python -m pytest tests/ -v
```

26 test, hepsi yesil: IOU/Kalman/adaptif-agirlik birim testleri (sentetik,
saniyeler icinde biter) + gercek YOLOv8/OSNet agirliklariyla calisan entegrasyon
testleri (`test_reid_extractor.py`, `test_pipeline_smoke.py` — ilk calistirmada
OSNet'in ImageNet-on-egitimli backbone'unu otomatik indirir, ~10MB).

## Oncelik skoru nasil hesaplanir

Her hedef icin:

```
skor = w_speed * hiz_skoru + w_zone * bolge_skoru + w_dwell * bekleme_skoru
```

Sistem her karar icin gerekce uretir (orn. `"hassas bolgede: hassas_bolge_1"`,
`"yuksek hiz (0.82)"`) — kara kutu degil, yorumlanabilir bir kural motoru.

## Olculen sonuclar (Faz 1 — gercek, kucuk olcekli)

`python -m evaluation.benchmark_fps` ile RTX 2080 uzerinde, repodaki gercek
`data/diag_*.png` kareleri (5 kare, 30x tekrar = varyant basina 150 cikarim)
uzerinden olculen uctan uca gecikme:

| Varyant | Ort. gecikme | p95 | FPS |
|---|---:|---:|---:|
| IOU baseline (ByteTracker) | 8.85 ms | 10.21 ms | 113.0 |
| Adaptive, ReID kapali | 9.22 ms | 13.49 ms | 108.4 |
| Adaptive, ReID acik (OSNet) | 8.92 ms | 14.53 ms | 112.1 |

**Onemli kisitlama:** bu 5 kare bir kamera-surucu tanilama cekimi
(`diagnose_camera.py` cikti), icinde **hic insan yok** (dogrulama:
`evaluation/benchmark_fps.py` bunu artik acikca uyariyor). Yani "ReID acik"
satirindaki OSNet hicbir kirpim uzerinde calistirilmadi — bu tablo IOU/Kalman/
Hungarian hattinin gercek gecikmesini dogru olcuyor, ama **Re-ID'nin gercek
maliyetini olcmuyor**. Kisi iceren gercek kareler uzerindeki gecikme +
dogruluk (HOTA/MOTA/IDF1/ID-switch) olcumu Faz 2'nin konusu (asagida).

## Yol haritasi

- [x] Proje iskeleti
- [x] YOLOv8 ile temel tespit (webcam/kayitli video) — test edildi
- [x] Baseline IOU tracker (`bytetrack_wrapper.py`) — birim testli
- [x] Kalman hareket modeli (`tracking/kalman_motion.py`) — birim testli
- [x] OSNet Re-ID entegrasyonu (`reid/feature_extractor.py`) — calisiyor, ama
      henuz egitilmis (Market-1501/MSMT17) agirlik yok, sadece ImageNet backbone
- [x] Adaptif coklu-ipucu birlesim (`tracking/adaptive_tracker.py`) — occlusion/
      hiz/kirpim-kalitesine gore agirlik kaymasi, birim testlerle dogrulandi
- [x] Karar katmani — hassas bolge + oncelik skoru
- [x] Track guven skoru (stable/uncertain/lost_candidate)
- [ ] **Faz 2:** Market-1501 uzerinde gercek OSNet egitimi (rank-1/mAP raporu)
- [ ] **Faz 2:** MOT17 uzerinde gercek HOTA/MOTA/IDF1/ID-switch ablation
      (IOU / IOU+Motion / ReID / Motion+ReID / Adaptive karsilastirmasi)
- [ ] Demo video, GitHub yayini

## Veri seti secimi

**Faz 2 icin planlanan:** [MOT17](https://motchallenge.net/data/MOT17/) (takip
ablation'i — HOTA/MOTA/IDF1/ID-switch icin gercek zemin dogrusu) ve
[Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
(OSNet'i gercekten egitmek ve rank-1/mAP raporlamak icin).

**Bilinçli olarak kullanilmayan:** DukeMTMC-reID — Duke Universitesi bu veri
kumesini 2019'da riza/mahremiyet gerekcesiyle geri cekti, cogu guncel
benchmark/leaderboard artik kullanmiyor; savunma sektorune yonelik bir
portfoyde bu tartismali kaynagi atlamak daha temiz. MSMT17 — yazarlardan izin
basvurusu gerektiriyor, bu projenin kapsaminda degil.

Drone acisina ozel senaryolar icin [VisDrone](http://aiskyeye.com/) veri seti
ile fine-tune ayrica onerilir; genel nesne siniflari icin COCO on-egitimli
agirliklar yeterlidir.
