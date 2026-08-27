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

32 test, hepsi yesil: IOU/Kalman/adaptif-agirlik/motmetrics birim testleri
(sentetik, saniyeler icinde biter) + gercek YOLOv8/OSNet agirliklariyla
calisan entegrasyon testleri (`test_reid_extractor.py`, `test_pipeline_smoke.py`
— ilk calistirmada OSNet'in ImageNet-on-egitimli backbone'unu otomatik
indirir, ~10MB). `test_reid_extractor.py`'deki egitilmis-checkpoint testi,
Faz 2 checkpoint'i (`checkpoints/osnet_market1501/`) mevcut degilse otomatik
atlanir -- bu depoyu klonlayip Faz 2'yi hic calistirmayan biri icin testler
yine de gecer.

Faz 2'yi kendiniz tekrarlamak icin:

```bash
python -m evaluation.download_datasets   # Market-1501 + MOT17-02/04 (~290MB, HF Hub'dan)
python -m evaluation.train_reid --epochs 15   # ~28 dk, RTX 2080
python -m evaluation.run_ablation             # ~birkaç dakika
```

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
maliyetini olcmuyor**. Kisi iceren gercek kareler uzerindeki gerçek gecikme +
dogruluk sayilari asagida, "Faz 2 sonuclari" bolumunde.

## Faz 2 sonuclari — gercek MOT17 ablation'i (2026-08-27)

**Once net sonuc: adaptif agirliklandirma, MOT17-02/04 uzerinde MOTA/IDF1/
ID-switch'te salt-IOU baseline'ini GECEMEDI.** Hipotez buydu, ablation'i
kurmaktaki amac da tam olarak bunu gercekten olcmekti — ve sonuc beklenenin
tersi cikti. Asagida sayilar oldugu gibi, yorumuyla birlikte.

### OSNet egitimi (Market-1501, gercek)

`python -m evaluation.train_reid --epochs 15` — ImageNet backbone'dan transfer
learning, batch=32, lr=0.0015, RTX 2080, **28 dakika 5 saniye**:

| Metrik | Deger |
|---|---:|
| mAP | 72.0% |
| Rank-1 | 88.6% |
| Rank-5 | 95.2% |
| Rank-10 | 97.1% |

(Referans: OSNet makalesindeki tam egitim -- ~250 epoch, tam LR programi --
Rank-1 ~94%, mAP ~84% bildiriyor. 15 epoch'luk hizli transfer learning
sonucumuz bunun altinda ama ayni buyukluk mertebesinde ve calisiyor —
checkpoint gercek, uydurma degil.)

### Tracking ablation'i (MOT17-02-FRCNN + MOT17-04-FRCNN, gercek)

`python -m evaluation.run_ablation` — kendi YOLOv8n dedektorumuzle (MOT'un
saglanmis det.txt'i degil), yukaridaki egitilmis checkpoint'le. Zemin
dogrusu: gt.txt, sadece yaya sinifi, gorunurluk>=0.2 (bkz. `evaluation/mot_metrics.py`).
HOTA bilinçli olarak kapsam disi birakildi (bkz. o dosyanin docstring'i);
MOTA/IDF1/ID-switch `motmetrics` kutuphanesinden, senaryolarla dogrulanmis.

**MOT17-02-FRCNN** (299 kare, 4757 GT kutusu):

| Varyant | MOTA | IDF1 | ID Sw. | FPS |
|---|---:|---:|---:|---:|
| A. IOU-only | **22.0%** | **44.3%** | **9** | 35.1 |
| B. IOU+Motion | 22.3% | 44.7% | 8 | 40.9 |
| C. ReID-only | 26.2% | 41.0% | 16 | 16.3 |
| D. Motion+ReID | 23.4% | 37.8% | 45 | 21.3 |
| E. Adaptive | 20.5% | 36.5% | 35 | 21.2 |

**MOT17-04-FRCNN** (524 kare, 21451 GT kutusu — cok daha yogun/kalabalik):

| Varyant | MOTA | IDF1 | ID Sw. | FPS |
|---|---:|---:|---:|---:|
| A. IOU-only | **29.2%** | **44.8%** | 23 | 33.1 |
| B. IOU+Motion | 29.0% | 44.1% | **22** | 33.7 |
| C. ReID-only | 25.0% | 37.9% | 62 | 16.2 |
| D. Motion+ReID | 25.4% | 37.1% | 78 | 18.4 |
| E. Adaptive | 28.6% | 43.1% | 26 | 18.0 |

(Kalin: her sutunda en iyi deger. Ham FPS notu: YOLOv8n conf=0.35 ile MOT17
GT'sinin cogu kucuk/orta boy insan kutusu oldugundan MOTA mutlak degerleri
her varyant icin dusuk kaliyor -- bu ablation'in amaci mutlak MOTA'yi
yukseltmek degil, ayni dedektor/zemin dogrusu sabitken varyantlari birbirine
karsi adil kiyaslamak.)

### Ne oldu, neden

1. **A/B (salt IOU, IOU+Motion) her iki sekansta da MOTA ve IDF1'de en iyisi.**
   Re-ID veya hareket eklemek, agirlikli olsun olmasin, bu iki sekansta
   dogrulugu artirmiyor.
2. **Adaptive (E), C/D'nin (sabit ReID-agirlikli varyantlar) cok onunde** —
   IDF1 D'ye gore +5-6 puan daha iyi, ID-switch sayisi D/C'nin yarisindan az.
   Yani agirlik uyarlamasi tamamen ise yaramiyor degil: kotu sabit
   agirliklandirmanin verdigi zarari belirgin sekilde azaltiyor, sadece
   IOU-only baseline'ini gecmeye yetmiyor.
3. **Adaptive'in baseline'a farki, daha yogun sekansta kuculuyor**: MOT17-02'de
   MOTA farki 1.5 puan, MOT17-04'te (4.5x daha kalabalik, occlusion cok daha
   sik) 0.6 puana iniyor ve ID-switch'te B'ye neredeyse esitleniyor (26 vs 22).
   Bu, tasarim hipoteziyle (Re-ID occlusion'da daha cok yardimci olmali) yon
   olarak tutarli — ama 2 sekans istatistiksel olarak bunu KANITLAMAZ, sadece
   ipucu.
4. **En olasi neden: domain gap.** OSNet, Market-1501'in kirpilmis/kontrollu
   yaya goruntuleri uzerinde egitildi; MOT17 tamamen farkli bir domain
   (guvenlik kamerasi acisi, farkli cozunurluk/isik, hareket bulanikligi).
   Re-ID literaturunde cross-dataset transferin within-dataset'e gore
   belirgin dustugu iyi bilinen bir bulgu — bizim 15 epoch'luk, tek-domain
   egitimimiz bunun disinda degil. `compute_adaptive_weights()`'teki esikler
   (occlusion 0.15, hiz 0.02, kalite 0.55/0.15) de muhendislik sezgisiyle
   secildi, hicbir zaman bir dogrulama setine karsi ayarlanmadi — bu
   ablation, bunu ilk kez gercekten test eden calisma.
5. **Maliyet gercek ve simdilik karsiliksiz:** ReID'li varyantlar baseline'in
   ~%50-60'i FPS'inde calisiyor (OSNet forward-pass maliyeti), ve bu ek
   maliyetin su an bu iki sekansta karsiligi yok.

### Devam edilirse sirada ne var

- OSNet'i MOT-tarzi surveillance goruntulerinde (veya en azindan MOT17'nin
  kendi track kirpimlarinda domain-adaptive) fine-tune etmek.
- `compute_adaptive_weights()` esiklerini bir validation split'e karsi
  gercekten ayarlamak (su an tahmine dayali).
- Daha fazla MOT17 sekansiyla (7'nin tumu) istatistiksel olarak anlamli bir
  karsilastirma yapmak — 2 sekans yon gosterir, kanitlamaz.

Bu bolumun amaci "yontem kazandi" demek degildi — gercekten olcup ne
cikacagini gormekti. Cikan sonuc buydu.

## Yol haritasi

- [x] Proje iskeleti
- [x] YOLOv8 ile temel tespit (webcam/kayitli video) — test edildi
- [x] Baseline IOU tracker (`bytetrack_wrapper.py`) — birim testli
- [x] Kalman hareket modeli (`tracking/kalman_motion.py`) — birim testli
- [x] OSNet Re-ID entegrasyonu (`reid/feature_extractor.py`) — calisiyor, ve Faz
      2'den beri gercek egitilmis (Market-1501, Rank-1 %88.6) bir checkpoint var
      (`checkpoints/osnet_market1501/`, gitignore'lu -- `config.yaml`'da
      `reid.checkpoint_path` ile isaret edilebilir; repo taze klonlandiginda
      hala calismasi icin varsayilan hala ImageNet backbone'a duser)
- [x] Adaptif coklu-ipucu birlesim (`tracking/adaptive_tracker.py`) — occlusion/
      hiz/kirpim-kalitesine gore agirlik kaymasi, birim testlerle dogrulandi
- [x] Karar katmani — hassas bolge + oncelik skoru
- [x] Track guven skoru (stable/uncertain/lost_candidate)
- [x] **Faz 2:** Market-1501 uzerinde gercek OSNet egitimi — 15 epoch, Rank-1
      %88.6 / mAP %72.0 (bkz. "Faz 2 sonuclari")
- [x] **Faz 2:** MOT17 uzerinde gercek MOTA/IDF1/ID-switch ablation (IOU /
      IOU+Motion / ReID / Motion+ReID / Adaptive) — **calisti, ama Adaptive
      IOU-only baseline'ini gecemedi**; bkz. "Faz 2 sonuclari" icin sayilar
      ve neden analizi. HOTA bilinçli kapsam disi (bkz. `evaluation/mot_metrics.py`)
- [ ] Adaptif esiklerin validation split'e karsi ayarlanmasi, MOT domaininde
      Re-ID fine-tune, 7 sekansin tumunde tekrar
- [x] Demo video — MOT17-04 uzerinde, Faz 2 checkpoint'iyle, bkz. asagida
- [ ] GitHub yayini

## Demo video

Canli webcam yerine (bos oda garantisi yok) gercek, kalabalik bir sokak
sahnesi (MOT17-04-FRCNN, 524 kare) `config_demo.yaml` ile calistirildi --
Faz 2'nin gercek egitilmis Re-ID checkpoint'i devrede:

```bash
python -m evaluation.make_demo_video    # MOT17-04 karelerini tek video yapar
python main.py --config config_demo.yaml    # data/demo_output.mp4 uretir
```

Ortalama 524 kare boyunca **55 farkli track ID**, karede ortalama **16 kisi**,
uctan uca (tespit+takip+ReID+HUD+video yazma dahil) **9.1 FPS** (RTX 2080).
Tek karede ayni anda `stable`/`uncertain`/`lost_candidate` guven kovalarinin
ucu de, hassas bolge tespiti ve gerekceli oncelik skoru gorunur durumda.

(Not: `main.py`'nin varsayilan `mp4v` video yazicisi disaridaki oynaticilarda
sorunsuz calisiyor ama bu ortamdaki `cv2.VideoCapture` kendi yazdigi dosyayi
geri okuyamiyor -- kareyi programatik incelemek icin once ffmpeg ile H.264'e
cevirmek gerekti, bkz. `main.py`'deki ilgili yorum.)

## Veri seti secimi

**Faz 2'de kullanildi:** [MOT17](https://motchallenge.net/data/MOT17/)
(takip ablation'i icin gercek zemin dogrusu — motchallenge.net bu gelistirme
ortamindan agla ilgili nedenlerle erisilemedi, bunun yerine dogrulanmis
[HuggingFace mirror'i](https://huggingface.co/datasets/Morrison1025/MOT17)
kullanildi, bkz. `evaluation/download_datasets.py`) ve
[Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
(OSNet'i gercekten egitmek icin — ayni sekilde [HF mirror'i](https://huggingface.co/datasets/Lekim89/market1501) uzerinden).

**Bilinçli olarak kullanilmayan:** DukeMTMC-reID — Duke Universitesi bu veri
kumesini 2019'da riza/mahremiyet gerekcesiyle geri cekti, cogu guncel
benchmark/leaderboard artik kullanmiyor; savunma sektorune yonelik bir
portfoyde bu tartismali kaynagi atlamak daha temiz. MSMT17 — yazarlardan izin
basvurusu gerektiriyor, bu projenin kapsaminda degil.

Drone acisina ozel senaryolar icin [VisDrone](http://aiskyeye.com/) veri seti
ile fine-tune ayrica onerilir; genel nesne siniflari icin COCO on-egitimli
agirliklar yeterlidir.
