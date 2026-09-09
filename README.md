# AutoTrack

Gercek zamanli coklu hedef tespit, takip ve otonom onceliklendirme sistemi.

Bir video akisinda (drone kamerasi, sabit kamera, simulasyon) birden fazla nesneyi
tespit eder, kare kare takip eder ve tanimli kurallara gore hangi hedefin
oncelikli oldugunu **acikca gerekcelendirerek** belirler. Algila -> takip et ->
karar ver zincirinin acik kaynak, portfoy olcekli implementasyonu.

## Öne çıkanlar

Bu proje "YOLO + tracker bağladım" seviyesinde değil -- gerçek MOT17
benchmark verisiyle, 7 sekansta, çok-fazlı bir **kök-neden araştırması**.
Her iddia bir sonraki fazda broader/held-out veriyle test edildi; yaklaşık
yarısı ilk halinde doğrulanmadı ve bu **saklanmadı**:

- **Gerçek eğitim + gerçek benchmark:** OSNet Re-ID, Market-1501 üzerinde
  sıfırdan eğitildi (Rank-1 %88.6, mAP %72.0), sonra MOT domenine fine-tune
  edildi; tüm sonuçlar 7 gerçek MOT17 sekansında (`motmetrics` ile
  MOTA/IDF1/ID-switch) ölçüldü.
- **Bir ID-switch şikayeti, tam bir araştırma zincirine dönüştü:**
  association algoritması (Hungarian -- zaten vardı) elendi → detection
  recall hipotezi zayıflatıldı (conf_threshold sweep) → asıl kök neden
  bulundu (ORR metriğinin sınır-karesi tanımı, kalabalıkla **r=0.892**
  korele, 7 sekansta doğrulandı) → ikinci, ayrı bir hata modu keşfedildi
  (sürekli kısmi görünürlük, **r=0.935** korele) → önerilen bir düzeltme
  izole bir prototipte test edildi (**negatif sonuç, saklanmadı**).
- **Kendi kodunda bulunup düzeltilen gerçek hatalar:** hayalet-kutu çizim
  hatası, iki ayrı mesafe-sınırı hatası (685px ve 309px'lik imkansız
  sıçramalar), fps-körlüğü (bir sekansta MOTA -%10.6'ya çöküyordu),
  NumPy 2.0/PyTorch 2.6 uyumluluk kırılmaları, ve arka-plan analiz
  araçlarının kendisinde bulunan 3 ayrı sınıflandırma hatası.
- **127 birim/entegrasyon test**, her yeni bulgu once sentetik veriyle
  kilitlendi.
- **Kapsam genişletme testi de aynı disiplinle:** "drone kamerası" senaryosu
  VisDrone-MOT ile ilk kez gerçek veride sınandı -- sonuç "iyi çalışıyor"
  değil, **detector'un COCO-eğitimli ağırlıklarla bu ölçekte "person"
  sınıfını hiç tespit edemediği**, kök nedeniyle (ham YOLO çıktısı
  incelenerek) doğrulandı. Sonuç olumsuz çıkınca pahalı takip-analizi
  scriptleri BİLİNÇLİ OLARAK çalıştırılmadı -- yanlış katmanda zaman
  harcamamak için.

Bu araştırmanın tam, kronolojik dökümü (her fazın gerçek sayıları, kod
referansları ve dürüst sınırlamalarıyla) [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)'de.

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
| Arayuz | `interface/app.py`, `interface/pipeline.py` | Video yukle / PC kamerasi / IP kamera arayuzu -- bkz. asagida |

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
Skor uc kovaya ayrilir: `stable` (>=0.75), `uncertain` (0.5-0.75),
`lost_candidate` (<0.5) -- HUD overlay'de `[bucket]` etiketi olarak, JSONL
logda `track_confidence` alani olarak gorunur.

## Arayüz -- canlı demo

```bash
python interface/app.py
```

`http://127.0.0.1:7860` adresinde üç modlu bir web arayüzü açılır -- üçü de
`interface/pipeline.py` üzerinden **aynı** tespit+takip+Re-ID+öncelik
motorunu kullanır, `main.py`'nin kendisine hiç dokunmadan:

| Sekme | Ne yapar |
|---|---|
| **Video Yükle** | Bir video dosyası yükleyin, tüm boru hattı üzerinden geçirilip overlay'li çıktı video + özet (kare sayısı, benzersiz ID sayısı, FPS) döner |
| **PC Kamerası** | Tarayıcı webcam izni ister, her kare canlı olarak işlenir |
| **IP Kamera** | Bir RTSP/HTTP akış URL'si girip sunucu tarafında bağlanır (örn. `rtsp://kullanici:sifre@192.168.1.50:554/stream1`) -- tarayıcı izni gerekmez |

Notlar:
- Her üç modda da "sadece `person` sınıfı" anahtarı var -- kapatılırsa
  tüm COCO sınıfları takip edilir ama Re-ID sadece kişiler için geçerli
  kalır (bkz. `REID_ELIGIBLE_CLASSES`).
- Hassas bölge (zone) overlay'i **bilinçli olarak kapalı**: `config.yaml`
  içindeki poligonlar belirli bir sahnenin çözünürlüğüne göre elle
  ayarlanmış, rastgele bir video/webcam görüntüsüne uygulanması anlamsız
  bir dikdörtgen çizer. Hız + bekleme süresi bileşenleri hâlâ çalışır.
- Fine-tune edilmiş Re-ID checkpoint'i (`checkpoints/osnet_mot_finetuned/`)
  büyük olduğu için git'e commit'lenmedi -- depoyu klonlayıp Faz 2/2.3'ü
  kendiniz çalıştırmadıysanız arayüz otomatik olarak ImageNet-backbone'a
  düşer ve bunu ekranda açıkça belirtir (kimlik eşleşmesi bu haliyle zayıf
  olur, ama arayüz çökmez).
- Video Yükle sekmesindeki çıktı, tarayıcıda oynatılabilmesi için `ffmpeg`
  ile H.264'e çevrilir (bkz. `main.py`'deki aynı mp4v/tarayıcı uyumluluk
  notu); `ffmpeg` PATH'te yoksa ham dosya (oynatma garantisi olmadan) döner.

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
Re-ID ayarlarini duzenleyebilirsiniz. Canlı bir arayüz için yukarıdaki
"Arayüz" bölümüne bakın.

## Testler

```bash
python -m pytest tests/ -v
```

127 test, hepsi yesil: IOU/Kalman/adaptif-agirlik/motmetrics/forensics birim
testleri (sentetik, saniyeler icinde biter) + gercek YOLOv8/OSNet
agirliklariyla calisan entegrasyon testleri (ilk calistirmada OSNet'in
ImageNet-on-egitimli backbone'unu otomatik indirir, ~10MB). Egitilmis-checkpoint
gerektiren testler, checkpoint mevcut degilse otomatik atlanir -- bu depoyu
klonlayip Faz 2'yi hic calistirmayan biri icin de testler yine de gecer.

## Oncelik skoru nasil hesaplanir

Her hedef icin:

```
skor = w_speed * hiz_skoru + w_zone * bolge_skoru + w_dwell * bekleme_skoru
```

Sistem her karar icin gerekce uretir (orn. `"hassas bolgede: hassas_bolge_1"`,
`"yuksek hiz (0.82)"`) — kara kutu degil, yorumlanabilir bir kural motoru.

## Sonuçlar özeti

Tam derivasyon, ara-adımlar ve her tablonun nasıl üretildiği için
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md). Son durum:

| Ölçüm | Sonuç |
|---|---|
| OSNet Re-ID eğitimi (Market-1501, gerçek) | Rank-1 %88.6, mAP %72.0 |
| MOT17 5-sekans agregat, Adaptive vs. IOU-only baseline | MOTA %24.8 vs %24.7, IDF1 %45.0 vs %44.0, ID-switch 123 vs **111** (baseline hâlâ önde) |
| Occlusion Recovery Rate (ölçülebilir tek sekans, MOT17-05) | %40.0–45.5 |
| Kalabalık yoğunluğu ↔ ORR sınır-hatası korelasyonu (7 sekans) | r = 0.892 |
| Kalabalık yoğunluğu ↔ kısmi-görünürlük hata sıklığı (7 sekans) | r = 0.935 |
| VisDrone (havadan görüntü) genelleme testi | Detector seviyesinde başarısız, kök nedeniyle doğrulandı |
| Uçtan uca hız (RTX 2080, MOT17-04 demo klibi) | ~9-11 FPS -- gerçek zamanlı değil |

## Bilinen sınırlamalar

Dürüst olmak, bu projenin metodolojisinin bir parçası -- bkz. research log'daki
her negatif sonuç. Öne çıkanlar:

- Mutlak MOTA (%20-31 aralığı) düşük -- bu projenin amacı SOTA doğruluk değil,
  gerçek bir kök-neden araştırma metodolojisi sergilemek.
- ~9-14 FPS (RTX 2080, tek makine ölçümü) -- gerçek zamanlı/gömülü kısıtlar
  için optimize edilmedi.
- Tüm nicel sonuçlar tek bir veri seti ailesine (MOT17, yer-seviyesi kamera,
  yaya) dayanıyor; havadan/drone görüntüsünde (VisDrone) mevcut COCO-eğitimli
  detector bu ölçekte çalışmıyor (bkz. Faz 4.0).
- Çoklu kamera / kameralar-arası (cross-camera) Re-ID henüz yok.
- Saf Python/PyTorch -- gömülü donanım veya TensorRT/ONNX gibi bir
  hızlandırma katmanı için ayrıca optimize edilmedi.

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

**Faz 4.0'da denendi:** [VisDrone](http://aiskyeye.com/) (havadan/drone
açısı) -- COCO-eğitimli detector bu ölçekte çalışmadığı için sonuç olumsuz
çıktı, detaylar [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)'de.
