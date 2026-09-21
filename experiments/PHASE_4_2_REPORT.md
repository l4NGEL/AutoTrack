# Faz 4.2 — Uçtan Uca Optimizasyon: Confidence + Tracker Sweep

Script: `evaluation/optimize_finetuned_tracker.py` · Test: `tests/test_optimize_finetuned_tracker.py` (14 test)
Veri: `experiments/phase4_2_optimization_results.csv` / `.json` (56 konfigürasyon, 2 sahne)
Checkpoint: `checkpoints/yolo_visdrone_finetune/weights/best.pt` (Faz 4.1)

## 1) Amaç

Faz 4.1, fine-tune edilmiş detector'ü COCO-kalibreli `conf_threshold=0.35` ile
çalıştırdığında yüksek-yoğunluklu sahnede recall'i büyük ölçüde artırdığını
(%15.1→%68.9) ama ID-switch'i patlattığını (15→300) bulmuş, bunun en olası
nedeni olarak "conf_threshold artık bu modele göre kalibre değil" çıkarımını
yapıp **test etmeden** bırakmıştı. Bu faz tam olarak o testi yapıyor: detector +
tracker'ı TEK bir uçtan-uca boru hattı olarak, 11 confidence değeri ve küçük bir
association-parametre ızgarasında sistematik olarak tarıyor.

## 2) KRİTİK BULGU — Faz 4.1'in raporlanmış sayılarında bir düzeltme

Sweep'in ilk (hatalı) çalıştırmasında sonuçlar Faz 4.1'in raporuyla **hiç
uyuşmadı** (ör. FP_trk 8215 yerine ~4200, MOTA işareti bile farklı). Kök neden
araştırıldı ve **iki ayrı sorun** bulundu, ikisi de düzeltilip sweep sıfırdan
tekrar çalıştırıldı:

**a) Bu script'e ait, kendi içinde bir hata (yayınlanmadan yakalandı):**
`run_one_config()` ilk yazımda `missed==0` olan track'leri (yani o karede
gerçekten eşleşmemiş, sadece Kalman ile "coasting" yapan track'leri) hipotez
kümesinden filtreliyordu. Bu, `run_confidence_sweep.py` (Faz 3.0) ve
`occlusion_event_forensics.py` (Faz 3.1/4.1) ile **tutarsız** bir değerlendirme
kuralıydı — projenin tüm önceki fazlarında tracker'ın coasting sırasında
ürettiği tahmini kutular da hipotez olarak sayılır (gerçek MOTChallenge
protokolüyle aynı ruhta: tracker o karede "burada biri var" diyorsa, doğru ya
da yanlış, skorlanır). Bu filtre kaldırıldı, script projenin kurulu
konvansiyonuna geri döndürüldü (bkz. `evaluation/optimize_finetuned_tracker.py`
satır ~163-171'deki yorum).

**b) Faz 4.1'in KENDİ raporunda, bu fazda ortaya çıkan bir etiketleme hatası:**
`AdaptiveTracker.track_thresh` (varsayılan 0.5) YOLO'nun kendi
`conf_threshold`'undan bağımsız, ikinci bir sessiz eşik uygular — Faz 3.0'da
bir kez bulunmuş, Faz 4.1'in `compare_finetune_vs_baseline.py`'si yazılırken
**tekrar unutulmuş** bir tuzak. Faz 4.1'in tracking tablosu (`_collect_
baseline_hyp`, `track_thresh` override'sız) `conf_threshold=0.35` ile tespit
yapıp, ama tracker'a sadece ≥0.5 güvenli tespitleri sokuyordu. Kanıt —
düzeltilmiş sweep'in `conf=0.50, track_thresh=0.0` satırı, Faz 4.1'in
"conf=0.35" diye raporladığı satırla **ondalık basamağına kadar birebir
aynı**:

| Kaynak | MOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Faz 4.1 raporu ("conf=0.35") | -0.253 | 0.401 | 300 | 8215 | 2925 |
| Bu sweep, conf=**0.50**, track_thresh=0.0 | -0.2533 | 0.4011 | 300 | 8215 | 2925 |

Yani Faz 4.1'in MOTA/IDF1/IDSW/FP sayıları **gerçekte etkin conf≈0.50'de**
ölçülmüştü, etikette yazan 0.35'te değil. (Faz 4.1'in ayrıca raporladığı
"Recall@IoU0.5 %68.9" rakamı bundan ETKİLENMEZ — o, tracker'a hiç girmeyen, saf
tespit-seviyesi bir ölçümdür ve gerçekten conf=0.35'te hesaplanmıştı; bu
sweep'in kendi conf=0.35 satırı da recall=%67.9 ile ona yakın çıkıyor, ~1
puanlık fark Hungarian-eşleşme ile Faz 4.1'in ayrı `_count_gt_misses`
fonksiyonu arasındaki küçük bir uygulama farkı.)

**Bu neden önemli:** Faz 4.1'in "fine-tune sonrası tracker MOTA=-%25.3'e
kötüleşti" başlığı, düzeltmeden sonra bile YÖN olarak doğru ama BÜYÜKLÜK olarak
yanıltıcıydı. Gerçek conf=0.35'teki tracker performansı raporlanandan daha da
kötü (aşağıya bkz.) — ve bu fazın asıl bulduğu şey, "conf=0.35 zaten kazara
iyi bir noktaydı" değil, "doğru confidence'ı seçmek MOTA'yı -%25.3'ten
neredeyse sıfıra getirebiliyor" oluyor.

**Doğrulama (aynı zamanda script'in Faz 4.0/4.1 ile tutarlılığının bağımsız
kanıtı):** düzeltilmiş sweep'in `baseline_vanilla` satırları (vanilla
`yolov8n.pt`, conf=0.35, track_thresh=0.5 varsayılan) Faz 4.0/4.1'in kayıtlı
sayılarıyla **birebir** eşleşiyor (uav0000137: MOTA=-0.0745/IDF1=0.1144/
IDSW=15/FP=1416/FN=8377; uav0000305: MOTA=0/IDF1=0/IDSW=0/FP=0/FN=602).

## 3) Confidence sweep sonuçları (11 değer, mevcut association parametreleri: dist_ratio=0.08, match_thresh=0.3, **track_thresh=0.0** — Faz 3.0'ın sessiz-eşik disiplini)

### uav0000137_00458_v (yüksek yoğunluk, GT=9128)

| conf | P | R | F1(det) | MOTA | IDF1 | IDSW | FP_trk | FN_trk | MT | PT | ML |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.506 | 0.726 | 0.596 | -1.115 | 0.293 | 572 | 16614 | 2122 | 40 | 33 | 8 |
| 0.25 | 0.539 | 0.713 | 0.614 | -0.988 | 0.318 | 507 | 15347 | 2288 | 39 | 33 | 9 |
| 0.30 | 0.568 | 0.696 | 0.626 | -0.762 | 0.318 | 495 | 13208 | 2377 | 36 | 37 | 8 |
| 0.35 | 0.594 | 0.679 | 0.634 | -0.642 | 0.328 | 442 | 12053 | 2493 | 34 | 38 | 9 |
| 0.40 | 0.622 | 0.663 | 0.642 | -0.493 | 0.367 | 399 | 10610 | 2623 | 32 | 39 | 10 |
| 0.45 | 0.651 | 0.644 | **0.647** | -0.376 | 0.377 | 325 | 9508 | 2731 | 28 | 44 | 9 |
| 0.50 | 0.678 | 0.620 | **0.648** | -0.253 | 0.401 | 300 | 8215 | 2925 | 26 | 44 | 11 |
| 0.55 | 0.705 | 0.590 | 0.642 | -0.137 | **0.439** | 279 | 6968 | 3128 | 23 | 46 | 12 |
| 0.60 | 0.732 | 0.547 | 0.626 | -0.094 | 0.422 | 247 | 6205 | 3534 | 20 | 48 | 13 |
| 0.65 | 0.762 | 0.491 | 0.597 | -0.051 | 0.402 | 214 | 5427 | 3956 | 19 | 47 | 15 |
| 0.70 | 0.800 | 0.418 | 0.549 | **-0.049** | 0.360 | **162** | **4838** | 4576 | 17 | 44 | 20 |

**Dört farklı metrik, dört farklı tepe noktası** — kullanıcının "farklı
metrikler anlaşmazsa açıkça bildir" talimatı burada somut olarak gerçekleşti:
- Ham tespit F1'i: conf≈0.45–0.50 (0.647-0.648)
- IDF1 (kimlik-ağırlıklı): conf=0.55 (0.439)
- MOTA (FP+FN+IDSW cezalı): conf=0.70 (-0.049, ama 0.65 ile pratikte eşit: -0.051)
- IDSW (yalnız kimlik sürekliliği): conf=0.70 (162, monoton azalıyor)

Neden: MOTA = 1-(FP+FN+IDSW)/GT_obj toplamda FP baskın; conf arttıkça FP hızla
düşüyor (16614→4838) ama FN de yükseliyor (2122→4576) — FP düşüşü ağır basıyor
ve MOTA'yı conf arttıkça iyileştiriyor. IDF1 kimlik-eşleştirmeli precision/
recall'ün harmonik ortalaması olduğu için daha dengeli bir orta noktada
(conf=0.55) tepe yapıyor.

### uav0000305_00000_v (düşük yoğunluk, GT=602)

| conf | tp | P | R | MOTA | IDF1 | FP_trk |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 4 | 0.006 | 0.007 | -6.023 | 0.0052 | 3637 |
| 0.25 | 3 | 0.006 | 0.005 | -5.071 | 0.0054 | 3063 |
| 0.30 | 2 | 0.005 | 0.003 | -4.120 | 0.0058 | 2489 |
| 0.35–0.45 | 1 | ~0.004 | 0.002 | -2.8…-3.7 | ~0.003 | 1715–2230 |
| 0.50–0.70 | 0 | 0.000 | 0.000 | -0.83…-2.45 | 0.0000 | 501–1476 |

**Bu sahnede hiçbir konfigürasyon anlamlı bir tracking üretmiyor.** Recall en
iyi ihtimalle %0.7 (conf=0.20'de 602 GT'den 4 TP) — confidence eşiği hiçbir
değerde detector'ün bu sahnedeki temel başarısızlığını telafi edemiyor. MOTA'nın
conf arttıkça "iyileşmesi" bir yanılsama: TP zaten ~0 olduğu için MOTA sadece
FP azaldıkça sıfıra yaklaşıyor — "hiçbir şey tahmin etme" stratejisine
yakınsıyor. Bu, **optimizasyon değil dejenerasyondur**; script'in kendi
top-2-by-IDF1 seçimi de bunu yansıtıyor (conf=0.30 ve 0.25 — ikisi de
gürültü seviyesinde, birbirinden anlamlı şekilde farklı değil).

## 4) Tracker (association) sweep sonuçları — sadece top-2 conf adayında

Sahne 1 adayları (IDF1'e göre): **conf=0.55, conf=0.60**. Izgara:
`assoc_dist_ratio ∈ {0.08, 0.10, 0.12}` × `match_thresh ∈ {0.2, 0.3, 0.4}`
(mevcut nokta 0.08/0.3 zaten adım 3'te ölçülmüştü, tekrar edilmedi).

| conf | dist_ratio | match_thresh | MOTA | IDF1 | IDSW | FP_trk |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.08 (mevcut) | 0.3 (mevcut) | -0.137 | 0.439 | 279 | 6968 |
| 0.55 | 0.08 | 0.2 | -0.134 | **0.4394** | 279 | 6943 |
| 0.55 | 0.08 | 0.4 | -0.238 | 0.419 | 230 | 8009 |
| 0.55 | 0.10 | 0.2 | -0.112 | 0.405 | 268 | 6696 |
| 0.55 | 0.10 | 0.3 | -0.106 | 0.400 | 258 | 6663 |
| 0.55 | 0.10 | 0.4 | -0.226 | 0.412 | 219 | 7919 |
| 0.55 | 0.12 | 0.2 | -0.089 | 0.400 | 331 | 6349 |
| 0.55 | 0.12 | 0.3 | **-0.076** | 0.403 | 281 | 6307 |
| 0.55 | 0.12 | 0.4 | -0.257 | 0.412 | 242 | 8148 |
| 0.60 | 0.08 (mevcut) | 0.3 (mevcut) | -0.094 | 0.422 | 247 | 6205 |
| 0.60 | 0.08 | 0.2 | -0.094 | 0.422 | 247 | 6205 |
| 0.60 | 0.08 | 0.4 | -0.158 | 0.418 | 207 | 6939 |
| 0.60 | 0.10 | 0.2 | -0.050 | 0.411 | 234 | 5746 |
| 0.60 | 0.10 | 0.3 | -0.055 | 0.418 | 231 | 5832 |
| 0.60 | 0.10 | 0.4 | -0.163 | 0.422 | **209** | 7017 |
| 0.60 | 0.12 | 0.2 | **-0.014** | 0.416 | 277 | **5325** |
| 0.60 | 0.12 | 0.3 | -0.041 | 0.404 | 259 | 5623 |
| 0.60 | 0.12 | 0.4 | -0.141 | **0.429** | 218 | 6803 |

**Bulgu:** association-parametre ızgarası, confidence eşiğinin sağladığı
kazancın YANINDA küçük kalıyor — en iyi association kombinasyonu bile IDF1'i
mevcut noktaya göre sadece ~0.0005 artırıyor (0.4389→0.4394), pratikte gürültü
seviyesinde. Buna karşılık `dist_ratio=0.12` (mevcut 0.08'den +50%) her iki
conf adayında da MOTA'yı tutarlı biçimde iyileştiriyor (conf=0.60'ta
-0.094→-0.014, sweep'teki EN İYİ MOTA). Bu, Faz 3.5'in MOT17'de bulduğu
"dist_ratio'yu genişletmek net negatif" sonucuyla **doğrudan çelişiyor gibi
görünse de** çelişmiyor: Faz 3.5 MOT17'nin yer-seviyesi, orta yoğunluklu
sahnelerinde test edilmişti; burada VisDrone'un çok daha gürültülü/kalabalık
hava-açılı sahnesinde farklı bir optimum ortaya çıkması, iki veri setinin
farklı davrandığının bir kanıtı — MOT17 sonucu buraya körlemesine
taşınmamıştı zaten (script'in kendi docstring'i bunu ayrı ayrı test edileceğini
belirtiyordu).

## 5) Birleşik optimizasyon — combine edilmiş en iyi adaylar

Sahne 1 için üç aday, üç farklı metrik ekseninde:

| Aday | conf | dist_ratio | match_thresh | P | R | MOTA | IDF1 | IDSW | FP_trk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IDF1-optimum** | 0.55 | 0.08 | 0.2 | 0.705 | 0.590 | -0.134 | **0.439** | 279 | 6943 |
| **MOTA-optimum** | 0.60 | 0.12 | 0.2 | 0.732 | 0.547 | **-0.014** | 0.416 | 277 | 5325 |
| **IDSW-optimum** | 0.70 | 0.08 | 0.3 | 0.800 | 0.418 | -0.049 | 0.360 | **162** | 4838 |

Sahne 2 için: **yok** — hiçbir konfigürasyon anlamlı iyileşme sağlamıyor (bkz. §3).

## 6) En iyi / Pareto konfigürasyon(lar)

Tek bir "kazanan" yok — üç aday birbirine baskın değil (biri diğerinden her
eksende iyi değil), yani gerçek bir Pareto seti bu. Öneri, gerekçesiyle:

**Önerilen: IDF1-optimum (conf=0.55, dist_ratio=0.08 [değişmedi], match_thresh=0.2)**
- Gerekçe: Faz 4.0/4.1'in tüm motivasyonu düşük recall'dü (hava görüntüsünde
  COCO-detector neredeyse hiç insan görmüyordu); MOTA-optimum bunu R=0.547'ye
  kadar feda ediyor, IDF1-optimum R=0.590'da kalıyor. IDF1 zaten literatürde
  MOT için precision/recall/kimlik-sürekliliğini birlikte özetleyen daha
  dengeli tekil metrik sayılıyor. Ayrıca `dist_ratio` mevcut değerinde
  kalıyor — Faz 3.5'in "gereksiz yere genişletme" dersini bu sahnede de
  gereksiz yere bozmuyoruz, sadece `match_thresh`'i 0.3'ten 0.2'ye küçük bir
  adımla değiştiriyoruz.
- Alternatif: MOTA'ya öncelik veren bir kullanım senaryosu (ör. az ama güvenilir
  alarm) için MOTA-optimum daha uygun olur.

**Sahne 2 için: hiçbir konfigürasyon önerilmiyor** — bu sahnedeki sorun bir
parametre sorunu değil, detector'ün bu görüntü koşullarında (muhtemelen farklı
irtifa/doku/arka plan) nesneyi hiç görmemesi; çözüm bu fazın kapsamı dışında
(muhtemel kök neden Faz 4.1'de zaten not edilmişti, bu oturumda araştırılmadı).

## 7) Nihai karşılaştırma — Baseline / Faz 4.1 (etiket hatalı) / gerçek conf=0.35 / optimize edilmiş

### uav0000137_00458_v (yüksek yoğunluk)

| Konfigürasyon | P | R | MOTA | IDF1 | IDSW | FP_trk | FN_trk |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (vanilla COCO, conf=0.35) | 0.817 | 0.150 | -0.074 | 0.114 | 15 | 1416 | 8377 |
| Faz 4.1 raporu ("conf=0.35", **gerçekte eff. conf≈0.50**) | 0.678 | 0.620 | -0.253 | 0.401 | 300 | 8215 | 2925 |
| Fine-tuned, GERÇEK conf=0.35 (track_thresh=0.0) | 0.594 | 0.679 | -0.642 | 0.328 | 442 | 12053 | 2493 |
| **Fine-tuned, IDF1-optimum (Faz 4.2)** | 0.705 | 0.590 | -0.134 | **0.439** | 279 | 6943 | 3129 |
| Fine-tuned, MOTA-optimum (Faz 4.2) | 0.732 | 0.547 | **-0.014** | 0.416 | 277 | 5325 | 3652 |

### uav0000305_00000_v (düşük yoğunluk)

| Konfigürasyon | P | R | MOTA | IDF1 | FP_trk | FN_trk |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (vanilla COCO, conf=0.35) | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 602 |
| Fine-tuned (herhangi bir conf) | ~0.000 | ≤0.007 | -0.83…-6.02 | ~0.000 | 41–3637 | 591–602 |

(Sahne 2'de "en iyi" satır anlamsız — tabloya alınmadı, bkz. §3/§6.)

**Özet yorum:** IDF1-optimum konfigürasyon, Faz 4.1'in (yanlış etiketli)
sonucuna göre bile IDF1'i 0.401→0.439 artırıyor ve MOTA'yı -%25.3'ten
-%13.4'e (neredeyse yarı yarıya) iyileştiriyor — **recall'i baseline'ın
%15.1'inin dört katından fazlasında (%59.0) tutarken**. Ama hâlâ MOTA<0 —
"çözüldü" değil, "ölçülebilir, gerçek bir ikinci-tur iyileştirme." Sahne 2
tamamen çözümsüz kalıyor; bu ikili sonuç Faz 4.1'in "tek yönlü zafer değil"
tespitiyle aynı ruhta.

## 8) Genelleme ve sızıntı sınırları

- Bu sweep'in kullandığı 2 sahne (`uav0000137_00458_v`, `uav0000305_00000_v`)
  **Faz 4.0'ın eğitimde hiç kullanılmamış değerlendirme sahneleridir**
  (`confirm_no_leakage()` her çalıştırmada programatik doğrular) — eğitim
  verisiyle (`uav0000086_00000_v`, `uav0000339_00001_v`) hiçbir kesişim yok.
- **Ama** bu sweep PARAMETRE SEÇİMİNİ de aynı 2 sahne üzerinde yapıyor — yani
  bu, gerçek/bağımsız bir train/val/test ayrımı üzerinde hyperparametre seçimi
  DEĞİL, keşif niteliğinde bir hassasiyet analizi. Seçilen konfigürasyonların
  "bu 2 sahnede en iyi çalışan" olduğu söylenebilir, "genellenmiş en iyi
  konfigürasyon" olduğu SÖYLENEMEZ.
- VisDrone-MOT'ta indirilmesi mümkün ama bu oturumda İNDİRİLMEMİŞ 3 sahne daha
  var: `uav0000117_02622_v`, `uav0000182_00000_v`, `uav0000268_05773_v`. Bunlar
  gerçek bir bağımsız doğrulama kümesi için doğal aday, ama yeni bir veri
  indirmesi bu oturumun onay protokolüne göre kullanıcının ayrı onayını
  gerektiriyor — bu faz bunu YAPMADI.

## 9) Önerilen sonraki deney

1. **Öncelik:** Kullanıcı onayı ile 3 indirilmemiş VisDrone sahnesini indirip
   IDF1-optimum konfigürasyonun (conf=0.55, match_thresh=0.2) bu genuinely-
   held-out sahnelerde de tutup tutmadığını doğrulamak — bu fazın "keşif"
   iddiasını gerçek bir "doğrulanmış" iddiaya çevirecek tek adım.
2. Sahne 2'nin (`uav0000305_00000_v`) neden hâlâ tespit edilemediğini
   araştırmak (irtifa/çözünürlük/arka plan farkı hipotezi test edilmedi).
3. Eğitim verisini 2 sahneden daha fazlasına genişletip fine-tune'u tekrar
   etmek — mevcut 739 kare/2 sahne muhtemelen sahne-özel overfitting riski
   taşıyor (Faz 2.3'ün 62-kimlik overfitting uyarısıyla aynı ruhta).
