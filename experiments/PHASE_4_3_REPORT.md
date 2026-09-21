# Faz 4.3 — Dogrulama ve Genelleme

Script: `evaluation/validate_finetuned_generalization.py` · Test: `tests/test_validate_finetuned_generalization.py` (17 test)
Veri: `experiments/phase4_3_validation_results.csv`/`.json`, `experiments/phase4_3_final_test_results.csv`/`.json`, `experiments/phase4_3_selection.json`
Checkpoint: `checkpoints/yolo_visdrone_finetune/weights/best.pt` (Faz 4.1 — YOLOv8n, COCO'dan devam, imgsz=1280, 30 epoch, egitim sahneleri: `uav0000086_00000_v`+`uav0000339_00001_v`, ultralytics 8.4.9/torch 2.13.0)

## 1) Amac

Faz 4.2, parametre secimini (confidence + tracker sweep) Faz 4.0'in AYNI 2
"dokunulmamis" sahnesinde yapmisti -- kendi raporunun acikca belirttigi gibi
bu bir dogrulama degil, kesif niteliginde bir hassasiyet analiziydi. Bu faz
soruyu soruyor: **Faz 4.2'nin sectigi/onerdigi konfigurasyonlar, hic
gormedikleri 3 YENI VisDrone sahnesinde de ayni sekilde davraniyor mu?**
Bunun icin gercek bir train/validation/test ayrimi kuruldu; parametre secimi
SADECE validation'da yapildi, test sahneleri secim sirasinda HIC
incelenmedi.

## 2) Veri ayrimi

| Ayrim | Sahneler |
|---|---|
| Train (Faz 4.1 fine-tune) | `uav0000086_00000_v`, `uav0000339_00001_v` |
| **Validation (YENI, Faz 4.3)** | `uav0000117_02622_v` (349 kare), `uav0000182_00000_v` (363 kare), `uav0000268_05773_v` (978 kare) |
| Final Test (Faz 4.0/4.1/4.2 ile AYNI) | `uav0000137_00458_v`, `uav0000305_00000_v` |

`confirm_no_leakage_3way()` uc kumenin ikili kesisimlerinin TAMAMININ bos
oldugunu programatik dogruladi (`train<->val`, `train<->test`, `val<->test`)
— hicbir kesisim YOK. Validation sahnelerinin ham piksel dosyalari bu fazda
`Voxel51/visdrone-mot` HF deposundan indirildi (1690 kare, `evaluation.
download_datasets.download_visdrone_scenes()` — `download_mot17()` ile AYNI
`snapshot_download`+`allow_patterns` deseni, kalici/tekrar calistirilabilir
hale getirildi), `evaluation/visdrone_adapter.py` ile (degistirilmeden)
MOT-format sahne dizinine cevrildi. Butunluk dogrulandi: her sahnenin
`img1/` dosya sayisi `samples.json`'daki kare sayisiyla birebir, `gt.txt`
bos degil, `seqinfo.ini` mevcut.

## 3) Validation adaylari

Kullanicinin spesifikasyonundan, TAHMIN EDILMEDI (dogrudan kaynaktan
okundu — `tracking/adaptive_tracker.AdaptiveTracker.__init__` ve
`evaluation/occlusion_event_forensics.BASELINE_CONF`):

| Aday | conf | dist_ratio | match_thresh | track_thresh | Kaynak |
|---|---:|---:|---:|---:|---|
| A_idf1_optimum_4_2 | 0.55 | 0.08 | 0.2 | 0.0 | Faz 4.2 IDF1-optimum |
| B_mota_optimum_4_2 | 0.60 | 0.12 | 0.2 | 0.0 | Faz 4.2 MOTA-optimum |
| C_idsw_optimum_4_2 | 0.70 | 0.08 | 0.3 | 0.0 | Faz 4.2 IDSW-optimum |
| D_finetuned_default | 0.35 | 0.08 | 0.3 | 0.5 | Faz 4.1'in fiilen kullandigi (AdaptiveTracker varsayilanlari) |
| E_baseline_vanilla | 0.35 | 0.08 | 0.3 | 0.5 | Vanilla COCO `yolov8n.pt` (referans, secime aday DEGIL) |

Ayrica CURRENT association parametrelerinde (0.08/0.3, track_thresh=0.0) 6
degerlik bir confidence sweep: **0.45, 0.50, 0.55, 0.60, 0.65, 0.70.**

## 4) Validation sonuclari

**Agregasyon yontemi:** tespit P/R/F1 icin HAM tp/fp/fn 3 sahne arasinda
TOPLANDI (yuzdelerin ortalamasi ALINMADI); MOTA/IDF1/IDSW/MT/PT/ML icin
`mot_metrics.summarize()`'a 3 sahnenin HAM `mm.MOTAccumulator`'i birlikte
verilip `generate_overall=True` ile motmetrics'in KENDI istatistiksel
agregasyonu kullanildi (Faz 2.7/2.8'in 7-sekans agregatiyla AYNI yontem).

| Config | P | R | F1 | MOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_idf1_optimum_4_2 | 0.505 | 0.149 | 0.231 | -0.472 | 0.110 | 274 | 7991 | 10494 |
| B_mota_optimum_4_2 | 0.516 | 0.127 | 0.204 | -0.345 | 0.108 | 239 | 6107 | 10792 |
| C_idsw_optimum_4_2 | 0.548 | 0.076 | 0.133 | **-0.273** | 0.081 | **139** | 4687 | 11395 |
| D_finetuned_default | 0.413 | **0.213** | 0.281 | -0.595 | 0.115 | 294 | 9807 | 10219 |
| E_baseline_vanilla (referans) | 0.538 | 0.034 | 0.064 | -0.073 | 0.028 | 5 | 1156 | 12511 |
| sweep_conf_0.45 | 0.463 | 0.182 | 0.262 | -0.718 | **0.116** | 341 | 11535 | 10014 |
| sweep_conf_0.50 (=D, izleme katmaninda) | 0.486 | 0.168 | 0.249 | -0.595 | 0.115 | 294 | 9807 | 10219 |
| sweep_conf_0.55 (=A, izleme katmaninda) | 0.505 | 0.149 | 0.231 | -0.472 | 0.110 | 274 | 7991 | 10494 |
| sweep_conf_0.60 | 0.516 | 0.127 | 0.204 | -0.388 | 0.097 | 224 | 6711 | 10748 |
| sweep_conf_0.65 | 0.530 | 0.103 | 0.173 | -0.307 | 0.095 | 157 | 5428 | 11067 |
| sweep_conf_0.70 (=C) | 0.548 | 0.076 | 0.133 | -0.273 | 0.081 | 139 | 4687 | 11395 |

**Onemli tutarlilik notu:** `sweep_conf_0.50` ile `D_finetuned_default`'in
MOTA/IDF1/IDSW/FP/FN'i BIREBIR ayni (P/R farkli) -- bu, Faz 4.2'nin
kesfettigi `track_thresh` sessiz-ikinci-esik davranisinin bu YENI veride de
AYNEN tekrarlandiginin bagimsiz bir dogrulamasi (D'nin track_thresh=0.5
varsayilani, conf=0.35 tespit havuzunu izleme asamasinda etkin olarak 0.50'ye
kisitliyor). Benzer sekilde `sweep_conf_0.55`≡A ve `sweep_conf_0.70`≡C
birebir ayni (bu ikisi zaten ayni parametrelerle tanimli).

### Sahne bazinda (aggregate'in gizledigi asimetri)

| Sahne | En iyi R (config) | MOTA araligi | Not |
|---|---|---|---|
| `uav0000117_02622_v` (9619 GT) | %25.7 (D) | -0.05 .. -0.40 | Kismen calisiyor -- Faz 4.0'in "yuksek yogunluk" sahnesine en yakin davranis |
| `uav0000182_00000_v` (1140 GT) | %20.9 (D) | **-0.27 .. -4.62** | **Katastrofik FP** -- D'de tp=238 iken fp_det=1912 (FP, TP'nin ~8 kati) |
| `uav0000268_05773_v` (1984 GT) | %0.0 (HEPSI) | ~-0.03 .. -0.18 | **TAM tespit basarisizligi** -- 5 konfigurasyonun TUMUNDE tp=0, baseline dahil |

`uav0000268_05773_v` tam olarak Faz 4.0/4.1/4.2'nin `uav0000305_00000_v`
(dusuk yogunluk) bulgusunu TEKRARLIYOR: recall HICBIR confidence degerinde
sifirin uzerine cikmiyor -- bu, parametre sorunu degil, detector'un o
sahnede nesneyi hic gormemesi (bkz. §8).

## 5) Validation secimi

- **En yuksek IDF1:** `sweep_conf_0.45` (IDF1=0.116)
- **En yuksek MOTA:** `C_idsw_optimum_4_2` / `sweep_conf_0.70` (MOTA=-0.273, aynı config)
- **En dusuk IDSW:** `C_idsw_optimum_4_2` / `sweep_conf_0.70` (IDSW=139)
- **En yuksek Recall:** `D_finetuned_default` (R=0.213)
- **Pareto-optimal kume:** **TUM 10 aday** (E haric) -- hicbiri hicbirini UC eksende (IDF1, MOTA, IDSW) birden domine etmiyor. Bu, gercek bir Pareto cephesi; tek bir "objektif olarak en iyi" konfigurasyon YOK.

**Metrikler GUCLU sekilde anlasmazlik halinde, saklanmiyor:** `sweep_conf_
0.45`'in IDF1 "zaferi" son derece ince bir farkla (0.116 vs D'nin 0.115 vs
A'nin 0.110 -- 0.001-0.006 araliginda, muhtemelen gurultu seviyesinde) ama
MOTA/FP/IDSW BEDELI buyuk: aggregate'teki EN KOTU MOTA (-0.718) ve EN YUKSEK
FP (11535) ve EN YUKSEK IDSW (341) tam olarak bu konfigurasyona ait. Sahne
bazinda kok neden acik: sahne 117'de IDF1 kazandiriyor (0.155, en iyisi) ama
sahne 182'de MOTA'yi -4.619'a (gorulen EN KOTU deger) itiyor.

**Secim (PRIMARY=IDF1, kullanicinin belirttigi oncelik kurali, agirlikli
skor UYDURULMADI):** `sweep_conf_0.45` KILITLENDI. Bu secim, script'in
onceden-kayitli ("IDF1'e gore en yuksegi sec") mekanik kuraliyla, final test
sonuclarina HIC bakilmadan yapildi -- ama asagidaki bolumlerde acikca
gosterildigi gibi, bu razor-thin bir IDF1 farki icin buyuk bir MOTA/FP
bedeli odeyen bir secim; okuyucunun bunu "acik ara en iyi konfigurasyon"
olarak OKUMAMASI gerekir.

## 6) KILITLENMIS konfigurasyon

```
checkpoint:            checkpoints/yolo_visdrone_finetune/weights/best.pt
confidence:             0.45
association_dist_ratio: 0.08   (degismedi -- CURRENT/Faz 4.1 degeri)
match_thresh:           0.3    (degismedi -- CURRENT/Faz 4.1 degeri)
track_thresh:           0.0    (Faz 3.0/4.2 disiplini -- sessiz ikinci esik kapali)
```

## 7) Nihai test sonuclari

**Bu tablo SADECE bir degerlendirmedir -- kilitlemeden SONRA, hicbir ayar
final test sonuclarina gore degistirilmedi.**

| Sahne | Konfigurasyon | P | R | F1 | FP | FN | MOTA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uav0000137_00458_v (yuksek yogunluk) | Baseline | 0.817 | 0.150 | 0.254 | 1416 | 8377 | -0.074 | 0.114 | 15 |
| uav0000137_00458_v | Fine-tuned varsayilan (D) | 0.594 | 0.679 | 0.634 | 8215 | 2925 | -0.253 | 0.401 | 300 |
| uav0000137_00458_v | Faz 4.2 kesif (A) | 0.705 | 0.590 | 0.642 | 6943 | 3129 | -0.134 | **0.439** | 279 |
| uav0000137_00458_v | **Faz 4.3 kilitli (conf=0.45)** | 0.651 | **0.644** | 0.647 | 9508 | 2731 | **-0.376** | 0.377 | 325 |
| uav0000305_00000_v (dusuk yogunluk) | Baseline | 0.000 | 0.000 | 0.000 | 0 | 602 | **0.000** | 0.000 | 0 |
| uav0000305_00000_v | Fine-tuned varsayilan (D) | 0.003 | 0.002 | 0.002 | 1476 | 602 | -2.452 | 0.000 | 0 |
| uav0000305_00000_v | Faz 4.2 kesif (A) | 0.000 | 0.000 | 0.000 | 1181 | 602 | -1.962 | 0.000 | 0 |
| uav0000305_00000_v | Faz 4.3 kilitli (conf=0.45) | 0.004 | 0.002 | 0.003 | 1715 | 598 | -2.842 | 0.003 | 0 |

**Tutarlilik dogrulamasi:** Baseline satiri Faz 4.0/4.1/4.2'nin kayitli
sayilariyla BIREBIR ayni; Fine-tuned-varsayilan(D) satiri Faz 4.2'nin hem
"gercek conf=0.35" (P/R) hem "etkin conf=0.50" (MOTA/IDF1/IDSW/FP) sayilariyla
dogru bilesimde eslesiyor; Faz 4.2 kesif(A) satiri Faz 4.2'nin kendi
raporuyla BIREBIR ayni -- yeni script'in eskiyle tutarli oldugunun bagimsiz
kaniti.

## 8) Genelleme analizi

- `sweep_conf_0.45`, validasyonda **en yuksek IDF1'e ulasti** (0.116), ama bu
  Faz 4.2'nin exploratory secimine (A) gore final testte MOTA'yi -0.134'ten
  **-0.376'ya dusurdu** (yuksek yogunluklu sahnede) ve IDF1'i 0.439'dan
  0.377'ye indirdi -- yani **validasyonda IDF1 kazandiran secim, final
  testte hem MOTA hem IDF1'de A'ya gore geriledi.**
- Faz 4.3'un kilitli konfigurasyonu, A'ya gore **recall'i artirdi (0.590->
  0.644) ama false positive'leri de artirdi (FP 6943->9508)** -- klasik bir
  duşuk-esik-tradeoff, kullanicinin ornek ifadesiyle: "recall'i iyilestirdi
  ama false positive'leri artirdi."
- Dusuk yogunluklu test sahnesinde (`uav0000305_00000_v`) **hicbir
  konfigurasyon -- ne Faz 4.2'nin ne Faz 4.3'un secimi -- iyilesme
  saglamadi**; recall tum konfigurasyonlarda %0.2'nin altinda kaldi.
- **Sahneye gore performans GUCLU sekilde farkliydi ve bu aggregate'in
  arkasina saklanmadi:** validasyonun 3 sahnesi ucuncu, daha once
  gorulmemis bir basarisizlik modu ortaya cikardi -- `uav0000182_00000_v`'de
  MOTA -4.6'ya kadar dustu (katastrofik false-positive orani), ki bu Faz
  4.0-4.2'nin gordugu "dusuk recall ama makul FP" (uav0000305) VEYA "yuksek
  recall + orta FP" (uav0000137) modellerinin HICBIRINE uymuyor.
- **Faz 4.2'nin secimi (A), Faz 4.3'un daha saglam metodolojiyle sectigi
  konfigurasyondan final test sahnelerinde daha iyi performans gosterdi.**
  Bu bir celiski degil -- Faz 4.3'un amaci "daha iyi bir sonuc" garantilemek
  degil, "test sahnelerine bakmadan yapilan durust bir secim" saglamakti.
  Sonuc, Faz 4.2'nin sonucunun test sahnelerine ORTUSEN veriyle secildigi
  icin şansli/iyimser gorunmus olabilecegini, VE 3 validasyon sahnesinin
  (birbirinden bu kadar farkli davranarak) guvenilir bir parametre secimi
  icin henuz yeterli olmayabilecegini gosteriyor.

## 9) Sizinti / metodoloji denetimi

- Egitim sahneleri: `uav0000086_00000_v`, `uav0000339_00001_v`
- Validasyon sahneleri: `uav0000117_02622_v`, `uav0000182_00000_v`, `uav0000268_05773_v`
- Final test sahneleri: `uav0000137_00458_v`, `uav0000305_00000_v`
- **Final test, parametre secimini HIC etkilemedi** -- `select_locked_config()` her satirin `split=="validation"` oldugunu ve hicbir satirin test-sahnesi verisi icermedigini calisma-zamaninda dogrular (aksi halde `SystemExit`); final test kodu secim TAMAMLANDIKTAN ve kilitlendikten SONRA calisir.
- **Validasyon secim icin kullanildi** -- 10 adayin (4 isimli + 6 sweep) TUMU 3 validasyon sahnesinde calistirildi, secim SADECE bu sonuclardan yapildi.
- **Sizinti TESPIT EDILMEDI** -- `confirm_no_leakage_3way()` uc kumenin ikili kesisimlerinin tamamen bos oldugunu dogruladi.

## 10) Degisen dosyalar

`evaluation/validate_finetuned_generalization.py` (yeni), `tests/test_validate_finetuned_generalization.py` (yeni, 17 test), `evaluation/download_datasets.py` (VisDrone validasyon sahneleri icin `download_visdrone_scenes()` eklendi, MOT17/Market1501 davranisi degismedi), `evaluation/optimize_finetuned_tracker.py` (`run_one_config`'a geriye-uyumlu `return_accumulator` parametresi eklendi, varsayilan davranis degismedi), `experiments/PHASE_4_3_REPORT.md` + `phase4_3_validation_results.{csv,json}` + `phase4_3_final_test_results.{csv,json}` + `phase4_3_selection.json` (hepsi yeni).

## 11) Testler

188 test yesil (onceki 171'den +17, hepsi bu fazin yeni testleri). Regresyon yok.

## 12) Git

Commit ve push detaylari ana rapor yanitinda.

## 13) Nihai sonuc

- **Tespit iyilestirmesi:** fine-tune edilmis detector, baseline'a gore
  validasyon sahnelerinde de recall'i tutarli sekilde artirdi (E'nin
  R=0.034'une karsi A-D'nin R=0.076-0.213 araligi) -- Faz 4.1'in tespit-
  katmani kazanci validasyonda da (zayiflamis olsa da) GORULUYOR.
- **Izleme (tracking) iyilestirmesi:** validasyonda BELIRSIZ -- Pareto
  cephesindeki 10 adayin hicbiri digerine acik ara ustun degil; final test
  sahnelerinde Faz 4.3'un secimi Faz 4.2'ninkinden IDF1/MOTA'da DAHA KOTU
  cikti.
- **Genelleme:** KISMEN VE ESIT OLMAYAN SEKILDE var. Yuksek-yogunluk-benzeri
  sahnede (117) tespit katmani anlamli sekilde calisiyor; ama 3 validasyon
  sahnesinden biri (182) daha once gorulmemis, katastrofik bir false-positive
  modu sergiliyor ve digeri (268) Faz 4.0'dan beri bilinen tam tespit
  basarisizligini tekrarliyor. **Iki sahneden genellenmis bir basari iddiasi
  YAPILMIYOR** (kullanicinin kritik kurali) -- YENI 3 sahnede bile sonuc
  tutarli degil, bu da modelin sadece 2 egitim sahnesiyle sinirli kaldigini
  dusundurur.
- **Kalan sinirlamalar:** dusuk-yogunluk/farkli-arka-plan sahnelerinde
  (268, 305) detector nesneyi hic gormuyor; sahne 182'nin katastrofik FP
  modu kok nedeniyle arastirilmadi (bkz. §14); 3 validasyon sahnesi kendi
  icinde bu kadar farkli davrandigi icin daha fazla sahne olmadan
  "genellenmis en iyi konfigurasyon" iddiasi hala erken.
- **"Sorun cozuldu" denemez** -- olculen kanit bunu desteklemiyor.

## 14) Onerilen sonraki deney

**Sahne 182'nin katastrofik false-positive modunun kok nedenini
arastirmak** (Faz 4.0'in orijinal kok-neden yontemiyle ayni ruhta: bir
karede ham YOLO ciktisini GT kutularinin uzerine cizip gozle incele).
Bu, olculen en buyuk kalan darbogaz -- sahne 268'in sifir-recall sorunu zaten
Faz 4.0'dan beri biliniyor ve arastirildi, ama sahne 182'nin "cok fazla
yanlis pozitif" modu bu projede ILK KEZ gorulen, kok nedeni bilinmeyen yeni
bir basarisizlik sinifi.
