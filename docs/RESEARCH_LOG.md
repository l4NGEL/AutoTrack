# AutoTrack — Araştırma Günlüğü

Bu dosya, [README.md](../README.md)'nin özetlediği bulguların tam, kronolojik
kaydıdır: Faz 1'den Faz 4.0'a, her fazın gerçek sayıları, kod referansları ve
—önemlisi— **doğrulanmadığında saklanmayan** negatif sonuçlarıyla birlikte.

README bu dosyanın "sıkıştırılmış" hali değil, farklı bir amaca hizmet eder:
README yeni birinin projeyi 2 dakikada anlaması için, bu dosya iddia edilen
her sayının nereden geldiğini, hangi hipotezin nasıl test edilip
elendiğini/doğrulandığını satır satır görmek isteyen biri için.

Aşağıdaki içerik, önceki bir README revizyonundan hiçbir değişiklik
yapılmadan taşınmıştır.

---

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
| A. IOU-only | **22.0%** | **44.3%** | **9** | 32.9 |
| B. IOU+Motion | 22.3% | 44.7% | 8 | 37.5 |
| C. ReID-only | 26.2% | 41.0% | 15 | 14.3 |
| D. Motion+ReID | 23.4% | 37.1% | 44 | 19.0 |
| E. Adaptive | 20.5% | 38.2% | 33 | 18.6 |

**MOT17-04-FRCNN** (524 kare, 21451 GT kutusu — cok daha yogun/kalabalik):

| Varyant | MOTA | IDF1 | ID Sw. | FPS |
|---|---:|---:|---:|---:|
| A. IOU-only | **29.2%** | **44.8%** | 23 | 31.8 |
| B. IOU+Motion | 29.0% | 44.1% | **22** | 29.6 |
| C. ReID-only | 24.9% | 36.0% | 67 | 12.3 |
| D. Motion+ReID | 25.4% | 36.7% | 78 | 13.9 |
| E. Adaptive | 28.6% | 40.7% | 24 | 14.0 |

(Kalin: her sutunda en iyi deger. Ham FPS notu: YOLOv8n conf=0.35 ile MOT17
GT'sinin cogu kucuk/orta boy insan kutusu oldugundan MOTA mutlak degerleri
her varyant icin dusuk kaliyor -- bu ablation'in amaci mutlak MOTA'yi
yukseltmek degil, ayni dedektor/zemin dogrusu sabitken varyantlari birbirine
karsi adil kiyaslamak. **Bu tablo `reid_recovery` (bkz. "Faz 2.1") dahil
guncel varsayilanlarla calistirilmistir** — asagidaki analiz metni de buna
gore guncellendi.)

### Ne oldu, neden

1. **A/B (salt IOU, IOU+Motion) her iki sekansta da MOTA ve IDF1'de en iyisi.**
   Re-ID veya hareket eklemek, agirlikli olsun olmasin, bu iki sekansta
   dogrulugu artirmiyor.
2. **Adaptive (E), C/D'nin (sabit ReID-agirlikli varyantlar) onunde** — IDF1
   MOT17-04'te D'ye gore +4.0 puan, MOT17-02'de +1.1 puan; ID-switch sayisi
   MOT17-04'te D'nin ucte birinden az (24 vs 78). Yani agirlik uyarlamasi
   tamamen ise yaramiyor degil: kotu sabit agirliklandirmanin verdigi zarari
   azaltiyor, sadece IOU-only baseline'ini gecmeye yetmiyor.
3. **Adaptive'in baseline'a MOTA farki, daha yogun sekansta kuculuyor**:
   MOT17-02'de 1.5 puan, MOT17-04'te (4.5x daha kalabalik, occlusion cok daha
   sik) 0.6 puana iniyor, ID-switch'te de B'ye neredeyse esitleniyor (24 vs 22).
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

## Faz 2.1 — canli demo'da bulunan uc gercek hata (2026-08-27)

Demo videosu izlenirken (kullanicinin ekran goruntuleriyle) uc ayri sorun
gozlemlendi; ucu de arastirilip dogrulandi, ikisi gercek hataydi ve
duzeltildi:

### 1) ID-switch: kaybolan kimlik kurtarma (lost-track gallery)

Gozlem: bir kisi bir sure izlenip kayboluyor, tekrar goründügünde yeni bir
ID aliyor. `evaluation/trace_id_switches.py` yazilip demo klibi tekrar
islendi: **16 aday ID-switch bulundu**, en netlerinden biri (20 kare bosluk,
1px konum farki) 3 kareye kirpilarak gorsel olarak dogrulandi -- **onemli
duzeltme: kisi bir seyin arkasinda degildi, tamamen gorunur ve hareketsizdi**
(occlusion degil, muhtemelen ara sira tespit atlamasi + `track_buffer=30`
asilinca eski track'in embedding'inin tamamen silinmesi).

**Fix:** `tracking/adaptive_tracker.py`'ye bir "kayip kimlik galerisi"
eklendi (`LostTrack`, `_try_revive`, `_revive_track` -- DeepSORT'un silinen
track'leri bir sure eslestirme adayi tutma fikrinin basitlestirilmisi).
Silinen bir track'in embedding'i `reid_recovery_window` (varsayilan 60) kare
boyunca saklanir; bu sure icinde eslesmemis bir tespit gelirse, once ona
karsi Re-ID benzerligi (`reid_recovery_thresh=0.75`, ham cosine) + kaba bir
mesafe siniri (`MAX_RECOVERY_DIST_RATIO=0.5 * kare kosegeni`) kontrol edilir.
Yeterince benzerse ID yeniden kullanilir. `config.yaml`'da
`tracker.adaptive.reid_recovery_*` ile ayarlanabilir/kapatilabilir. 8 yeni
birim test (`tests/test_adaptive_tracker.py`).

**Sonuc — iki farkli olcekte, iki farkli buyuklukte:**

| | Once | Sonra |
|---|---:|---:|
| Demo klibi, benzersiz track ID (heuristik) | 55 | **42** |
| Demo klibi, `trace_id_switches.py` adayi | 16 | **2** |
| MOT17-02, resmi ID-switch (motmetrics) | 35 | **33** |
| MOT17-04, resmi ID-switch (motmetrics) | 26 | **24** |

**Durustce soylemek gerekirse: demo klibindeki carpici iyilesme (55->42,
16->2) ile resmi ablation'daki (motmetrics, gercek zemin dogrusuna karsi)
iyilesme (birkaç birim) AYNI OLCEKTE DEGIL.** `trace_id_switches.py`
sadece "bir track biter, yakinda yenisi baslar" seklinde kaba bir sezgisel
arama yapiyor -- gercek zemin dogrusuna karsi dogrulanmiyor, bu yuzden
gercekte farkli iki kisinin sirf yakin/ardisik olmasindan kaynaklanan bircok
"sahte pozitif" adayi da sayiyor olabilir. Resmi metrik (motmetrics, gt.txt'e
karsi) daha az etkileyici ama guvenilir olan sayi. Ayrica MOT17-04'te IDF1
kotulesti (43.1%'den 40.7%'ye) -- kurtarma mekanizmasinin bir riski de bu:
sadece gorunum+kaba mesafeyle "canlandirmak", bazen FARKLI ama benzer
giyimli iki kisiyi yanlislikla birlestirebilir, ozellikle kalabalik
sahnelerde. Yani bu, net bir kazanc degil, **fragmantasyon hatasini kismen
birlestirme hatasi riskiyle takas eden, sekansa gore degisen bir iyilestirme.**

### 2) "Hayalet kutu": tespit edilmemis track'ler cizilmiyordu

Ayri bir gozlem: ekranda kimsenin olmadigi bir yerde kutu goruldu (bir
direk/lamba civarinda, `[uncertain]` etiketli). Sebep: `AdaptiveTrack.xyxy`
her karede Kalman TAHMINIYLE guncelleniyor -- bu tespit edilmemis
(`missed>0`) track'ler icin de gecerli, ve `viz/overlay_renderer.py` bunlari
gercek tespitmis gibi ayni sekilde ciziyordu. **Fix:** `draw_tracks()` artik
sadece `missed==0` (bu karede gercekten eslesen) track'leri ciziyor;
track'in kendisi hala hafizada kaliyor (buffer/kurtarma icin), sadece
ekranda gosterilmiyor. HUD'daki hedef sayisi da ayni mantikla duzeltildi
(`main.py`). 3 yeni birim test (`tests/test_overlay_renderer.py`).

### 3) ID 7, 23, 28 arastirmasi -- yanlis alarm, ama sezgi dogruydu

Kullanici videoyu izlerken ID 7'nin sirasiyla ID 57, 28 ve 23 oldugunu
dusundu. `main.py`'nin birebir ayni kare-okuma sirasiyla lifecycle
cikartilip kontrol edildi: **ID 7, 378 kare boyunca (klibin %72'si)
KESINTISIZ takip edildi** ve son konumu kare kenarina cok yakindi (y≈5px,
1080px'lik kare) -- kadraj disina cikti. ID 23 kare 95'te, ID 28 kare
216'da zaten BASLAMISTI, yani ID 7 hala aktifken -- ID 7 bu ikisi
"olamazdi", bu spesifik iddia yanlisti.

Ama kullanici ISRAR ETTI ve iki ayri kareden ayni gorunumlu (siyah kiyafet,
beyaz spor ayakkabi) bir kisiyi farkli ID'lerle (23 ve 7) gosteren ekran
goruntuleri paylasti. Bu sefer sadece ilk/son degil, ID 7 ve 23'un TUM
kare-kare konumlari cikartildi -- ve bu arastirma **gercek, ayri bir hata**
buldu (bkz. asagida). **Kullanicinin sezgisi dogruydu, sadece hangi ID
ciftinin yanlis oldugunu tam isaret edememisti.**

### 4) Asil hata: kayip galeride yanlis birlestirme (gercek kok neden)

ID 23'un kare-kare verisinde carpici bir sey bulundu: kare 227'de konum
(1177, 679, 1259, 930), kare 228'de (492, 565, 555, 805) -- **TEK KAREDE
685 piksel ziplama**, fiziksel olarak imkansiz. Bu, kayip-galeri devreye
girmeden ONCE, **normal Hungarian eslesmesinin kendisinde** oldu: `conf`
0.79'dan 0.54'e dustu, yani bu aslinda BASKA bir kisiyle yanlis eslesmeydi.
Bu yanlis eslesmeden sonra Kalman 30 kare boyunca (artik gecersiz) bu
konumdan ekstrapolasyon yapti (kare 255'te kutu **bos zeminde** oldugu
gorsel olarak dogrulandi), track_buffer asilinca kayip galeriye
tasindi -- ve kare 295'te uzak baska bir konumda "canlandirildi".

**Kok neden:** adaptif agirliklandirma, occlusion/hiz sinyaline gore
alpha'yi (IOU) yeterince dusurdugunde, kalabalik bir sahnede TESADUFEN
guclu bir Re-ID/hareket skoru, IOU=0 olsa bile kombine skoru
`match_thresh`'in ustune cikarabiliyordu -- yani mesafeyle ilgili hicbir
sinir yoktu.

**Fix (`tracking/adaptive_tracker.py`):**
- `MAX_ASSOCIATION_DIST_RATIO=0.15` (kare kosegeninin %15'i): birincil
  eslesmede, agirliklardan BAGIMSIZ sert bir mesafe on-elemesi -- skor ne
  olursa olsun, bu mesafenin otesindeki ciftler hic degerlendirilmiyor.
- `MAX_RECOVERY_DIST_RATIO` kayip-galeri icin 0.5'ten **0.2**'ye sikilastirildi.
- `LostTrack.last_xyxy` artik `t.xyxy` (Kalman'in surunmus tahmini) yerine
  `t.last_confirmed_xyxy` (son GERCEK tespit) kullaniyor -- kurtarma mesafe
  kontrolu artik guvenilir bir referans noktasina karsi calisiyor.

**Dogrulama:** tum 524 kare uzerinde, ardisik iki gercek tespit arasindaki
EN BUYUK merkez sicramasi artik **80px** (once: 685px, sinir: 330px) --
tarafsiz bir script ile (`tests/` disinda, tek seferlik) tum track'ler
taranarak dogrulandi. 1 yeni birim test
(`test_impossible_single_frame_jump_is_rejected_even_with_matching_appearance`).

## Faz 2.3 — domain-adaptive Re-ID fine-tune: ilk net kazanc (2026-08-27)

Kullanicinin "model mi egitsek" onerisi uzerine, Faz 2'nin ana bulgusu olan
**domain gap** hipotezini gercekten test ettik: OSNet'i Market-1501'in
kirpilmis/kontrollu goruntuleri yerine, MOT17'nin kendi (guvenlik kamerasi
acili, farkli cozunurluklu) goruntuleriyle fine-tune ettik.

**Metodoloji -- sizinti (leakage) olmasin diye:** fine-tune verisi
**MOT17-09 + MOT17-11**'den (`evaluation/extract_mot_reid_crops.py`, gt.txt
kimlikleri zayif etiket olarak) geldi -- **degerlendirmede kullanilan
MOT17-02/04'e kesinlikle dokunulmadi.** 62 kimlik, 5216 kirpim.
`evaluation/finetune_reid_mot.py`, Faz 2'nin Market-1501 checkpoint'ini
baslangic noktasi alip 15 epoch ince ayar yapti (lr=0.0003, Faz 2'nin
0.0015'inden dusuk -- sifirdan degil, ustune ekleme).

**Durustce not:** train accuracy epoch 5'ten itibaren **%100**'e ulasti --
sadece 62 kimlikle bu beklenir ve ezberleme (overfitting) riskini gosterir.
Bunun gercekten ise yarayip yaramadigini anlamanin tek yolu tamamen ayri
tutulan MOT17-02/04'te tekrar olcmekti.

**Sonuc -- E_Adaptive, `--checkpoint checkpoints/osnet_mot_finetuned/model.pth.tar` ile:**

| Sekans | Metrik | Faz 2.2 (Market-1501) | Faz 2.3 (MOT fine-tune) | A. IOU-only (baseline) |
|---|---|---:|---:|---:|
| MOT17-02 | MOTA | 20.5% | **24.0%** | 22.0% |
| MOT17-02 | IDF1 | 38.2% | **48.4%** | 44.3% |
| MOT17-02 | ID Sw. | 33 | **7** | 9 |
| MOT17-04 | MOTA | 28.6% | 28.9% | 29.2% |
| MOT17-04 | IDF1 | 40.7% | **43.0%** | 44.8% |
| MOT17-04 | ID Sw. | 24 | **20** | 23 |

**MOT17-02'de Adaptive artik ucu de metrikte (MOTA, IDF1, ID-switch) baseline'i
GECIYOR -- bu projede ilk kez.** MOT17-04'te MOTA/IDF1 hala baseline'in
biraz altinda ama fark kuculdu ve ID-switch'te simdi Adaptive daha iyi
(20 vs 23). A/B (ReID kullanmayan varyantlar) beklendigi gibi **birebir
ayni** kaldi -- checkpoint degisikligi sadece ReID kullanan varyantlari
etkiledi, bu da degisikligin dogru izole edildigini dogruluyor.

**Yorum:** domain gap hipotezi dogrulandi -- ayni mimari, ayni ablation
kodu, tek fark Re-ID agirliklarinin egitildigi veri. Ama abartmayalim: 62
kimlik kucuk bir ornek, %100 train accuracy'nin genellemeyi ne kadar
yansittigi belirsiz, ve MOT17-04'te hala tam bir zafer yok. Daha fazla
sekans (7'nin tumu, egitim/degerlendirme ayrimini koruyarak) ve gercek bir
validation split ile tekrarlanmadan bu sonucu "kanitlanmis" saymayin --
ama yon net: dogru yonde, ve MOT17-02'de gercek, olculebilir bir kazanc var.

## Faz 2.4 — Faz 2.1'in mesafe sinirinin kendisi hala gevsekti (2026-08-27)

Kullanicinin ayni gun ("direğin arkasından 1 saniye görünmezken yeni ID
alıyor") ısrarla tekrarlanan gozlemi uzerine, verdigi kesin zaman damgalari
(saniye 14 / saniye 24) ile videoda o araligi bulup **kare-kare, gorsel
olarak** dogrulandi: **kare 450'de acik renkli kiyafetli bir kadin,
kare 479'da (28 kare sonra) direk yanindaki koyu kiyafetli TAMAMEN FARKLI
bir kisi -- ayni ID'yi paylasiyorlardi.** Merkez mesafesi 306px -- Faz 2.1'in
`MAX_ASSOCIATION_DIST_RATIO=0.15` (kare kosegeninin ~330px'i) sinirinin
hemen altinda, tam sinirda kacmis.

**Kok neden -- iki kat:**
1. Sinir, track'in ne kadar suredir eslesmedigini (`missed`) hic hesaba
   katmiyordu. 28 kare boyunca (`track_buffer=30`'a cok yakin) sadece
   Kalman ekstrapolasyonuyla surunmus bir tahmine hala TAM 330px butce
   taniniyordu -- oysa bu kadar uzun sureli bir tahmine guven cok daha
   dusuk olmali.
2. Daha da temel bir sorun: taban sinirin kendisi (330px, missed=0 gibi
   "taze" bir track icin bile) tek kare icin fazla gevsekti --
   `FAST_MOTION_NORM_THRESH=0.02` (44px/kare) zaten "hizli hareket" sayiyor,
   330px bunun **7.5 kati**. Ayri bir global tarama, missed=0'dan missed=0'a
   TEK karede 309px'lik BASKA bir siçrama daha buldu (ID40, kare 478->479)
   -- yani sorun sadece uzun-coasting durumunda degil, "taze" eslesmelerde
   de vardi.

**Fix (`tracking/adaptive_tracker.py`):**
- `MAX_ASSOCIATION_DIST_RATIO` 0.15'ten **0.08**'e sikilastirildi (taban
  butce, missed=0 icin).
- `MIN_MISSED_TRUST=0.3`: butce artik `missed`e gore lineer daraliyor --
  `missed=0` -> tam butce, `missed=track_buffer` -> butcenin sadece %30'u.
- 2 yeni birim test.

**Dogrulama:** tum videoda en buyuk 5 sicrama artik hepsi **uzun bosluklu**
(30-71 kare, yani 1.5-3.5 saniye) ve mesafeleri o sureye gore fizikselcek
makul (115-340px) -- kisa-bosluklu ani ziplama deseni tamamen kayboldu.
Benzersiz track ID sayisi 43'ten **46**'ya cikti (yanlis birlestirmelerin
duzelmesiyle tutarli -- daha once birlesen farkli kisiler artik doğru
sekilde ayri ID'ler aliyor).

Bu, ayni gun icinde ayni sinifin (mesafe siniri) **ikinci** sikilastirilmasi
oldu (Faz 2.1: 685px'i durdurmak icin 330px sinir kondu; Faz 2.4'te o
330px'in kendisi 306px'e bile izin verdigi anlasilinca taban butce
~176px'e (0.08 orani) indirildi, missed'e gore daha da daralabiliyor).
Durustce: bu, "bir kere düzelt, bitti" degil,
kullanicinin ayni bolgeyi (direk civarindaki kisi) ısrarla farklı zaman
damgalarıyla tekrar tekrar isaret etmesi sayesinde bulunan, katmanli bir
hata idi -- ilk fix (Faz 2.1) gercekti ve ise yaradi, ama tek basina yeterli
degildi.

## Faz 2.5 — gorunum galerisi (embedding gallery) (2026-08-27)

Faz 2.4'teki ID7/ID46 tartismasi sirasinda (9 saniyelik bosluk -- gercek
sonuc: recovery window'u zorla uzatmak yanlis olurdu, bkz. yukarida)
kullanicinin onerdigi bir fikir degerlendirildi: tek bir EMA-harmanlanmis
embedding yerine, her track'in son gorulen **belirgin sekilde farkli**
gorunumlerinin kucuk bir galerisini (`embedding_gallery`, en fazla 10,
`GALLERY_NOVELTY_THRESH=0.85`'in altinda benzeyen yeni gorunumler eklenir)
tutmasi. Kurtarma sirasinda (`_try_revive`) hem EMA embedding'e hem
galerideki HER gorunume karsi benzerlik hesaplanip en iyisi kullaniliyor --
kisi kaybolmadan once yandan, geri donduğunde onden gorunuyorsa bile
eslesme sansi artiyor. 5 yeni birim test (biri ozellikle: EMA tek basina
basarisiz olurdu ama galerideki eski bir gorunum kurtariyor -- gercekten
calistigini kanitlayan senaryo).

**Metodolojik durustce not:** bu degisiklik Faz 2.4'un mesafe siniri
sikilastirmasiyla AYNI ablation calistirmasinda test edildi -- ikisinin
etkisini ayri ayri izole eden kontrollu bir kosum yapilmadi. Asagidaki
sayilar "Faz 2.4+2.5 birlikte, Faz 2.3'e karsi" karsilastirmasidir:

| Sekans | Metrik | Faz 2.3 | Faz 2.4+2.5 | A. IOU-only (guncel) |
|---|---|---:|---:|---:|
| MOT17-02 | MOTA | 24.0% | 23.3% | 21.5% |
| MOT17-02 | IDF1 | 48.4% | **50.1%** | 41.6% |
| MOT17-02 | ID Sw. | 7 | 10 | 10 |
| MOT17-04 | MOTA | 28.9% | **29.4%** | 29.2% |
| MOT17-04 | IDF1 | 43.0% | 44.5% | 44.8% |
| MOT17-04 | ID Sw. | 20 | **13** | 23 |

**Yorum:** MOT17-04'te net iyilesme -- artik MOTA ve ID-switch'te de
baseline'i geciyor (once sadece IDF1'e yakindi), IDF1 de baseline'a
neredeyse esit. MOT17-02'de karisik: IDF1 daha da iyilesti (+1.7 puan) ama
MOTA hafif dustu (-0.7) ve ID-switch baseline'a esitlendi (once 7, simdi
10 -- hala baseline'in (10) altinda degil ustunde de degil). **A_IOU_only
baseline'inin KENDI sayilari da degisti** (Faz 2.3'te MOTA 22.0/IDF1 44.3/
IDSw 9, simdi 21.5/41.6/10) -- cunku Faz 2.4'un mesafe siniri ReID
kullanmayan varyantlari da etkiliyor (agirliklardan bagimsiz, TUM ciftlere
uygulaniyor). Yani bu tablodaki degisim sadece galerinin etkisi degil,
mesafe siniri + galeri birlikte. Net sonuc: yon genelde olumlu (ozellikle
MOT17-04), ama "galeri X puan kazandirdi" diye kesin bir sayi veremeyiz --
bunu izole etmek icin ayri bir kontrollu kosum gerekir.

## Faz 2.6 — Faz 2.4 ile Faz 2.5'i izole eden kontrollu ablation (2026-08-27)

Kullanicinin (ChatGPT'den aktardigi) onerisi uzerine `tracking/adaptive_tracker.py`'ye
iki bagimsiz anahtar eklendi (`use_missed_trust`, `use_embedding_gallery`,
ayrica `association_dist_ratio`), boylece Faz 2.4 (mesafe siniri) ve Faz 2.5
(gorunum galerisi) 4 ayri kosulda tek tek olculebildi
(`evaluation/run_isolation_ablation.py`, aynı fine-tune edilmis checkpoint,
E_Adaptive agirliklariyla, MOT17-02/04):

| Sekans | Kosul | MOTA | IDF1 | ID Sw. |
|---|---|---:|---:|---:|
| MOT17-02 | Baseline (ikisi de kapali) | 24.0% | 48.4% | 7 |
| MOT17-02 | Sadece Faz 2.4 | 23.3% | 50.1% | 10 |
| MOT17-02 | Sadece Faz 2.5 | **24.0%** | **48.4%** | **7** |
| MOT17-02 | Faz 2.4+2.5 | 23.3% | 50.1% | 10 |
| MOT17-04 | Baseline (ikisi de kapali) | 28.9% | 43.0% | 20 |
| MOT17-04 | Sadece Faz 2.4 | 29.4% | 44.3% | 14 |
| MOT17-04 | Sadece Faz 2.5 | 28.9% | 43.5% | 18 |
| MOT17-04 | Faz 2.4+2.5 | 29.4% | 44.5% | 13 |

**Net, artik kesin cevap: "Sadece Faz 2.5" satiri MOT17-02'de Baseline'la
BIREBIR AYNI** (24.0/48.4/7 -- virgulüne kadar). Yani bu sekansta gorunum
galerisinin olcume yansiyan HICBIR etkisi olmadi -- muhtemelen bu sekansta
galerinin devreye girecegi turden (kaybolup farkli aciyla geri donen) bir
kurtarma senaryosu hic yasanmadi. MOT17-04'te ise kucuk ama gercek bir etki
var: Baseline->Sadece 2.5, ID-switch 20->18 (-2); 2.4-only->2.4+2.5,
ID-switch 14->13 (-1). Yani onceki Faz 2.5 bolumundeki iyilesmenin BUYUK
COGUNLUGU Faz 2.4'ten (mesafe siniri) geliyormus, galeri sadece MOT17-04'te
kucuk bir ek katki sagliyor, MOT17-02'de olcumde hic gorunmuyor. Bu, "gallery
teorik bir fikir degil, gercekten calisiyor" derken abartmamak gerektigini
gosteriyor -- calisiyor ama etkisi kucuk ve sekansa bagli, bulunmasi zor
degildi cunku dogru deneyi kurunca hemen ortaya cikti.

## Faz 2.7 — 7 sekansta genisletilince Faz 2.4 aslinda NET NEGATIF cikti (2026-08-27)

Kullanicinin onerisiyle, MOT17'nin FRCNN varyantindaki 7 sekansin TUMUNDE
(02,04,05,09,10,11,13) ayni 3-kosullu (Baseline / sadece Faz 2.4 /
Faz 2.4+2.5) ablation calistirildi (`evaluation/run_seven_sequence_ablation.py`).
**09 ve 11, Re-ID checkpoint'inin fine-tune EGITIMINDE kullanildigi icin**
(bkz. Faz 2.3) "kirli" sayilip aggregate'in disinda tutuldu; aggregate
sadece 5 "temiz" (hic egitimde gorulmemis) sekans uzerinden hesaplandi.

| Sekans (temiz) | Kosul | MOTA | IDF1 | ID Sw. |
|---|---|---:|---:|---:|
| MOT17-02 | Baseline / 2.4+2.5 | 24.0% / 23.3% | 48.4% / 50.1% | 7 / 10 |
| MOT17-04 | Baseline / 2.4+2.5 | 28.9% / **29.4%** | 43.0% / 44.5% | 20 / **13** |
| MOT17-05 | Baseline / 2.4+2.5 | **8.1%** / -10.6% | 37.3% / 31.7% | 50 / 66 |
| MOT17-10 | Baseline / 2.4+2.5 | **26.5%** / 25.5% | 51.1% / 50.4% | 11 / 14 |
| MOT17-13 | Baseline / 2.4+2.5 | **5.7%** / 4.6% | 38.0% / 38.6% | 23 / 22 |
| **AGGREGATE (5 temiz)** | Baseline / 2.4+2.5 | **24.7%** / 23.3% | 44.0% / 44.7% | **111** / 125 |

(Kalin: o satirdaki en iyi deger. Tam FP/FN dahil tum sayilar ve
kirli/09-11 satirlari icin bkz. betigin ciktisi / git tarihi.)

**Sonuc durustce: AGREGAT SEVIYEDE Faz 2.4 (mesafe siniri) MOTA'da (24.7%->23.3%)
VE ID-switch'te (111->125) Baseline'dan KOTU.** Sadece IDF1 hafif iyilesiyor
(44.0%->44.7%). MOT17-04, oncesinde Faz 2.4'un "net kazanc" oldugu tek
kanit noktasiydi ve **temsili degilmis** -- 5 sekansin sadece 1'i acikca
kazaniyor (04), 3'u aciyor (02'de ID-switch, 05'te COK KOTU, 10'da), 1'i
karisik (13).

**MOT17-05'teki cokus icin somut, dogrulanmis bir aciklama var:** bu sekans
**640x480 cozunurlukte, 14 FPS'te** cekilmis -- kosegeni 800px (digerleri
1920x1080'de 2203px). `MAX_ASSOCIATION_DIST_RATIO` kare KOSEGENININ bir
orani (0.08) olarak tanimli, kare HIZINDAN (fps) bagimsiz. 14 FPS'te her
kare 1/14s'yi temsil eder (30 FPS'in ~2.1 kati) -- yani ayni gercek yurume
hizinda bir kisi, dusuk fps'te kare basina COK DAHA FAZLA piksel kat eder,
ama sabit-oranli sinirim bunu hesaba katmiyor. Sonuc: MOT17-05'te GECERLI
eslesmeler bile reddediliyor -- Faz 2.4/2.6'nin cozdugu sorunun (imkansiz
ziplamalari kabul etmek) TAM TERSI bir hata modu. MOT17-13 (25 FPS, 1920x1080,
resmi MOT17 aciklamalarina gore hareketli/otobus-monteli kamera -- bu
ikinci kismi bu oturumda gorsel olarak dogrulamadim, sadece bilinen
benchmark bilgisiyle) de hafif kotulesiyor, muhtemelen benzer ama daha
hafif bir sebeple.

**Anlami:** mesafe sinirini SADECE MOT17-04'te (statik kamera, 30fps,
1080p) dogrulamak yeterli degilmis -- 0.08 orani o sekansa fazla ozel
kalibre olmus olabilir. Gercek duzeltme muhtemelen kare HIZINA gore
olceklenen bir butce (`piksel/saniye` cinsinden sabit bir tavan, `piksel/kare`
degil) -- ama bu, bu oturumda UYGULANMADI, sadece TESHIS EDILDI. Su anki
kod (Faz 2.4-2.6) MOT17-04 gibi 25-30fps/1080p sekanslarda iyi, dusuk-fps
veya farkli cozunurluklu goruntude DIKKATLI kullanilmali.

## Faz 2.8 — fps-farkinda mesafe butcesi: teshis dogrulandi (2026-08-27)

Faz 2.7'de teshis edilen duzeltme uygulandi: `tracking/adaptive_tracker.py`'ye
`fps` parametresi ve `REFERENCE_FPS=30.0` sabiti eklendi.
`max_dist = association_dist_ratio * frame_diag * missed_trust * (REFERENCE_FPS / fps)`
-- dusuk fps -> daha genis butce (kare basina daha fazla gercek zaman
gectigi icin). `evaluation/mot_metrics.read_seq_fps()` her sekansin
`seqinfo.ini`'sinden gercek kare hizini okur; `main.py` da artik canli
kaynaktan (`cap.get(cv2.CAP_PROP_FPS)`) okuyor. 3 yeni test.

Ayni 7 sekansta, D kosulu (Faz 2.4+2.5+fps-farkindalik) olarak yeniden
olculdu:

| Sekans (temiz) | A. Baseline | D. +fps-farkinda (Faz 2.8) |
|---|---:|---:|
| MOT17-02 MOTA | 24.0% | 23.3% |
| MOT17-04 MOTA | 28.9% | **29.4%** |
| MOT17-05 MOTA | 8.1% | **10.1%** (B/C'de -10.6%'ydi!) |
| MOT17-10 MOTA | 26.5% | 25.5% |
| MOT17-13 MOTA | 5.7% | 5.3% |
| **AGGREGATE MOTA** | 24.7% | **24.8%** |
| **AGGREGATE IDF1** | 44.0% | **45.0%** |
| **AGGREGATE ID Sw.** | **111** | 123 |

**Teshis dogrulandi: MOT17-05'teki felaket (-%10.6 MOTA) tamamen duzeldi --
%10.1'e cikti, orijinal Baseline'dan (8.1%) bile iyi.** Agregatta artik
MOTA'da Baseline'i (marjinal, 24.7->24.8) ve IDF1'de acikca (44.0->45.0)
geciyor. **Ama ID-switch agregatta hala Baseline'in gerisinde** (111 vs 123)
-- MOT17-05'in kendisinde ID-switch hala yuksek (50->65, MOTA duzelmesine
ragmen), yani bu sekans butunuyle "cozulmus" degil, sadece en buyuk hata
modu (gecerli eslesmeleri reddetme) giderildi. MOT17-02/10'da da ID-switch
hala Baseline'in az ustunde. Ozet: fps-farkindaligi net, olculebilir bir
iyilestirme (ozellikle MOT17-05'i kurtardigi icin) ama "artik her sekansta
Baseline'i her metrikte geciyoruz" demek hala doğru olmaz.

## Faz 2.9 — Occlusion Recovery Rate: olcum calisti, sonuc mutevazi (2026-08-27)

Kullanicinin orijinal sorusuna dogrudan cevap arandi: **occlusion'dan sonra
ayni kisi ayni tracker ID'siyle mi donuyor?** `evaluation/occlusion_recovery_rate.py`
(Faz 2.7'de yazilmis, hic calistirilmamisti) fps-farkindaligina (Faz 2.8)
guncellendi ve iki sekansta calistirildi. Yontem: bir GT kimlik >=5 kare
boyunca kaybolursa (satir yok VEYA gorunurluk<0.2), hem oncesinde hem
sonrasinda gercekten gorunurse, "occlusion event" sayilir; o an tracker'in
hemen once/sonraki hipotez ID'si IoU>=0.5 ile bulunur; ikisi ayniysa
"kurtarildi". 11 birim testle (`tests/test_occlusion_recovery_rate.py`)
mantik senkron veriyle once dogrulandi.

**Ilk calistirmada MOT17-04'te 28 event'in HICBIRINDE gecerli (once+sonra
bulunabilen) eslesme yoktu -- kod hatasi mi diye ayrica arastirildi.**
Genel kare-seviyesi eslesme orani (`frame_level_match_rate`, search_window'dan
bagimsiz) %34.7 (7445/21451) cikti -- yani eslestirme mekanizmasi GENEL
OLARAK calisiyor, event-sinir karelerinde ozellikle basarisiz. Ornek
event'lerin sinir karelerini tek tek incelemek uc ayri gercek sebep gosterdi:
bazen o karede hic hyp kutusu yok (best IoU=0 -- kismi gorunen kisi YOLO
esigini (0.35) gecmiyor), bazen bir aday var ama esigin (0.5) altinda
(0.12-0.27 arasi kismi ortusme), bazen esigi GECEN bir aday var (0.677) ama
kalabalik sahnede AYNI karedeki BASKA bir GT kimlik o hyp kutusunu greedy
eslestirmede daha yuksek skorla once kapiyor. Ucu de kod hatasi degil,
kalabalik sokak sahnesinde occlusion'dan cikis anlarinin gercekten zor
oldugunu gosteriyor.

`search_window` (sinir karesinden kac kare geri/ileri aranacak) 5/15/30
olarak ayri ayri denendi:

| Sekans | Genel eslesme orani | search_window=5 | search_window=15 | search_window=30 |
|---|---:|---:|---:|---:|
| MOT17-04 (30fps, kalabalik) | %34.7 | 0/28 gecerli | 0/28 gecerli | 1/28 gecerli (ORR=100%, n=1) |
| MOT17-05 (14fps, seyrek) | **%81.6** | 10/22 -> **ORR=%45.5** | 10/24 -> ORR=%41.7 | 10/25 -> ORR=%40.0 |

**MOT17-04 icin ORR pratikte olculemedi** -- 28 occlusion event'inin
neredeyse tamami "gecersiz" (esiz eslesme yok), search_window=30'da (1
saniye) bile sadece 1 ornek var; bunu "%100 kurtarma" diye raporlamak
yaniltici olur (n=1). Durustce: **bu sekansta olcum metodolojisinin kendisi
kalabalik sahnede yeterince guvenilir bir before/after hipotezi bulamiyor**
-- bu, ID-tutarliligindan once gelen, ayri bir sorun (tracker'in kendi
ciktisi occlusion sinirinda birkac kare kararsiz).

**MOT17-05'te olcum calisti ve anlamli bir ornek boyutu verdi (22-25 gecerli
event).** ORR search_window=5'te %45.5, window genisledikce (15 -> %41.7,
30 -> %40.0) hafifce DUSUYOR -- cunku genis pencerede bulunan EK gecerli
event'ler kurtarilmis degil (kurtarilan sayisi hep 10'da sabit kaliyor,
sadece payda buyuyor). Bu onemli: pencereyi genisletmek sonucu YUKARI degil
asagi cekiyor, yani "daha iyi gorunsun diye pencereyi genislettik" itirazi
gecersiz.

**Faz 2.8'in MOTA sayilariyla birlikte okununca:** MOT17-05'te fps-fix
sonrasi MOTA %8.1 -> %10.1'e cikti (bkz. "Faz 2.8"), ama ID-switch hala
yuksek (50 -> 65) ve simdi olculen ORR de yalnizca **%40-46** -- yani bu
sekansta gercek bir occlusion'dan sonra kisinin ayni ID'yi koruma sansi
YARIDAN AZ. **Kullanicinin video'da gozlemledigi "direk arkasindan yeni
ID'yle cikma" sikayeti, tek tek bulunup duzeltilen 2-3 hata disinda, hala
sistemin GENEL bir zayifligi -- Faz 2.1/2.4/2.5/2.8 hepsi bunu iyilestirdi
ama "cozuldu" demek durust olmaz.**

Yol haritasindaki acik maddeler (MOT17-05'te hala yuksek ID-switch,
agregat ID-switch'in Baseline'in gerisinde kalmasi) bu sonucla tutarli --
ayni kok nedenin (kalabalik/occlusion sinirinda kirilgan eslesme) farkli
metriklerdeki yansimalari.

## Faz 2.10 — "association failure" hipotezi test edildi ve ELENDI (2026-08-27)

Faz 2.9'un teshisinde bulunan 5 ornekten 1'i ("kalabalik sahnede iyi bir aday
baska bir GT kimlik tarafindan greedy eslestirmede once kapiliyor") ile
digerlerini ("hic tespit yok" veya "esik alti kismi ortusme") ayirt etmek
icin **tek, izole degisken** test edildi: `evaluation/occlusion_recovery_rate.py`'nin
`_match_gt_to_hyp()` fonksiyonu greedy'den Hungarian'a (`scipy.optimize.
linear_sum_assignment`) gecirildi -- `evaluation/mot_metrics.py`'nin (motmetrics)
MOTA/IDF1 icin zaten kullandigi prensiple tutarli hale getirildi. **Onemli
ayrim:** bu, tracker'in KENDI canli atamasini degistirmiyor -- `tracking/
adaptive_tracker.py` ve `tracking/bytetrack_wrapper.py` zaten bu oturumdan
ONCE beri Hungarian kullaniyordu (`linear_sum_assignment`, satir 294 ve 70).
Degisen sadece ORR'u OLCEN degerlendirme betiginin GT<->hipotez eslestirmesi.

13 birim testle (`tests/test_occlusion_recovery_rate.py`) once senkron veride
doğrulandı -- ozellikle `test_hungarian_matching_recovers_candidate_greedy_would_have_stolen`,
Faz 2.9'daki gercek deseni elle kurup Hungarian'in greedy'nin kacirdigi
gecerli eslesmeyi bulduğunu kanıtlıyor. Sonra MOT17-04 gercek verisiyle
yeniden olculdu:

| Metrik | Greedy (Faz 2.9) | Hungarian (Faz 2.10) |
|---|---:|---:|
| Toplam occlusion event | 28 (23 pencere-ici + 5 pencere-disi) | 28 (ayni -- GT gorunurluk zaman cizelgesine bagli, eslestirmeden bagimsiz) |
| Gecerli once/sonra cifti (search_window=5) | 0/23 | 0/23 |
| Gecerli once/sonra cifti (search_window=15) | 0/23 | 0/23 |
| Gecerli once/sonra cifti (search_window=30) | 1/23 | 1/23 |
| ORR (search_window=30, n=1) | %100 (raporlanamaz) | %100 (raporlanamaz) |
| Genel kare-seviyesi eslesme orani | %34.7 (7445/21451) | **%34.7 (7445/21451) -- bit-bit ayni** |

**Sonuc: SIFIR olculebilir degisiklik.** Kullanicinin onceden kaydedilmis
hipotezi ("kucuk bir artis olabilir ama dramatik cozmemeli") dogrulanmadi --
gercekte beklenenden de netti, hicbir sayı degismedi. Bu, MOT17-04'un
frame_level_match_rate'inde binlerce (kare,gt) ciftinden gecen genel toplamda
bile hicbir fark yaratmiyor; sinir-karesi ornekleri de dahil. **Association
failure hipotezi deneysel olarak elendi: tracker zaten Hungarian kullaniyordu,
degerlendirme katmanini da Hungarian yapmak MOT17-04'un olculebilirlik
sorununu hic etkilemedi.** Faz 2.9'daki 5 orneklik teshis genellemesi dogru
cikti: baskın sebep atama algoritmasi degil, **detection recall/IoU hassasiyeti**
-- occlusion sinirinda YOLO bazen kismen gorunen kisiyi hic tespit etmiyor,
bazen tespit ediyor ama kutu GT ile yeterince ortusmuyor. Bu, atama
algoritmasi (greedy VEYA Hungarian, farketmiyor) tarafindan duzeltilemeyecek
bir sinir.

**Sonraki adim icin anlami:** "Faz 3" yatirimi association-katmanina
(state machine, farkli eslestirme) degil, **occlusion sinirindaki detection
guvenilirligine** yonelmeli -- ornegin YOLO conf_threshold sweep'i, veya
kalabalikta ortusen kisileri bastiran NMS'in occlusion-sinirindaki etkisinin
olculmesi. Bu oturumda henuz UYGULANMADI, sadece bir onceki hipotezin
elenmesiyle yon netlesti.

## Faz 3.0 — YOLO conf_threshold sweep: recall arttı, ORR-olculebilirligi degismedi (2026-08-27)

Faz 2.10'un isaret ettigi yonde, kullanicinin spesifik istegiyle
`evaluation/run_confidence_sweep.py` yazildi: MOT17-04'te 5 conf_threshold
degeri (0.35/0.30/0.25/0.20/0.15) denendi, **SADECE bu degisti** -- ReID
checkpoint, fps-farkinda mesafe butcesi, embedding gallery, LostTrack
penceresi, Hungarian atama, NMS IoU esigi hep sabit (Faz 2.7-2.10'daki
"en iyi" -- condition D -- konfigurasyonu).

**Yazarken ayri, onemli bir bulgu ortaya cikti:** `AdaptiveTracker.
track_thresh` (varsayilan 0.5) `update()` icinde YOLO'nun kendi
`conf_threshold`'undan BAGIMSIZ, IKINCI bir guven esigi uyguluyormus.
Faz 2.x'in TUM onceki scriptleri bunu hic override etmedi -- yani
YOLO conf_threshold=0.35 olarak ayarlanmis olsa bile, tracker 0.35-0.5
arasindaki her tespiti SESSIZCE atiyordu; efektif esik hep 0.5'ti. Bu
sweep'in 0.5 altindaki degerleri gercekten test edebilmesi icin
`track_thresh=0.0` set edildi. (Sweep'in conf=0.35 satirinin MOTA'sinin
-- %29.6 -- Faz 2.8'in raporladigi %29.4'e yakin cikmasi kismen
tesaduf: 0.35-0.5 araligindaki ek tespitlerin katkisi/gurultusu bu
sekansta birbirini büyük ölçüde götürmüş.)

**Performans:** 5 kez YOLO+OSNet calistirmak yerine, en dusuk esikte
(0.15) BIR KEZ tespit+embedding cikarilip daha yuksek esikler o tek
pass'in ciktisi filtrelenerek elde edildi. Bu bir yaklasiklik degil,
matematiksel olarak esdeger: Ultralytics conf-filtreyi NMS'ten ONCE
uyguluyor, standart greedy NMS'te bir kutu sadece kendinden YUKSEK
skorlu bir kutu tarafindan bastirilabilir -- yani X esiginde hayatta
kalacak her kutu, 0.15'teki NMS'ten de ayni sekilde sag cikar (bkz.
`run_confidence_sweep.py` docstring'i icin tam ispat). 5 gercek karede
(baslangic/orta/occlusion-bolgesi/son) fresh-run ile post-filtre
birebir ayni sonucu verdigi ayrica dogrulandi. 3 yeni birim test
(`tests/test_confidence_sweep.py`).

| Conf | Frame Match% | Valid ORR Ev. | ORR | MOTA | IDF1 | IDSW | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 (baseline) | 41.6% | 1/23 | %100 | 0.296 | 0.456 | 31 | 2595 | 12468 |
| 0.30 | 44.8% | 1/23 | %100 | **0.310** | 0.501 | **27** | 2992 | 11779 |
| 0.25 | 47.5% | 1/23 | %0 | 0.297 | **0.508** | 34 | 3860 | 11192 |
| 0.20 | 49.0% | 1/23 | %0 | 0.260 | 0.481 | 47 | 4997 | 10840 |
| 0.15 | 51.5% | 1/23 | %0 | 0.181 | 0.439 | 82 | 7205 | **10284** |

**Iki ayri, kismen celisen sonuc var -- ikisi de dogru, karistirilmamali:**

1. **MOT metrikleri kullanicinin hipoteziyle uyumlu bir "sweet spot" gosteriyor
   (~0.25-0.30).** IDF1 0.25'te tepe yapiyor (%50.8), MOTA 0.30'da (%31.0).
   0.25'in altinda cöküs baslıyor: IDSW 27'den 82'ye (3x), FP 2992'den
   7205'e (2.4x) firliyor, MOTA %31.0'dan %18.1'e cakiliyor. Kullanicinin
   onceden yazdigi egri tarifiyle ("Detection Recall ↑, ama bir noktadan
   sonra FP ↑↑, IDSW ↑, MOTA ↓") birebir eslesiyor.

2. **Ama asil arastirma sorusu -- occlusion sinirinin ÖLÇÜLEBİLİRLİĞİ --
   conf_threshold'a HİÇ tepki vermedi.** Genel kare-seviyesi eslesme orani
   %41.6'dan %51.5'e (+10 puan) net bir sekilde yukselirken, "gecerli
   once/sonra cifti" 23 event'in TAMAMINDA sabit kaldi: 1/23, degismedi.
   Yani genel sahnede COK daha fazla insan tespit edilir hale geldi, ama
   Faz 2.9/2.10'da tespit edilen 22 olculemeyen event hala olculemiyor --
   conf_threshold'u 0.35'ten 0.15'e (genis bir aralik) indirmek bile bu
   spesifik 22 event'e tek bir tane bile eklemedi. **Bu, "detection recall
   dusuk oldugu icin olculemiyor" aciklamasini da buyuk olcude zayiflatiyor**
   -- eger basit recall sorunu olsaydi, esik bu kadar genis bir aralikta
   degismesine ragmen hicbir yeni event'in olculebilir hale gelmemesi
   beklenmezdi. Daha olasi aciklama: bu 22 event'teki kisiler o denli
   agir/tam occlude oluyor ki (veya kutu GT ile o kadar kotu ortusuyor ki)
   hicbir makul guven esigi yardimci olmuyor -- recall degil, lokalizasyon/
   siddet sorunu.

3. **Tek olculebilir event (n=1) bile conf dustukce KOTULESTI:** 0.35/0.30'da
   kurtarilmis (ORR=%100), 0.25 ve altinda kurtarilmamis hale geldi
   (ORR=%0) -- daha dusuk esikte gelen ekstra (muhtemelen gurultulu)
   tespitler, tam da tutarliligini olcebildigimiz TEK ornekte kimlik
   karisikligina katkida bulunmus olabilir (IDSW egrisiyle tutarli).

**Sonuc: conf_threshold sweep, MOT metrikleri icin gercek ve kullanisli bir
sweet-spot (~0.25-0.30) buldu, ama Faz 2.9'un ana sorusunu (occlusion
sinirinda ORR neden olculemiyor) COZMEDI -- sadece "basit recall sorunu"
hipotezini de byuk olcude zayiflatti.** Kullanicinin talep ettigi disiplin
geregi baska hicbir sey degistirilmedi; bu net bir sonuc, hosumuza gitmese
de saklanmiyor. Sonraki adim muhtemelen NMS/crowded-scene analizi degil,
bu 22 event'in kendisini (kutu boyutlari, gorunurluk yuzdeleri, kalabalik
yogunlugu) tek tek incelemek olmali -- "esik ayari" tipi bir cozumun bu
sorunu cozmesi beklenmiyor artik.

## Faz 3.1 — Occlusion Event Forensics: kok neden bulundu, tracker/detector DEGIL (2026-08-27)

Kullanicinin onerdigi taksonomiyle (`evaluation/occlusion_event_forensics.py`),
MOT17-04'un 23 pencere-ici event'inin HER BIRI tek tek incelendi. Her event
icin once/sonra sinir karesinde: gorunurluk, GT kutu alani, kalabalik sayisi,
en-iyi-tespit IoU'su (`conf>=0.15`, tam sekans taramasi degil sadece 44
benzersiz sinir karesinde -- hizli), ve o TAM karede Hungarian GT<->hyp
eslesmesi olup olmadigi olculdu. Basarisiz taraflar E (belirsiz-gorunurluk
sinir karesi, vis<0.3) > A (hic tespit yok) > B (dusuk IoU) > C (iyi IoU var
ama baska kimlige kapilmis) onceligiyle, basarili event'ler D (ID-switch)
veya RECOVERED olarak etiketlendi. 16 birim test (`tests/test_occlusion_event_forensics.py`).

**Yazip ilk calistirdiktan sonra KENDI script'imde gercek bir hata bulundu
ve duzeltildi -- bu da raporlanmasi gereken bir sey:** ilk versiyon, bir
tarafin basarili/basarisiz oldugunu SADECE o TAM sinir karesindeki dar
tanidan (`reason_before`/`after`) belirliyordu, `compute_orr`'un asil
arama-penceresi (search_window=30) ile bulmus oldugu gercek sonucu
(`before_hyp`/`after_hyp`) once kontrol etmeden. Sonuc: Faz 2.9/2.10'da
zaten bilinen TEK gecerli+kurtarilmis event (gt_id=134, kare 381-400) bile
yanlislikla "E" etiketi almisti, cunku tam sinir karesinde gorunurluk
belirsizdi -- ama gercekte arama penceresi birkac kare otede gecerli bir
eslesme BULMUSTU. Duzeltme: basari/basarisizlik ONCE `before_hyp`/`after_hyp`'e
(otoriter, arama-penceresi-farkinda sonuc) gore belirleniyor; sinir-karesi
tanisi SADECE o taraf GERCEKTEN basarisizsa devreye giriyor. Duzeltmeden
sonra 134 dogru sekilde "RECOVERED" cikti -- yeni bir regresyon testiyle
(`test_classify_event_search_window_rescue_is_not_mislabeled_by_exact_frame_reason`)
kilitlendi. Script, pahali (13-15 dk) tespit+takip kismini tekrar
calistirmadan siniflandirma mantigini duzeltip yeniden denemeyi mumkun kilan
bir cache (`evaluation/_cache/`, gitignore'lu) da kaydediyor.

**Sonuc, cok net ve tek bir aciklamaya isaret ediyor:**

| Etiket | Sayi |
|---|---:|
| E (belirsiz-gorunurluk sinir karesi) | **22 / 23** |
| RECOVERED | 1 / 23 |

**23 event'in 22'sinde de -- ISTISNASIZ -- once/sonra sinir karesindeki
gorunurluk 0.20 ile 0.23 arasinda, yani vis>=0.2 esiginin HEMEN ustunde.**
Bu rastlantisal degil: "sinir karesi" tanimi zaten "occlusion baslamadan
ONCEKI SON gorunur kare" / "occlusion bittikten sonraki ILK gorunur kare" --
gorunurluk kare-kareye YUMUSAK degistigi icin, bir esigi asagi/yukari GECEN
ilk/son kare, MATEMATIKSEL OLARAK o esigin hemen yaninda olmak zorunda.
`VIS_MARGIN_THRESH=0.3` esigi veriye BAKMADAN once sectirildi (0.2 taban
cizgisinin ustunde, "gercekten guvenli gorunur" saymak icin makul bir pay)
-- ve gozlenen degerlerin TAMAMI (max 0.23) zaten bunun cok altinda kaldi,
yani esik veriye gore ayarlanmadi, veri esigi rahatlikla gecti.

**Bu, Faz 3.0'in bulgusunu tam olarak aciklıyor.** conf_threshold'u
0.35'ten 0.15'e indirmek hicbir event'i kurtarmadi, cunku sorun detector'in
o kisiyi YETERINCE guvenle tespit edememesi degildi -- sorun, ORR'un
"sinir karesi" olarak SECTIGI anin, tanimi geregi zaten en-belirsiz-gorunum
anı olmasi. Hicbir guven esigi bunu duzeltemez, cunku mesele detector'in
performansi degil, HANGI KARENIN referans alindigi. **Association algoritmasi
(Faz 2.10) degil, detection recall (Faz 3.0) degil -- kok neden ORR
metriginin sinir-karesi TANIMI.**

**Sonraki adim (bu bolumun devaminda):** metriğin TANIMINI degistirmeden
once, kullanicinin onerisiyle, ayni script AYNI parametrelerle (VIS_MARGIN_
THRESH, arama penceresi, taksonomi -- HICBIRI degistirilmeden) MOT17-05'te
de calistirildi -- "bu genel bir ORR sorunu mu, yoksa MOT17-04'e ozgu mu?"
sorusunu bagimsiz bir sekansla test etmek icin. Once metrigi "sonuca gore"
ayarlamak yanlis metodoloji olurdu.

### Faz 3.1 replikasyonu — 7 sekansin TUMU: E-orani kalabalikla GUCLU korele

Once MOT17-05 (kod/esik/taksonomi MOT17-04'tekiyle BIREBIR AYNI), sonra
kullanicinin istegiyle kalan 5 sekans da (02, 09, 10, 11, 13) ayni sekilde
tarandi -- MOT17'nin FRCNN varyantindaki 7 sekansin TUMU artik forensics'ten
gecti:

| Sekans | Event | RECOVERED | E | A | B | C | D | Gecerli | ORR | E-orani | Ort. kalabalik |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 54 | 9 | 32 | 2 | 9 | 0 | 2 | 11 | %81.8 | %59.3 | 13.5 |
| MOT17-04 | 23 | 1 | 22 | 0 | 0 | 0 | 0 | 1 | %100 (n=1) | **%95.7** | **41.0** |
| MOT17-05 | 31 | 10 | 2 | 1 | 3 | 0 | 15 | 25 | %40.0 | **%6.5** | **6.2** |
| MOT17-09 | 13 | 7 | 2 | 0 | 1 | 0 | 3 | 10 | %70.0 | %15.4 | 7.2 |
| MOT17-10 | 29 | 2 | 18 | 0 | 3 | 1 | 5 | 7 | %28.6 | %62.1 | 15.0 |
| MOT17-11 | 19 | 5 | 5 | 0 | 1 | 0 | 8 | 13 | %38.5 | %26.3 | 6.1 |
| MOT17-13 | 12 | 3 | 5 | 0 | 0 | 1 | 3 | 6 | %50.0 | %41.7 | 15.7 |

(E-etiketli taraflarin gorunurluk araligi TUM 7 sekansta da 0.20-0.29
civarinda -- kok neden hepsinde ayni, sadece SIKLIGI degisiyor. "Ort.
kalabalik" = event'lerin once/sonra sinir karelerinde, ayni anda gorunur
BASKA GT kimlik sayisinin ortalamasi, `crowd_before`/`crowd_after`'dan.)

**Bu bir varsayim degil, olculmus bir korelasyon: E-orani ile ortalama
kalabalik arasinda Pearson r = 0.892 (n=7).** En dusuk-kalabalik 3 sekans
(05, 11, 09 -- ortalama 6.1-7.2 kisi) hepsi dusuk E-orani (%6.5-%26.3); en
yuksek-kalabalik sekans (04 -- 41.0, digerlerinin ~3 kati) acik ara en
yuksek E-orani (%95.7). Orta-kalabalik sekanslar (02, 10, 13 -- 13.5-15.7)
orta-yuksek E-orani (%41.7-%62.1) gosteriyor. Mukemmel monoton degil
(09 vs 11 sirasi ufak bir istisna) ama n=7 icin cok guclu ve tutarli bir
sinyal.

**Sonuc: sinir-belirsizligi sorunu GENEL bir ORR problemi DEGIL, sahne
YOGUNLUGUYLE olculebilir sekilde korele.** MOT17-04, 7 sekans arasinda
kalabalik acisindan gercek bir DIS DEGER (outlier) -- digerlerinin en az
2.6 kati. ORR degerleri de guvenilirlik acisindan boylece siralanabilir:
**05, 09, 11 (E<%30, gecerli ornek 10-25) en guvenilir**; 02, 13 (E
%42-%59, gecerli ornek 6-11) orta guvenilir; **10 (E %62, gecerli sadece
7) ve ozellikle 04 (E %96, gecerli sadece 1) neredeyse olcum-disi.**

**Anlami:** ORR metriginin TANIMI genel olarak bozuk degil -- yogunluk
arttikca gorunurluk esik civarinda o kadar sik/kaotik dalgalaniyor ki
"sinir karesi" secimi giderek daha sik belirsiz bir ana denk geliyor,
olcumu SISTEMATIK olarak (rastgele degil) bozuyor. **Faz 3.2 (sinir
tanimini vis>=0.5'e gore degistirmek) bu yuzden GENEL bir duzeltme olarak
hala oncelikli degil** -- 5 sekansta (05,09,10,11,13) zaten calisiyor,
sadece asiri-yogun (04) ve yuksek-yogun (02'ye yakin 10) sekanslarda
faydali olurdu. Kullanicinin onceden belirttigi risk ("sonuc vermedi ->
esigi degistir -> duzeldi gibi gorunur") boylece 6 bagimsiz replikasyonla
(sadece 1 degil) ELENDI -- metrik degistirilmeden once, HANGI kosullarda
anlamli oldugu once, sayisal olarak anlasilmis oldu.

## Faz 3.2 — kullanicinin ekran goruntusu dogrulandi: "orta siddette surekli kismi gorunurluk" YENI bir hata modu (2026-08-31)

Kullanicinin gerçek demo videosunda ("bu proje çalışıyor mu" sorusuna cevaben
uretilen, `config_demo.yaml`, guncel en iyi pipeline) verdigi iki ekran
goruntusu ("5. saniye" ve "25. saniye", ayni kisi farkli ID) tahminle degil,
GT zemininde dogrulandi. `evaluation/trace_track_lifecycle.py` (Hungarian
GT<->hyp eslestirmesi, HER karede) + `evaluation/snapshot_frames.py`
(tam overlay'li PNG, kullanicinin ekran goruntusuyle piksel-piksel
karsilastirma icin) yazildi. 9 birim test.

**Onemli, ayri bir duzeltme:** ekran goruntusundeki "P=0.04" degeri tracker
guveni DEGIL -- `decision/priority_engine.py`'nin "hassas bolge" uyari
onceligi (hiz/bolge/bekleme agirlikli, `viz/overlay_renderer.py:53`). Gercek
tracking-guven sinyali "[uncertain]" etiketi (`track_confidence` -- det_conf/
motion/reid agirlikli kombinasyon -- 0.5-0.75 araliginda).

**Bulgu (kesin, tahmin degil):** iki ekran goruntusu de AYNI GT kimlige
(gt_id=3) ait -- 100. karede hyp=7, 500. karede hyp=**40**, GT konumu
(`crop_gt3_frame500.png`) ile piksel-piksel dogrulandi. Tam hyp gecmisi:
`7 (kare 1-292, kucuk bir bosluk haric) -> BOSLUK (kare 293-477, 184 kare)
-> 40 (kare 478-524)`.

**Asil onemli kisim:** bu 184 karelik bosluk boyunca GT gorunurlugu
**HICBIR ZAMAN 0.2'nin ALTINA DUSMEDI** (minimum ~0.34, kare 280-475
arasinda surekli 0.33-0.45 araliginda) -- yani Faz 2.9-3.1'in TUM
metodolojisinin (`occlusion_recovery_rate.py`, `occlusion_event_forensics.py`)
"occlusion event" tanimina (vis<0.2) gore bu GENÇ **hic bir occlusion
event'i olarak bile sayilmiyor**. Kisi direge yakin durup ORTA siddette,
SURDURULEBILIR sekilde kismen gorunur kaliyor (~6 saniye gercek zaman) --
ne tam kaybolmus (Faz 2.5'in "7/46, 9 saniyelik gercek occlusion, tasarim
disi" durumu GIBI degil) ne de tam gorunur. Tracker bu orta-siddetli
durumda IoU>=0.5 eslesmeyi 184 kare boyunca (60 karelik `reid_recovery_window`'un
3 kati) kaybediyor.

**Anlami:** bu, Faz 2.9-3.1'in olcemedigi/kapsamadigi YENI bir hata modu --
"agir occlusion" ile "hic occlusion yok" arasindaki bosluk. Simdiki ORR/
forensics metodolojisi SADECE vis<0.2 olaylarini event sayiyor; bu tur
"surekli orta-kismi-gorunurluk" durumlarini hic gormuyor. Sonraki adim
(henuz uygulanmadi, kullanicinin karari bekleniyor): occlusion event
tanimini vis<0.2'den genisletip orta-araliktaki (orn. 0.2-0.5) uzun
surekli-dusuk-gorunurluk pencerelerini de "event" sayan ayri bir analiz --
bu, daha once bahsedilen "sinir karesini vis>=0.5'e gore yeniden tanimlama"
fikrinden FARKLI bir iş (o boundary-frame SECIMIYLE ilgiliydi, bu ise
occlusion event'in TANIMININ kendisiyle ilgili).

## Faz 3.3 — Partial Visibility Persistence Forensics: 7 sekansta yayginlik olculdu (2026-08-31)

Kullanicinin acik istegiyle -- **ORR'un tanimini SESSIZCE genisletmeden**,
ayri bir script/metrik olarak -- `evaluation/partial_visibility_forensics.py`
yazildi. PARTIAL_VISIBILITY event = bir GT kimligin ARDISIK >=60 kare
(reid_recovery_window ile ayni) boyunca 0.20<=vis<0.50 kalmasi. Her episode
RETAINED/RECOVERED/FRAGMENTED/PERMANENT_LOSS'a ayrildi; basarisiz olanlarin
kayip-baslangic karesi P1-P4'e (NO_DETECTION/LOW_IOU/TRACK_THRESH_GATE/
ASSIGNMENT_CONFLICT) siniflandirildi. 17 birim test -- biri, ilk versiyonda
gercek bir hata buldu (`find_loss_transition_frame` bosluksuz ani ID
degisimlerini kacirip yanlislikla None donduruyordu; duzeltildi).

7 sekansin TUMUNDE calistirildi (kullanicinin "once yayginligi olcelim"
talebi):

| Sekans | Episode | RETAINED | RECOVERED | FRAGMENTED | PERM._LOSS | PVTR | Frag. Orani |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 1 | 0 | 0 | 0 | 1 | %0.0 | %0.0 |
| MOT17-04 | **38** | 0 | 3 | 2 | 33 | %7.9 | %5.3 |
| MOT17-05 | 1 | 0 | 0 | 0 | 1 | %0.0 | %0.0 |
| MOT17-09 | 2 | 0 | 1 | 1 | 0 | %50.0 | %50.0 |
| MOT17-10 | 1 | 0 | 0 | 1 | 0 | %0.0 | %100.0 |
| MOT17-11 | 0 | - | - | - | - | - | - |
| MOT17-13 | 0 | - | - | - | - | - | - |
| **TOPLAM** | **43** | **0** | **4** | **4** | **35** | **%9.3** | **%9.3** |

**Bulgu: bu hata modu MOT17-04'e YOGUN sekilde ozgu.** 43 episode'un
**%88.4'u (38/43) tek basina MOT17-04'te.** Faz 3.1'in "mean_crowd"
olcusuyle (ayni sekans-basi ortalama kalabalik sayisi) korelasyon
**r=0.935 (n=7)** -- Faz 3.1'in E-orani korelasyonundan (r=0.892) bile
daha guclu. MOT17-11 ve MOT17-13'te bu hata modu HIC gozlenmiyor (0
episode). Yani bu, Faz 3.1'in kok-neden bulgusuyla (sinir-belirsizligi de
yogunlukla korele) AYNI dogrultuda: **AutoTrack'in tracker/detector
zayifliklari genel olarak degil, ozellikle asiri-yogun sahnelerde
(MOT17-04 gibi) yoğunlaşiyor.**

**Carpici, ayrı bir gozlem: 43 episode'un HICBIRI (%0) RETAINED degil** --
yani 60+ karelik hicbir surekli-kismi-gorunurluk penceresinde tracker
BOSLUKSUZ takip saglayamamis. Bu, "kismen gorunen bir kisiyi 2+ saniye
kesintisiz izlemek" gercekten zor bir bar oldugunu gosteriyor. Agregat
P1-P4 dagilimi (39 basarisiz episode'un tumunde): **P1 (hic tespit yok)
%48.7 (19/39)** en baskin, sonra P2 (dusuk IoU) %33.3 (13/39), P3
(tracker'in kendi guven esigi reddediyor) %12.8 (5/39), P4 (baska kimlige
kapilma) %5.1 (2/39) -- Faz 3.0/3.1'in bulgusuyla tutarli: asil darbogaz
detection recall/lokalizasyon, association algoritmasi degil.

**Durustce belirtilmesi gereken bir sinir-etkisi:** Faz 3.2'yi baslatan
gt_id=3 vakasi (MOT17-04, hyp 7->40) burada **RECOVERED** olarak
etiketlendi, FRAGMENTED degil -- cunku 278-464 penceresinde (episode
sinirlari) hyp SADECE 7'ye eslesiyor (sonra surekli None); hyp=40'a
GERCEK gecis frame 478'de, yani vis 0.5'i asip "normal gorunur" rejimine
GIRDIKTEN SONRA oluyor. Kok-neden teshisi (orta-siddette surekli kismi
gorunurluk sirasinda kaybolma) hala dogru, ama "FRAGMENTED" etiketi
episode'un KENDI sinirlarina gore hesaplandigi icin bu spesifik ornek
farkli bir kutuya dusuyor -- bu bir hata degil, sinir-tabanli tanimin
dogal bir sonucu, ama gizlenmemesi gereken bir nuans.

**Sonraki adim kullanicinin karari:** yayginlik artik olculdu (MOT17-04'e
yogun ozgu, diger sekanslarda nadir/yok). Yeni bir tracker fazina gerek
olup olmadigina (ornegin partial-visibility rejiminde daha agresif bir
Re-ID/motion agirlik kaymasi) bu veriyle karar verilebilir -- bu oturumda
UYGULANMADI.

## Faz 3.4 — kucuk prototip: galeriyi PRIMARY eslesmede de kullanmak, SIFIR fark yaratti (2026-08-31/09-01)

Faz 3.3'un actigi soruya kullanicinin istegiyle **kucuk, izole bir
prototip** denendi -- `tracking/adaptive_tracker.py`'ye (AutoTrack'in asil
kodu) HIC DOKUNMADAN. Somut gozlem: `_reid_score` (PRIMARY cost matrix,
her karede) SADECE `track.embedding`'i (tek EMA-blended gorunum)
kullaniyor; `embedding_gallery` (Faz 2.5) SADECE `_try_revive`'de (kayip
kurtarma) kontrol ediliyor, aktif takip sirasinda HIC degil. Hipotez:
galeriyi PRIMARY eslesmede de (EMA ile galeri-en-iyi-benzerliginin
MAKSIMUMU) kullanmak, partial-visibility sirasinda gorunum degisince
yardimci olabilir.

`evaluation/prototype_gallery_reid.py`: `AdaptiveTracker`'in bir ALT
SINIFI (`GalleryAwareTracker`), SADECE `_reid_score`'u override eder, geri
kalan HER SEY birebir ayni. 6 birim test. MOT17-04'un 38 episode'unda
baseline'a karsi calistirildi:

| Metrik | Baseline | Prototip |
|---|---:|---:|
| RETAINED / RECOVERED / FRAGMENTED / PERMANENT_LOSS | 0/3/2/33 | 0/3/2/33 |
| PVTR | %7.9 | %7.9 |
| Fragmentation Rate | %5.3 | %5.3 |

**Sonuc: BIT-BIT AYNI -- hicbir episode'un sonucu degismedi.** Bu net,
temiz bir negatif sonuc, saklanmiyor. Onceden beklenen bir sinir vardi:
basarisiz 35 episode'un P1(19)+P3(4)=23'u (%65.7) bu fikirle PRENSIPTE
duzelemezdi (P1: hic tespit yok, cost matrix'e giren bir sey yok; P3:
tespit `track_thresh` esiginde -- cost matrix'e/ReID skoruna HIC
ULASMADAN -- zaten eleniyor). Ama geriye kalan P2(11)+P4(1)=12 (%34.3,
"teorik olarak ulasilabilir") icin de SIFIR degisiklik olmasi ayrica
bilgilendirici: **muhtemel aciklama, `_cost_matrix`'teki mesafe siniri
(`max_dist`) adaylari ReID skoru hesaplanmadan ONCE eliyor** -- yani bu
durumlarda darbogaz ReID'nin ayirt edicilik gucu degil, mesafe kapisinin
kendisi (bu hipotez bu oturumda AYRICA test EDILMEDI, sadece sonuctan
cikarilan en olasi aciklama).

**Anlami:** "galeriyi primary eslesmede kullan" fikri, en azindan bu
haliyle, AutoTrack'e tasinmaya deger DEGIL -- olcumlenebilir hicbir
katkisi yok. Gercek darbogaz (P1 detection recall + muhtemelen mesafe
kapisi) baska yerde. Bu, kullanicinin sordugu "denemeye deger mi"
sorusunu net bir "hayir, bu haliyle degil" ile cevapliyor -- ve bunu
gercek koda dokunmadan, ucuz bir prototiple ogrenmis olduk.

## Faz 3.5 — mesafe siniri de darbogaz DEGIL: tracker-tarafi mudahaleler tukendi (2026-09-15)

Faz 3.4'un actik biraktigi soru -- "_reid_score degistirmenin sifir etkisi,
max_dist mesafe sinirinin adaylari ReID hesaplanmadan once elemesinden mi
kaynaklaniyor?" -- bu oturumda DOGRUDAN test edildi (Faz 3.4'te sadece bir
cikarimdi, ayrica dogrulanmamisti).

`evaluation/test_distance_gate_hypothesis.py`: ayni 38 MOT17-04 partial-
visibility episode'u, `association_dist_ratio` icin iki degerle yeniden
siniflandirildi -- 0.08 (guncel, Faz 2.4'un sikilastirdigi) ve 0.15
(Faz 2.4 ONCESI, Faz 2.1'in ilk koydugu sinir). `tracking/adaptive_tracker.py`'ye
HICBIR DOKUNUS yok, parametre zaten disaridan geciriliyordu.

**Sonuc: 0/38 episode outcome degisti.** Hicbiri -- P2/P4 (teorik olarak
mesafe-sinirindan kurtarilabilir sayilan 12 episode) dahil. Faz 3.4'un
ReID-skoru degisikligiyle (bit-bit sifir) tam olarak ayni kesinlikte bir
negatif sonuc.

**Ustune, riskli taraf da dogrulandi:** 0.15'e genisletmek sekanstaki en
buyuk ardisik-kare merkez sicramasini 78px'ten **307px'e** cikardi (track
21, kare 478) -- Faz 2.4'un duzelttigi turden (309px, gercek dogrulanmis
bir yanlis-birlestirme) bir sicrama yeniden ortaya cikti. Yani 0.15 hem
hicbir episode'u kurtarmiyor hem de bilinen bir hata sinifini geri
getiriyor -- bu oturumun en net "hayir" sonuclarindan biri, cift yonlu.

**Anlami:** tracker-tarafi mudahaleler artik ucu de test edildi ve ucu de
negatif -- ReID skoru (Faz 3.4), mesafe siniri (bu faz), confidence esigi
(Faz 3.0, farkli bir bulgu icin ama ayni sonuc: hicbir sey degismedi).
Partial-visibility episode'lerinin geri kalaninda (agirlikli olarak
P1_NO_DETECTION) gercek darbogaz artik buyuk olasilikla saf detection
recall/lokalizasyon -- tracker'in kendi mantigiyla duzeltilebilecek bir
sey degil. 5 yeni birim test (`_max_consecutive_jump` -- Faz 2.4/2.6'nin
o zaman tek seferlik/tests-disi olan sicrama taramasinin artik tekrar
kullanilabilir hali). 144 test yesil.

## Faz 4.0 — Hava/Drone Genelleme Doğrulaması: detector'un kendisi çöktü (2026-09-05)

Kullanıcının "kapsamı genişletelim" isteğiyle -- README'nin en basindan
beri iddia ettigi ama bu proje boyunca (Faz 1-3.4) HİÇ test edilmedi
"drone kamerasi" senaryosu ilk kez gerçek veriyle sınandı.
**VisDrone-MOT** (Hugging Face `Voxel51/visdrone-mot` mirror'i -- gerçek
VisDrone sekans adlandırması, `pedestrian_density` etiketi, gerçek
`index` (track ID) alanı doğrulandı) -- yoğunluk ekseninde en net
kontrastı veren 2 sahne seçildi:

| Sahne | Yoğunluk | Kare | GT nesne |
|---|---|---:|---:|
| `uav0000305_00000_v` | low | 184 | 602 |
| `uav0000137_00458_v` | high | 233 | 9128 |

`evaluation/visdrone_adapter.py` (yeni) VisDrone'un FiftyOne-native
`samples.json`'ini MOT17 ile BİREBİR AYNI dizin yapısına (`img1/`+`gt/`+
`seqinfo.ini`) çevirir -- 7 birim test. **Dürüstçe belirtilmesi gereken
iki yaklaşıklık:** (1) VisDrone'un `visibility` alanı İKİLİ (0/1),
MOT17'nin sürekli değeri gibi değil -- bu, Faz 3.3'ün kısmi-görünürlük
analizini bu veri kümesinde yapısal olarak imkansız kılıyor; (2)
`frameRate=30`, VisDrone'un dogrulanmis cekim hizi degil, FiftyOne'in
kendi app-config varsayilani. Mevcut `occlusion_recovery_rate.py`/
`occlusion_event_forensics.py`/`partial_visibility_forensics.py`'ye
geriye-uyumlu bir `--data-dir` parametresi eklendi (varsayılan hala
MOT17) -- kod TEKRARLANMADI, aynı script'ler VisDrone'da da çalışır.

**Sonuç (guncel en iyi config, HIÇBIR parametre değişmeden):**

| Sahne | MOTA | IDF1 | IDSW | FP | FN | GT |
|---|---:|---:|---:|---:|---:|---:|
| `uav0000305_00000_v` (low) | **%0.0** | %0.0 | 0 | 0 | **602/602** | 602 |
| `uav0000137_00458_v` (high) | **-%7.4** | %11.4 | 15 | 1416 | 8377/9128 | 9128 |

**Kök neden, tahmin degil dogrudan dogrulandi:** low-density sahnede
FN=GT_obj=602 -- tracker'in bulabildigi SIFIR gercek eslesme var. Bunun
bir pipeline hatasi mi yoksa gercek bir genelleme basarisizligi mi
oldugunu ayirt etmek icin tek bir karede ham YOLO ciktisi (`conf>=0.01`,
TUM siniflar) incelendi: model **baska sınıfları tespit ediyor** (araba,
kamyon -- yanlis siniflandirilmis olsa da bir sinyal var) ama GT'deki 3
gercek insan kutusunun (30x20, 27x20, 25x24 PIKSEL -- yani 500-600 piksel
alan) hicbirine yakin, "person" etiketli, kullanilabilir-guvenli hicbir
tespit yok; `conf>=0.01` gibi gurultu seviyesinde bir esikte bile
bulunan 10 "person" adayi GT konumlariyla ORTUSMUYOR (en yakini bile
~150px uzakta). **Bu, MOT17'de bulunan hicbir hata moduyla (crowd-
density korelasyonu, partial-visibility) ilgisi olmayan, cok daha temel
bir sorun: `yolov8n.pt` (vanilla COCO agirliklari) bu hava-acisi/olcekte
(~20-30px insan kutulari) "person" sinifini pratikte HIC tespit
edemiyor.** COCO'nun egitim dagilimi (yer-seviyesi/yakin-mesafe
insanlar) ile VisDrone'un ust-tan-asagi, cok kucuk-olcekli
goruntusu arasindaki fark, tracker/Re-ID katmaninin hic devreye
giremeyecegi kadar buyuk.

**Bu yuzden asil arastirma sorusu (MOT17'nin kalabalik-bagimli hata
modlari hava goruntusune genelleniyor mu?) bu turda CEVAPLANAMADI** --
tespit katmani calismadigi icin takip katmaninin davranisi olculemez
(pahali `occlusion_event_forensics.py`/`partial_visibility_forensics.py`
kosulari bu yuzden BİLİNÇLİ OLARAK calistirilmadi -- buyuk ihtimalle
sadece "%100 tespit yok" tekrarlarlardi, yeni bilgi vermeden saatlerce
hesaplama harcardi). **Sonraki adim netlesti ama bu oturumda
UYGULANMADI:** VisDrone'un KENDI egitim-bolumundeki insan kutulariyla
YOLO'yu (veya hazir bir VisDrone-egitilmis checkpoint) fine-tune etmeden,
bu veri kumesinde anlamli bir MOT16/olcum yapilamaz -- karar
kullaniciya ait.

## Faz 4.1 — VisDrone fine-tune: tespit gercekten iyilesti, ama tek sahnede ve yeni bir bedelle (2026-09-15)

Faz 4.0'in "sonraki adim" olarak birakip UYGULAMADIGI seyi bu faz yapti:
yolov8n.pt, VisDrone'un KENDI egitim verisiyle fine-tune edildi -- Faz 4.0'in
degerlendirme sahnelerine (`uav0000137_00458_v`, `uav0000305_00000_v`)
KESINLIKLE dokunulmadan.

**Sizinti onlemi (Faz 2.3'un MOT17-09/11-vs-02/04 ayrimiyla ayni disiplin):**
egitim icin 2 YENI sahne indirildi -- `uav0000086_00000_v` (yuksek yogunluk,
464 kare, 1344x756px) ve `uav0000339_00001_v` (dusuk yogunluk,
275 kare, 1904x1071px), toplam 739 kare / **125.5MB** (onceden tahmin edilen
~800MB'den kucuk cikti). `evaluation/compare_finetune_vs_baseline.py`
egitim/degerlendirme kumeleri arasinda kesisim olmadigini HER CALISTIGINDA
programatik olarak dogruluyor, sadece iddia etmiyor.

**Veri hazirligi (`evaluation/prepare_yolo_finetune_data.py`):** VisDrone'un
MOT-tarzi gt.txt'i Ultralytics YOLO tespit formatina cevrildi -- 629 egitim
+ 110 dogrulama karesi, her sahnenin kendi ZAMAN EKSENINDE (rastgele degil)
bolundu (ardisik kareler neredeyse ozdes oldugu icin rastgele bolme sizinti
yaratirdi). Donusum, gercek bir karede kutulari elle cizip goz ile
dogrulandi (Faz 2'nin egitim/dogrulama disipliniyle ayni ruhta).

**Fine-tune (`evaluation/finetune_yolo_visdrone.py`):** COCO agirliklarindan
DEVAM (sifirdan degil), imgsz=1280 (640 DEGIL -- Faz 4.0'in kok nedeni tam
olarak "insanlar cok kucuk" oldugu icin kucultmemek), dusuk lr0, 30 epoch,
patience=10. **Bir gercek, ortam-kaynakli hata bulunup duzeltildi:**
Ultralytics'in otomatik MLflow entegrasyonu (bu makinede baska bir proje --
muhtemelen Baykar NLP RAG/QLoRA/DPO calismasi -- icin acikken) kurulu mlflow
surumunun dosya-sistemi izlemeyi artik reddetmesiyle CRASH oluyordu; global
Ultralytics ayarina DOKUNULMADAN (diger projeyi etkilemesin diye), sadece bu
surece ozel bir ortam degiskeniyle (`MLFLOW_ALLOW_FILE_STORE`) asildi.
Egitim 20 dakikada tamamlandi (30 epoch, RTX 2080) -- kendi dogrulama
setinde (ayni 2 egitim sahnesinin gormedigi son %15'i) mAP50=%86.2,
recall=%80.1.

**Asil test -- DOKUNULMAMIS 2 degerlendirme sahnesi, `evaluation/
compare_finetune_vs_baseline.py`:**

| Sahne | Config | GT-miss% | Recall@IoU0.5 | Ham tespit | MOTA | IDF1 | IDSW | FP | FN | Det.FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uav0000137_00458_v (yuksek yogunluk) | baseline | %84.9 | %15.1 | 1680 | -0.074 | 0.114 | 15 | 1416 | 8377 | 14.2 |
| uav0000137_00458_v | **fine-tuned** | **%31.1** | **%68.9** | 10427 | -0.253 | 0.401 | **300** | 8215 | 2925 | 10.3 |
| uav0000305_00000_v (dusuk yogunluk) | baseline | %100.0 | %0.0 | 1 | 0.000 | 0.000 | 0 | 0 | 602 | 28.0 |
| uav0000305_00000_v | fine-tuned | %99.8 | %0.2 | 328 | -2.452 | 0.000 | 0 | 1476 | 602 | 23.9 |

(Baseline satirlari Faz 4.0'in kayitli sayilariyla BIREBIR eslesiyor --
MOTA=-%7.4/IDF1=%11.4/FP=1416/FN=8377 -- bu, yeni script'in Faz 4.0'in
metodolojisini dogru tekrarladiginin bagimsiz bir dogrulamasi.)

**Sonuc, tek yonlu bir zafer DEGIL -- durustce iki parcali:**

1. **Yuksek yogunluklu sahnede tespit katmaninda GERCEK, BUYUK bir
   iyilesme:** recall@IoU0.5 %15.1 -> **%68.9** (GT-miss %84.9 -> %31.1).
   Faz 4.0'in "COCO agirliklari bu olcekte insan hic gormuyor" bulgusuna
   DOGRUDAN cevap -- fine-tune gercekten calisiyor, en azindan tespit
   katmaninda, en azindan bu sahnede.
2. **Ama dusuk yogunluklu sahnede NEREDEYSE HIC iyilesme yok** (recall
   %0.0 -> %0.2, 601/602 GT hala kayip) -- fine-tune bu ikinci sahnenin
   kosullarina (muhtemelen farkli irtifa/doku/arkaplan) genellemedi.
3. **Ve yuksek yogunluklu sahnede MOTA/ID-switch KOTULESTI** (IDSW 15 ->
   **300**, FP 1416 -> 8215, MOTA -%7.4 -> -%25.3): fine-tune edilmis
   model COK DAHA FAZLA ham tespit uretiyor (1680 -> 10427) ve bunlarin
   onemli bir kismi gurultu -- asil COCO-egitimli model icin kalibre
   edilmis `conf_threshold=0.35` sabiti, fine-tune edilmis modelin farkli
   guven dagilimina artik uymuyor olabilir (test EDILMEDI, sadece en
   olasi aciklama). Yani "tespit katmani" duzeldi ama "takip katmani"
   simdi farkli, yeni bir sorunla karsi karsiya.
4. **Dusuk yogunluklu sahnede de FP'ler arttigi halde (0 -> 1476) recall
   HALA sifira yakin** -- fine-tune edilmis model bu sahnede gercek
   insanlari degil, baska bir seyi (gurultu/yanlis pozitif) tespit etmeye
   basliyor. Kok nedeni bu oturumda arastirilmadi.

**Anlami:** Faz 4.0'in "detector calismiyor" bulgusu artik kismi olarak
duzeltildi (bir sahnede kanitlanmis, buyuk bir tespit-recall kazanci var),
ama bu hem sahneye-ozel/genellemeyen hem de tracker'da yeni bir maliyetle
geliyor. **"Fine-tune ettik, sorun cozuldu" demek durust olmaz** -- durust
olan: "tespit katmaninda gercek ilerleme var, ama (a) genelleme sinirli,
(b) conf_threshold'un fine-tune edilmis modele gore yeniden ayarlanmasi
gerekiyor (Faz 3.0'in orijinal YOLO icin yaptigi sweep'in fine-tune edilmis
model icin bir benzeri, bu oturumda UYGULANMADI, acik bir sonraki adim)."

13 yeni birim test (`_write_yolo_label`, `_read_gt_by_frame`,
`_read_seqinfo_size`, `_count_gt_misses`, `confirm_no_leakage`). 157 test yesil.

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
- [x] **Faz 2.1:** canli demo'da bulunan uc gercek hata bulundu/duzeltildi --
      kayip kimlik kurtarma (lost-track gallery), hayalet-kutu cizim hatasi,
      ve kayip-galeride yanlis birlestirmeye yol acan mesafesiz eslesme
      (kok neden: sert mesafe siniri eklendi); bkz. "Faz 2.1" bolumu icin
      durustce raporlanmis once/sonra sayilari
- [x] **Faz 2.3:** MOT domaininde Re-ID fine-tune (MOT17-09/11'den, 02/04'e
      dokunmadan) — **MOT17-02'de Adaptive artik uc metrikte de baseline'i
      geciyor** (ilk kez); bkz. "Faz 2.3" bolumu icin sayilar ve durustce
      belirtilen sinirlamalar (62 kimlik, %100 train acc = overfitting riski)
- [x] **Faz 2.4:** Faz 2.1'in mesafe siniri (330px) hala gevsekti -- kullanicinin
      israrla isaret ettigi ayni bolgede (306px, sinirin hemen altinda) gercek
      bir yanlis-birlestirme bulundu ve gorsel olarak dogrulandi; taban sinir
      ~176px'e sikilastirildi + `missed`e gore ek daralma eklendi; bkz. "Faz 2.4"
- [x] **Faz 2.5:** gorunum galerisi (embedding gallery) -- tek EMA embedding
      yerine son 10 belirgin-farkli gorunum saklaniyor, kurtarmada en iyisi
      kullaniliyor; MOT17-04'te net iyilesme, MOT17-02'de karisik (bkz. "Faz 2.5")
- [x] **Faz 2.6:** Faz 2.4 (mesafe siniri) ile Faz 2.5'i (galeri) izole eden
      kontrollu ablation -- **galeri MOT17-02'de olcumde hic gorunmuyor
      (Baseline'la birebir ayni), MOT17-04'te kucuk ama gercek katki**
      (ID-switch -1/-2); onceki iyilesmenin cogu Faz 2.4'ten geliyormus.
      Bkz. "Faz 2.6" icin tam 4-kosullu tablo
- [x] **Faz 2.7:** 7 sekansta genisletildi -- **agregat seviyede Faz 2.4 aslinda
      Baseline'dan KOTU cikti** (MOTA 24.7%->23.3%, ID-switch 111->125); tek
      "kazanc" sekansi MOT17-04 temsili degilmis. Kok neden teshis edildi (dusuk
      cozunurluk/fps'te -- MOT17-05, 640x480@14fps -- sabit-oranli mesafe siniri
      GECERLI eslesmeleri reddediyor); duzeltme henuz UYGULANMADI. Bkz. "Faz 2.7"
- [x] **Faz 2.8:** mesafe sinirini kare hizina (fps) gore olcekleyen duzeltme
      uygulandi -- **MOT17-05'teki -%10.6 MOTA felaketi %10.1'e duzeldi**,
      agregat MOTA/IDF1'de artik Baseline'i geciyor (ID-switch hala hafif
      geride). Bkz. "Faz 2.8"
- [x] **Faz 2.9:** Occlusion Recovery Rate olculdu -- **MOT17-04'te (kalabalik)
      olcum pratikte imkansiz cikti** (28 event'in ~tumu gecersiz, kok neden
      arastirildi ve kod hatasi olmadigi dogrulandi), **MOT17-05'te (seyrek)
      calisti: ORR %40-46** (search_window'a gore). Kullanicinin orijinal
      "direk arkasi" sikayeti hala genel bir zayiflik, cozulmus degil. Bkz. "Faz 2.9"
- [ ] MOT17-05'te ID-switch hala yuksek (50->65) -- fps-farkindaligi MOTA'yi
      duzeltti ama bu sekansin kimlik-surekliligi sorunu ayrica arastirilmali
      (Faz 2.9'daki %40-46 ORR de ayni sorunu dogruluyor)
- [x] **Faz 2.10:** "association failure" hipotezi izole test edildi --
      ORR'un GT<->hipotez eslestirmesi greedy'den Hungarian'a gecirildi
      (tracker'in kendisi zaten Hungarian kullaniyordu). **MOT17-04'te SIFIR
      olculebilir degisiklik** (kare-seviyesi eslesme orani bit-bit ayni,
      %34.7); hipotez elendi. Bkz. "Faz 2.10" -- sonraki yon artik detection
      recall/IoU hassasiyeti (occlusion sinirinda), association algoritmasi degil.
- [x] **Faz 3.0:** YOLO conf_threshold sweep (0.35->0.15, MOT17-04, tek
      degisken disipliniyle) -- **frame-level recall %41.6->%51.5 arttı ama
      23 event'in "gecerli olculebilir" sayisi 1'de SABIT kaldi**, hicbir
      esik degeri yeni bir event acmadi. MOT metrikleri icin gercek bir
      sweet-spot var (~0.25-0.30: IDF1 tepe %50.8, MOTA tepe %31.0, altinda
      IDSW 3x/FP 2.4x patlıyor) ama bu, Faz 2.9'un ana ölçülebilirlik
      sorununu cozmuyor. Yan bulgu: `track_thresh=0.5` tum onceki fazlarda
      sessiz IKINCI bir esikmiş. Bkz. "Faz 3.0" -- "basit recall sorunu"
      hipotezi de zayifladi, sonraki adim esik ayari degil.
- [x] **Faz 3.1:** 23 event'in HER BIRI tek tek incelendi (Occlusion Event
      Forensics, A/B/C/D/E taksonomisi) -- **KOK NEDEN BULUNDU: 22/23'unde
      once/sonra sinir karesindeki gorunurluk 0.20-0.23 arasinda, yani
      vis>=0.2 esiginin hemen ustunde.** Bu tracker/detector hatasi degil,
      ORR'un "sinir karesi" TANIMININ matematiksel sonucu (esigi
      gecen ilk/son kare, esigin hemen yaninda olmak ZORUNDA). Faz 3.0'in
      neden hicbir sey degistirmedigini tam olarak aciklıyor. Yazarken
      script'in kendi siniflandirma mantiginda gercek bir hata bulunup
      duzeltildi (bkz. "Faz 3.1").
- [x] **Faz 3.1 replikasyonu (7 sekansin TUMU):** ayni kod/esik/taksonomi,
      BASKA hicbir sey degistirilmeden, kalan 6 sekansta (02,05,09,10,11,13)
      da calistirildi. **E-orani ile ortalama kalabalik arasinda OLCULMUS
      Pearson r=0.892 (n=7)** -- sinir-belirsizligi sorunu GENEL bir ORR
      problemi degil, sahne yogunluguyla korele, MOT17-04 (kalabalik 41.0,
      digerlerinin ~3 kati) gercek bir dis deger. ORR guvenilirlik sirasi:
      05/09/11 en guvenilir (E<%30), 02/13 orta, 10/04 neredeyse olcum-disi
      (E>%60). Sinir-karesi esigini vis>=0.5'e gore yeniden tanimlama fikri
      6/7 replikasyonla artik GENEL bir duzeltme olarak oncelikli DEGIL --
      once metrigin ne zaman anlamli oldugu sayisal olarak anlasildi, sonra
      degistirilmedi. Bkz. "Faz 3.1 replikasyonu"
- [x] **Faz 3.2:** kullanicinin gercek demo videosunda gordugu "5s/25s ayni
      kisi farkli ID" ekran goruntusu GT zemininde dogrulandi (tahmin degil)
      -- **YENI bir hata modu bulundu: "orta siddette surekli kismi
      gorunurluk"** (gt_id=3, hyp 7->40, 184 karelik bosluk, GT gorunurluk
      HICBIR ZAMAN 0.2 altina dusmedi). Faz 2.9-3.1'in TUM metodolojisi
      SADECE vis<0.2 event sayiyor, bu durumu hic kapsamiyor. Ayrica "P=0.04"
      etiketinin tracker guveni DEGIL, priority_engine skoru oldugu
      duzeltildi. Sonraki adim kullanicinin kararini bekliyor. Bkz. "Faz 3.2"
- [x] **Faz 3.3:** Partial Visibility Persistence Forensics -- Faz 3.2'nin
      hata modunun yayginligi 7 sekansta olculdu (AYRI script/metrik,
      ORR'un tanimi degistirilmedi). **43 episode'un %88.4'u (38/43) TEK
      BASINA MOT17-04'te** -- kalabalikla korelasyon r=0.935 (Faz 3.1'in
      E-orani korelasyonundan bile guclu). MOT17-11/13'te bu hata modu HIC
      yok. 43 episode'un HICBIRI RETAINED degil; basarisiz episode'lerin
      %48.7'si P1 (hic tespit yok) -- Faz 3.0/3.1'in bulgusuyla tutarli.
      17 birim test (biri gercek bir hata buldu, duzeltildi). Bkz. "Faz 3.3"
- [x] **Faz 3.4:** kucuk, izole prototip (`GalleryAwareTracker`, asil koda
      dokunmadan) -- embedding gallery'yi PRIMARY eslesmede de kullanmak
      MOT17-04'un 38 partial-visibility episode'unda **BIT-BIT SIFIR fark**
      yaratti (PVTR %7.9->%7.9). Basarisiz episode'lerin %65.7'si (P1+P3)
      bu fikirle prensipte duzelemezdi; kalan %34.3'te de degisiklik
      olmamasi mesafe sinirinin (`max_dist`) darbogaz olabilecegini
      dusunduruyor (test EDILMEDI, sadece cikarim). Bu fikir AutoTrack'e
      TASINMAYA DEGER DEGIL. Bkz. "Faz 3.4"
- [x] **Faz 3.5:** Faz 3.3/3.4'un actigi soru test edildi -- mesafe siniri
      (`max_dist`) partial-visibility darbogazi DEGIL: 0.08->0.15 genisletmek
      **0/38 episode'u degistirmedi**, ustune en buyuk ardisik-kare sicramayi
      78px->307px'e cikarip Faz 2.4'un duzelttigi turden bir hatayi geri
      getirdi. Tracker-tarafi mudahaleler (ReID skoru + mesafe siniri +
      confidence esigi) ucu de test edildi, ucu de negatif -- darbogaz
      buyuk olasilikla saf detection recall. Bkz. "Faz 3.5"
- [ ] Adaptif esiklerin validation split'e karsi ayarlanmasi, 7 sekansin
      tumunde (egitim/degerlendirme ayrimini koruyarak) tekrar
- [x] Demo video — MOT17-04 uzerinde, Faz 2 checkpoint'iyle, bkz. asagida
- [x] GitHub yayini — private, `github.com/l4NGEL/AutoTrack`
- [x] **Faz 4.0:** hava/drone genelleme dogrulamasi (VisDrone-MOT, 2 sahne,
      low/high yogunluk) -- **detector'un kendisi cokuyor**: low-density'de
      MOTA/IDF1=%0, FN=GT_obj (SIFIR gercek eslesme); dogrudan ham-YOLO
      incelemesiyle dogrulandi -- `yolov8n.pt` bu hava-acisi/olcekte
      (~20-30px) "person" sinifini pratikte tespit edemiyor. Asil arastirma
      sorusu (kalabalik-korelasyonu hava goruntusune genelleniyor mu)
      CEVAPLANAMADI -- tespit calismadan takip katmani olculemez. Sonraki
      adim (VisDrone-egitilmis detector) kullanicinin kararini bekliyor.
      Bkz. "Faz 4.0"
- [x] **Faz 4.1:** VisDrone'un kendi egitim verisiyle YOLO fine-tune --
      DOKUNULMAMIS 2 degerlendirme sahnesinde tekrar test edildi (sizinti
      yok, programatik dogrulandi). **Yuksek yogunluklu sahnede recall@IoU0.5
      %15.1->%68.9 (gercek, buyuk kazanc)**, ama dusuk yogunluklu sahnede
      neredeyse hic iyilesme yok (%0.0->%0.2) VE yuksek yogunluklu sahnede
      ID-switch 15->300'e firladi (fine-tune edilmis model icin
      conf_threshold=0.35 artik kalibre degil, test edilmedi). Tek yonlu
      zafer degil -- tespit katmaninda kismi, sahneye-ozel bir ilerleme.
      Bkz. "Faz 4.1"
- [ ] Faz 4.1'in actigi soru: fine-tune edilmis model icin conf_threshold
      sweep'i (Faz 3.0'in orijinali gibi) -- ID-switch patlamasini
      duzeltebilir mi, test edilmedi
- [ ] Faz 4.2 (gercek-zamanli/gomulu) ve Faz 4.3 (coklu kamera) -- Faz 4.0/4.1
      sonucuna gore hala ERTELENMIS durumda (bkz. onaylanmis plan)

## Demo video

Canli webcam yerine (bos oda garantisi yok) gercek, kalabalik bir sokak
sahnesi (MOT17-04-FRCNN, 524 kare) `config_demo.yaml` ile calistirildi --
Faz 2'nin gercek egitilmis Re-ID checkpoint'i devrede:

```bash
python -m evaluation.make_demo_video    # MOT17-04 karelerini tek video yapar
python main.py --config config_demo.yaml    # data/demo_output.mp4 uretir
```

Ortalama 524 kare boyunca **46 farkli track ID** (Faz 2.1-2.4'teki kayip-kimlik
kurtarma + mesafe siniri duzeltmeleri sonrasi -- oncesinde 55'ti; sayinin
tekrar artmasi -- 42/43'ten 46'ya -- Faz 2.4'un yanlis-birlestirmeleri
duzelttigiyle tutarli, bkz. "Faz 2.4"), karede ortalama **16
kisi**, uctan uca (tespit+takip+ReID+HUD+video yazma dahil) tipik olarak
~9-11 FPS (RTX 2080; bu oturumda arka arkaya cok sayida GPU isi
calistirildigi icin tek tek calistirmalar arasi FPS gozle gorulur sekilde
degisebiliyor -- bu tek-makinelik bir gozlem, kalici bir performans iddiasi
degil). Tek karede ayni anda `stable`/`uncertain`/`lost_candidate` guven
kovalarinin ucu de, hassas bolge tespiti ve gerekceli oncelik skoru gorunur
durumda. Kutular artik sadece o karede gercekten eslesen track'ler icin
ciziliyor (bkz. Faz 2.1 #2 -- eskiden Kalman'in "hayalet" tahminleri de
ciziliyordu).

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
