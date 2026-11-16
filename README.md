# NumPy ile Sıfırdan Perceptron: Kedi / Kedi Değil Sınıflandırması

Tek bir yapay nöron (perceptron), hazır bir derin öğrenme kütüphanesi kullanılmadan yalnızca NumPy ile yazılmış ve görüntülerin kedi içerip içermediğini ayırt edecek şekilde eğitilmiştir. Projede aktivasyon fonksiyonu **sigmoid**, kayıp fonksiyonu **ortalama karesel hata (MSE)**, optimizasyon yöntemi **stokastik gradyan inişi (SGD)**'dir.

## İçerik

| Dosya | Açıklama |
|---|---|
| `Module16_Perceptron_Cat_vs_NonCat.ipynb` | Tüm kodu, grafikleri ve sonuçları içeren defter |
| `requirements.txt` | Gerekli Python paketleri |

## Yöntem

Defter 28 bloktan oluşur ve şu adımları izler:

1. **Veri:** Cat vs Non-Cat veri seti (209 eğitim, 50 test görseli; 64×64 piksel, RGB). Eğitim verisi 167 eğitim + 42 doğrulama olarak bölünür; test dosyasına yalnızca en sonda bakılır.
2. **Ön işleme:** Görüntüler 4×4 ortalama havuzlama ile 16×16×3 = 768 özelliğe indirilir, [0, 1] aralığına getirilir ve eğitim ortalaması çıkarılır. Yalnızca eğitim görsellerine veri artırımı (yatay çevirme + küçük döndürme) uygulanır: 167 → 668 örnek.
3. **Model:** Parametre başlatma, sigmoid, ileri yayılım, MSE, geri yayılım ve tek örneklik SGD güncellemesi ayrı fonksiyonlar olarak yazılır, sonra `model()` fonksiyonunda birleştirilir.
4. **Doğrulama:** Analitik gradyanlar sayısal türevle karşılaştırılır (mutlak fark < 1e-6).
5. **Hiperparametre seçimi:** L2 katsayısı (λ = 0,03) ve epoch sayısı (41) yalnızca eğitim kısmı üzerinde 5 katlı çapraz doğrulamayla seçilir. Karar eşiği (0,33) aynı çapraz doğrulamanın out-of-fold tahminlerinden belirlenir.
6. **Değerlendirme:** Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, karışıklık matrisi, ROC ve Precision–Recall eğrileri, yanlış sınıflandırılan görseller ve öğrenilen ağırlık haritaları.
7. **Kendi fotoğraflarla deneme:** Eğitilen model, kullanıcının yüklediği fotoğraflar üzerinde denenir.

## Sonuçlar

`RANDOM_STATE = 42` ile elde edilen değerler:

| Küme | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---|---|---|
| Eğitim | 0,71 | 0,75 | 0,84 |
| Doğrulama | 0,57 | 0,57 | 0,69 |
| Test | 0,84 | 0,78 | 0,77 |

Test karışıklık matrisi: TN = 10, FP = 7, FN = 1, TP = 32.

### Sonuçları yorumlarken

- Test doğruluğunun eğitimden yüksek görünmesi modelin testte daha iyi olduğunu göstermez. Test kümesinin %66'sı kediyken eğitim verisinin yaklaşık üçte biri kedidir ve düşük karar eşiği (0,33) kedi sınıfını kayırır. Aynı model 0,50 eşiğiyle testte 0,54 doğruluk verir. Bu yüzden Balanced Accuracy ve ROC-AUC daha güvenilir ölçülerdir.
- Kendi seçilen 16 fotoğrafta model yalnızca 7'sini doğru sınıflandırmıştır (ROC-AUC 0,33). Aslan fotoğraflarına yüksek kedi olasılığı vermesi, modelin şekil yerine renk ve parlaklık örüntüsü öğrendiğini gösterir.
- Tek nöronlu doğrusal bir model kenar ve doku gibi uzamsal ilişkileri öğrenemez; görüntü sınıflandırmada bu nedenle evrişimli sinir ağları (CNN) kullanılır.

## Kurulum

Python 3.11 veya üzeri önerilir. Defter Google Colab üzerinde çalıştırılıp test edilmiştir.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Veri seti

Veri seti depoya dahil değildir: <https://www.kaggle.com/datasets/muhammeddalkran/catvnoncat>

Defter veriyi şu sırayla arar:

1. Defterle aynı klasör, `data/`, `datasets/` veya `catvnoncat/` klasörü
2. Bulamazsa `kagglehub` ile otomatik indirme

Elle indirmek için `train_catvnoncat.h5` ve `test_catvnoncat.h5` dosyalarını defterle aynı klasöre ya da `data/` klasörüne koymak yeterlidir.

## Çalıştırma

**Google Colab:** Defteri yükleyip *Runtime → Run all* seçin. Blok 26'da dosya seçici açılır; kedi ve kedi olmayan fotoğrafları birlikte seçin. Dosya adında `degil` geçen fotoğraflar "kedi değil" olarak etiketlenir.

**Yerel bilgisayar:**

```bash
jupyter notebook Module16_Perceptron_Cat_vs_NonCat.ipynb
```

Colab dışında dosya seçici çalışmaz. Blok 26'da `OWN_IMAGE_PATHS` listesine kendi fotoğraflarınızın yollarını yazın; liste boş kalırsa o hücre durur.

Tam deney (çapraz doğrulama dahil) Colab'da birkaç dakika sürer. Hızlı bir deneme için Blok 2'de `FAST_MODE = True` yapılabilir.

## Çıktılar

Defter çalıştırıldığında `outputs_module16_v5/` klasörü oluşur ve şunları içerir:

- Grafikler (`.png`): örnek görseller, sınıf dağılımı, sigmoid, çapraz doğrulama, öğrenme eğrileri, karışıklık matrisi, ROC/PR, eşik analizi, yanlış tahminler, ağırlık haritaları, kendi fotoğraflar
- Tablolar (`.csv`): metrikler, tahminler, çapraz doğrulama ve eşik seçimi
- `final_results.json`: tüm ayarların ve sonuçların özeti
- `perceptron_sgd_mse.pkl`: eğitilmiş model

Aynı içerik `Module16_Perceptron_Sonuclar.zip` olarak da paketlenir. Bu çıktılar yeniden üretilebildiği için `.gitignore` ile depo dışında tutulur.

## Hazırlayan

Rümeysa Hilal Özkaraman
