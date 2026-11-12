Cat vs Non-Cat Image Classification — Perceptron from Scratch
DLI Accelerated Data Science Teaching Kit · Module 16 Lab: Binary Classification with Perceptron

Bu proje, bir görselin kedi (1) veya kedi değil (0) olduğunu tahmin etmek için NumPy ile sıfırdan tek nöronlu bir Perceptron (sigmoid sınıflandırıcı) geliştirir. Ödevin gerektirdiği Sigmoid aktivasyonu, Mean Squared Error (MSE) kaybı ve örnek bazlı Stochastic Gradient Descent (SGD) kullanılır.

Model eğitimi için TensorFlow, PyTorch veya scikit-learn sınıflandırıcıları kullanılmaz. Kod; paylaşılan Random Forest çalışmasının düzenine benzer şekilde, 28 numaralı blok ve Türkçe açıklamalarla hazırlanmıştır.
Sonuçların doğruluğu: Notebook, H5 biçimiyle uyumlu örnek veri üzerinde uçtan uca teknik olarak test edilmiştir. Gerçek Cat vs Non-Cat veri setine ait doğruluk, F1 ve AUC değerleri bu README'ye yazılmamıştır; bunlar Notebook gerçek veri üzerinde çalıştırıldığında hesaplanır.

İçindekiler
- Projenin amacı
- Kullanılan teknolojiler
- Veri seti
- Kurulum ve çalıştırma
- Modelin matematiksel yapısı
- Deney tasarımı
- Notebook bölümleri
- Üretilen dosyalar
- Kendi görselinle tahmin
- Sınırlılıklar ve notlar
- Kaynaklar
Projenin amacı
PDF'de belirtilen temel hedefler:
1. Perceptron ağırlıklarını (w) ve bias değerini (b) başlatmak.
2. İleri yayılımda sigmoid aktivasyonunu ve MSE kaybını hesaplamak.
3. MSE'nin analitik türevleriyle gradyanları bulmak.
4. Her eğitim örneğinden sonra ağırlıkları SGD ile güncellemek.
5. Bu adımları tek bir model() fonksiyonunda birleştirmek.
6. Cat vs Non-Cat görsellerinde ikili sınıflandırma yapmak ve yeni bir fotoğraf için tahmin üretmek.
Ek olarak gradyan kontrolü, eğitim/doğrulama eğrileri, karışıklık matrisi, ROC–AUC, Precision–Recall eğrisi, yanlış sınıflandırmalar ve model kaydetme işlemleri uygulanır.
Kullanılan teknolojiler
- Python 3 ve Jupyter Notebook / Google Colab
- NumPy: Perceptron matematiği, ağırlıklar ve SGD
- h5py: H5 veri dosyalarını okuma
- Pandas: Sonuç tabloları ve CSV kayıtları
- Matplotlib, Seaborn: Görselleştirme
- scikit-learn: Eğitim/doğrulama ayrımı ve metrikler (model eğitimi için değil)
- Pillow, SciPy: Haricî fotoğraf yükleme ve yeniden boyutlandırma
- kagglehub: Veri setini indirme girişimi
GPU veya RAPIDS gerekli değildir. Notebook eksik Python paketlerini çalışma anında yüklemeyi dener; internet erişimi gerekebilir.

Yerel ortam için isteğe bağlı kurulum:
python -m pip install numpy pandas h5py matplotlib seaborn scipy pillow scikit-learn kagglehub notebook
Veri seti
Cat vs Non-Cat: Kaggle — muhammeddalkran/catvnoncat

Beklenen dosyalar:
train_catvnoncat.h5
test_catvnoncat.h5
H5 içindeki anahtarlar:
Dosya	Görseller	Etiketler
train_catvnoncat.h5	train_set_x	train_set_y
test_catvnoncat.h5	test_set_x	test_set_y


Etiketler 0 = kedi değil, 1 = kedi olacak şekilde ele alınır. Görseller RGB biçimindedir; veri setinin yaygın sürümünde boyut 64 × 64 × 3 olup görüntü başına 12.288 özellik elde edilir. Notebook, gerçek H5 boyutlarını okuyarak özellik sayısını hesaplar.

Ön işleme: Her görsel tek boyutlu bir vektöre düzleştirilir ve piksel değerleri 255'e bölünerek [0, 1] aralığına çekilir. Matris biçimi X = (özellik_sayısı, örnek_sayısı) şeklindedir.
Kurulum ve çalıştırma
Seçenek A — Google Colab (önerilen)
1. Google Colab sayfasını aç.
2. Module16_Perceptron_Cat_vs_NonCat_Adim_Adim.ipynb dosyasını File → Upload notebook ile yükle.
3. Runtime → Run all ile hücreleri sırayla çalıştır.
4. Notebook mevcut veri dosyalarını arar, bulamazsa kagglehub üzerinden indirmeyi dener. İndirme için internet, bazı ortamlarda Kaggle erişimi/kimlik doğrulaması gerekebilir.
5. Otomatik indirme başarısız olursa veri setini Kaggle sayfasından indir ve iki .h5 dosyasını /content/data/ klasörüne yükleyip hücreleri yeniden çalıştır.
6. Sonuçlar çalışma dizinindeki outputs_module16/ klasörüne ve Module16_Perceptron_Sonuclar.zip arşivine kaydedilir.
Seçenek B — Yerel Jupyter
python -m pip install numpy pandas h5py matplotlib seaborn scipy pillow scikit-learn kagglehub notebook
jupyter notebook Module16_Perceptron_Cat_vs_NonCat_Adim_Adim.ipynb
Veri dosyalarını Notebook ile aynı dizine veya data/ klasörüne koyabilirsin:
proje/
├── README.md
├── Module16_Perceptron_Cat_vs_NonCat_Adim_Adim.ipynb
└── data/
    ├── train_catvnoncat.h5
    └── test_catvnoncat.h5
Ardından Notebook'ta hücreleri üstten alta çalıştır.
Modelin matematiksel yapısı
Bu çalışmadaki model, gizli katman içermeyen tek sigmoid nöronlu doğrusal sınıflandırıcıdır. Bazı eğitim kaynaklarında bu yapı Perceptron olarak adlandırılır.

İleri yayılım:
z = w.T @ x + b
a = sigmoid(z) = 1 / (1 + exp(-z))
Kayıp fonksiyonu (MSE):
MSE = mean((a - y)²)
Tek örnek için gradyan:
delta = 2 * (a - y) * a * (1 - a)
dw = x * delta
db = delta
SGD güncellemesi:
w = w - learning_rate * dw
b = b - learning_rate * db
Her epoch'ta eğitim örnekleri karıştırılır ve her örnekten sonra parametre güncellenir. Bu uygulama tam yığın (full-batch) veya mini-batch optimizasyonu değildir. Analitik türevler, küçük bir örneklemde sayısal gradyan testiyle de kontrol edilir.
Deney tasarımı
- Orijinal eğitim verisinin %20'si doğrulama için ayrılır (VAL_SIZE = 0.20).
- Orijinal test kümesi eğitim, erken durdurma veya karar eşiği seçimi için kullanılmaz.
- Rastgelelik tohumu 42 olarak belirlenmiştir.
- Başlangıç öğrenme hızı 0.0007, maksimum epoch sayısı 300 ve erken durdurma sabrı 65 epoch'tur (FAST_MODE = False).
- Hızlı kontrol için FAST_MODE = True: en fazla 40 epoch ve 15 epoch sabır.
- Temel sınıflandırma eşiği 0.50 olarak kullanılır. Alternatif eşik değerleri doğrulama kümesinde incelenir.
- Doğrulama MSE'sine göre en iyi ağırlıklar saklanır; bağımsız test metrikleri eğitimden sonra hesaplanır.
Bu parametreler Notebook'un Blok 3 bölümünden değiştirilebilir.
Notebook bölümleri
Bloklar	İçerik
1–3	Ortam, bağımlılıklar, hiperparametreler
4–8	H5 veri setinin bulunması, yüklenmesi, incelenmesi ve ön işleme
9–13	Parametre başlatma, sigmoid, forward, MSE, backpropagation, gradyan testi
14–16	Tek örneklik SGD, tahmin ve ana model() eğitim döngüsü
17–22	Öğrenme eğrileri, metrikler, confusion matrix, ROC/PR ve eşik analizi
23–24	Hatalı tahmin örnekleri ve öğrenilmiş RGB ağırlıkları
25–26	Modeli kaydetme/yükleme ve kişisel fotoğrafla tahmin
27–28	Deneyin yorumlanması, özet sonuçlar ve çıktıların ZIP olarak paketlenmesi


Üretilen dosyalar
Notebook çalıştığında aşağıdaki dosyalar üretilir:
outputs_module16/
├── 01_ornek_gorseller.png
├── 02_sinif_dagilimi.png
├── 03_sigmoid.png
├── 04_ogrenme_egrileri.png
├── 05_confusion_matrix.png
├── 06_roc_pr.png
├── 07_sinif_bazinda_metrikler.png
├── 08_esik_analizi.png
├── 09_yanlis_tahminler.png
├── 10_agirlik_haritalari.png
├── 11_tek_gorsel_tahmini.png
├── training_history.csv
├── metrics_train_validation_test.csv
├── per_class_metrics.csv
├── threshold_analysis_validation.csv
├── misclassified_samples.csv
├── test_predictions.csv
├── sinif_dagilimi.csv
├── roc_points.csv
├── final_results.json
├── yorumlar.txt
└── perceptron_sgd_mse.pkl

Module16_Perceptron_Sonuclar.zip
Değerlendirilen metrikler Accuracy, Balanced Accuracy, Precision, Recall, F1, MSE, ROC-AUC ve Average Precision gibi ölçümleri kapsar. Bazıları tablo, bazıları grafik veya JSON özetinde sunulur.
ZIP ve çıktı dosyaları gerçek veri setiyle Notebook çalıştırıldıktan sonra oluşur. Bunlar bu depoya önceden hesaplanmış gerçek deney sonuçları olarak eklenmiş değildir.

Kendi görselinle tahmin
Notebook'un Blok 26 bölümünde iki seçenek vardır:
# Google Colab'da dosya seçme penceresi açmak için:
AUTO_UPLOAD = True
veya:
# Var olan bir görselin yolunu kullanmak için:
MY_IMAGE_PATH = '/content/ornek_kedi.jpg'
AUTO_UPLOAD = False
Görsel RGB biçimine dönüştürülür, eğitimdeki giriş boyutuna ölçeklenir ve modelden P(kedi) olasılığıyla birlikte tahmini sınıf elde edilir. Görsel verilmezse Notebook, bağımsız test kümesinden ilk görseli demo olarak kullanır; bu kendi fotoğrafınız değildir.
Sınırlılıklar ve notlar
- Tek sigmoid nöron, karmaşık görüntüler için sınırlı bir doğrusal karar sınırı öğrenir; CNN performansına ulaşması beklenmez.
- Sigmoid + MSE seçimi ödevin açık gereksinimidir; başka bir kayıp fonksiyonuyla değiştirilmemiştir.
- Test skorları veri setine ve çalıştırma ortamına bağlıdır; doğrulanmamış örnek skorlar rapora eklenmemelidir.
- .pkl model dosyasını yalnızca güvenilir kaynaklardan yükleyin; Python pickle, güvenilmeyen dosyalar için güvenli değildir.
- Eğitim için CPU yeterlidir; RAPIDS/CUDA bağımlılığı yoktur.
Kaynaklar
1. DLI Accelerated Data Science Teaching Kit, Module 16 Lab: Binary Classification with Perceptron (ödev PDF'si).
2. Kaggle, Cat vs Non-Cat Dataset.
3. NumPy Documentation, https://numpy.org/doc/.
4. scikit-learn Documentation, https://scikit-learn.org/stable/.
Proje türü: Eğitim amaçlı, sıfırdan geliştirilen ikili görüntü sınıflandırma uygulaması.
