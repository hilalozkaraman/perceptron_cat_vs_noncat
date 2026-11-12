Cat vs Non-Cat Görüntü Sınıflandırması | Perceptron
Bu proje, DLI Accelerated Data Science Teaching Kit – Module 16: Binary Classification with Perceptron laboratuvar ödevi kapsamında hazırlanmıştır. Amaç, görsellerin kedi içerip içermediğini belirleyen tek nöronlu bir Perceptron modelini NumPy ile sıfırdan geliştirmek ve modelin performansını değerlendirmektir.
Proje Hakkında
Hazır bir yapay sinir ağı modeli kullanılmadan, görüntüler üzerinde ikili sınıflandırma yapılmıştır. Modelde Sigmoid aktivasyon fonksiyonu, Mean Squared Error (MSE) kayıp fonksiyonu ve örnek bazlı Stochastic Gradient Descent (SGD) optimizasyonu uygulanmıştır. Çalışma, veri hazırlamadan tahmin sonuçlarının yorumlanmasına kadar adım adım bir Jupyter Notebook içinde sunulmuştur.
Veri Seti
Cat vs Non-Cat – Kaggle veri seti kullanılır. Görseller iki sınıfa ayrılır:
- 0 – Kedi değil: Görselde kedi bulunmuyor.
- 1 – Kedi: Görselde kedi bulunuyor.
Veriler train_catvnoncat.h5 ve test_catvnoncat.h5 dosyalarından okunur. Eğitim verisinin bir bölümü doğrulama için ayrılır; test verisi ise modelin son değerlendirmesinde kullanılır.
Kullanılan Teknolojiler
- Python ve NumPy: Perceptron modelinin geliştirilmesi ve sayısal işlemler
- Pandas: Sonuçların tablolaştırılması ve CSV çıktıları
- h5py ve Pillow: Görsellerin okunması ve işlenmesi
- Matplotlib ve Seaborn: Grafikler ve görselleştirmeler
- SciPy: Kullanıcı görselinin yeniden boyutlandırılması
- Scikit-learn: Veri bölme ve performans metrikleri (model eğitimi için değil)
Uygulama Aşamaları
1. Veri setinin yüklenmesi, örnek görsellerin ve sınıf dağılımının incelenmesi
2. Görsellerin sayısal vektörlere dönüştürülmesi ve piksellerin normalleştirilmesi
3. Ağırlıkların ve bias değerinin başlatılması; ileri ve geri yayılımın uygulanması
4. Sigmoid, MSE ve SGD kullanılarak modelin eğitilmesi
5. Doğrulama kaybına göre en iyi modelin seçilmesi ve erken durdurma
6. Bağımsız test verisiyle tahmin ve performans değerlendirmesi
7. Grafiklerin oluşturulması, modelin kaydedilmesi ve yeni bir görselle tahmin yapılması
Model Değerlendirmesi
Notebook içinde Accuracy, Precision, Recall, F1-score, MSE ve ROC-AUC metrikleri hesaplanır. Ayrıca eğitim/doğrulama öğrenme eğrileri, Confusion Matrix, ROC ve Precision–Recall eğrileri, sınıflandırma eşiği analizi ve yanlış tahmin edilen görüntüler incelenir.

Sonuçlar çalıştırma sırasında hesaplandığından burada doğrulanmamış bir başarı oranı verilmemiştir.
Nasıl Çalıştırılır?
1. Module16_Perceptron_Cat_vs_NonCat_Adim_Adim.ipynb dosyasını Google Colab veya Jupyter Notebook ile açın.
2. Hücreleri sırasıyla çalıştırın. Notebook gerekli Python paketlerini ve veri setini indirmeyi dener.
3. Veri seti otomatik indirilemezse Kaggle'dan iki .h5 dosyasını indirip Notebook ile aynı klasöre veya data/ klasörüne yerleştirin.
4. Eğitim ve değerlendirme tamamlandığında grafikler, CSV/JSON sonuçları ve kaydedilmiş model outputs_module16/ klasöründe bulunur.
5. Kendi fotoğrafınızla tahmin yapmak için Blok 26'da MY_IMAGE_PATH alanına dosya yolunu yazın veya Colab'da AUTO_UPLOAD = True ayarını kullanın.
Çıktılar
Proje; öğrenme grafikleri, performans tabloları, test tahminleri ve perceptron_sgd_mse.pkl model dosyasını üretir. Sonuçlar ayrıca ZIP paketi olarak dışa aktarılır.
Not: Bu çalışma, tek nöronlu bir modelin temel işleyişini göstermek amacıyla hazırlanmıştır. Daha karmaşık görüntü sınıflandırma modelleriyle aynı performansı hedeflemez.
