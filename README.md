Cat vs Non-Cat Sınıflandırması | Perceptron
Bu projede, NumPy kullanılarak sıfırdan geliştirilen tek nöronlu bir Perceptron ile görsellerin kedi içerip içermediği sınıflandırılır. Çalışma, DLI Module 16 – Binary Classification with Perceptron laboratuvarı kapsamında hazırlanmıştır.
Kullanılan Teknolojiler
Python, NumPy, Pandas, Matplotlib, Seaborn, h5py, Pillow, SciPy ve Scikit-learn (yalnızca veri bölme/değerlendirme için).
Veri Seti
Cat vs Non-Cat – Kaggle
- 0: Kedi değil
- 1: Kedi
Projede Neler Yapıldı?
- Görsellerin yüklenmesi, incelenmesi ve ön işlenmesi
- Perceptron, Sigmoid aktivasyonu, MSE kaybı ve SGD eğitiminin NumPy ile oluşturulması
- Eğitim, doğrulama ve test sonuçlarının karşılaştırılması
- Accuracy, Precision, Recall, F1 ve ROC-AUC ile model değerlendirmesi
- Öğrenme grafikleri, karışıklık matrisi ve hatalı tahminlerin görselleştirilmesi
- Modelin kaydedilmesi ve kendi görselinle tahmin yapılması
Çalıştırma
1. Module16_Perceptron_Cat_vs_NonCat_Adim_Adim.ipynb dosyasını Google Colab veya Jupyter Notebook ile açın.
2. Hücreleri sırasıyla çalıştırın. Notebook gerekli kütüphaneleri ve veri setini indirmeyi dener.
3. Otomatik indirme çalışmazsa Kaggle'dan train_catvnoncat.h5 ve test_catvnoncat.h5 dosyalarını indirip Notebook ile aynı klasöre veya data/ klasörüne koyun.
4. Grafikler, metrikler ve model dosyaları outputs_module16/ klasörüne kaydedilir.
Not
Model hazır bir sınıflandırıcı kullanılmadan geliştirilmiştir. Başarı değerleri, gerçek veri seti üzerinde Notebook çalıştırıldığında hesaplanır.
1. DLI Accelerated Data Science Teaching Kit, Module 16 Lab: Binary Classification with Perceptron (ödev PDF'si).
2. Kaggle, Cat vs Non-Cat Dataset.
3. NumPy Documentation, https://numpy.org/doc/.
4. scikit-learn Documentation, https://scikit-learn.org/stable/.
Proje türü: Eğitim amaçlı, sıfırdan geliştirilen ikili görüntü sınıflandırma uygulaması.
