# mushroom_final_project_Buket_Kalfa_
mushroom_final_project_Buket_Kalfa_

# Mantar Sınıflandırma Projesi
Bu proje, mantarların yenilebilir veya zehirli olduğunu sınıflandırmak için makine öğrenmesi tekniklerini kullanır. 
İki temel öğrenme tekniği uygulanmıştır: Gözetimli Öğrenme (Logistic Regression) ve Gözetimsiz Öğrenme (K-Means Kümeleme).

[Kaggle Notebook "Buket_Kalfa_ML" linki]https://www.kaggle.com/code/buketkalfa/buket-kalfa-ml

## Veri Seti
Veri seti, Kaggle'dan alınan mushrooms.csv dosyasını içerir. Bu veri seti 8124 örnek ve 23 özelliğe sahiptir. Özellikler, mantarların çeşitli fiziksel ve kimyasal özelliklerini temsil eder.

## Kurulum
Projede kullanılan kütüphaneler:
*•	numpy*
*•	pandas*
*•	matplotlib*
*•	seaborn*
*•	sklearn*
Bu kütüphaneler, veri işleme, görselleştirme ve modelleme işlemleri için kullanılır.

## 1. Veri Setini Yükleme
```ruby
import pandas as pd
```
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
## 2. Veri Analizi
Veri setindeki bağımlı değişken olan class'ın (yenilebilir/zehirli) dağılımı görselleştirildi.

odor (koku) özelliğinin sınıf dağılımı üzerinde nasıl bir etkisi olduğuna bakıldı.

```ruby
import matplotlib.pyplot as plt
import seaborn as sns
```

# Bağımlı değişkenin (class) dağılımı
```ruby
sns.countplot(x='class', data=df)
plt.title('Sınıf Dağılımı (Yenilebilir/Zehirli)')
plt.show()
```

# Koku Özelliği ile Sınıf Dağılımı
```ruby
plt.figure(figsize=(12,6))
sns.countplot(x='odor', data=df, hue='class')
plt.title('Koku Özelliği ile Sınıf Dağılımı')
plt.show()
```

## 3. Veri Ön İşleme
Veriler etiketlendi ve modellemeye hazır hale getirildi.
Eğitim ve test veri setleri ayrıldı.

```ruby
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 4. Gözetimli Öğrenme: Logistic Regression
Logistic Regression algoritması kullanılarak mantarların sınıflandırılması sağlandı.
Modelin doğruluk skoru ve sınıflandırma raporu alındı.


```ruby
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Doğruluk Skoru:', accuracy_score(y_test, y_pred))
print('Sınıflandırma Raporu:')
print(classification_report(y_test, y_pred))
```

## 5. Gözetimsiz Öğrenme: K-Means Kümeleme

K-Means algoritması ile mantarlar kümelere ayrıldı ve kümelerin sınıflarla olan ilişkisi incelendi.


```ruby
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
```

# K-Means Kümeleme
```ruby
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
df['cluster'] = kmeans.labels_
```

# Kümeleme Sonuçları
```ruby
sns.countplot(x='cluster', data=df)
plt.title('K-Means Kümeleme Sonuçları')
plt.show()
```

# Kümelerin Sınıflarla Eşleşmesi
```ruby
ct = pd.crosstab(df['class'], df['cluster'])
print(ct)
```

Sonuçlar
# Gözetimli Öğrenme (Logistic Regression)
•	Doğruluk Skoru: 0.9477

•	Sınıflandırma Raporu:

o	Yenilebilir: Precision 0.95, Recall 0.95, F1-Score 0.95

o	Zehirli: Precision 0.94, Recall 0.95, F1-Score 0.95

# Gözetimsiz Öğrenme (K-Means Kümeleme)

K-Means algoritması, mantarları iki kümeye ayırdı. Kümeler ve gerçek sınıflar arasındaki eşleşme:
•	Kümeler ve sınıfların eşleşmesi aşağıda gösterilmiştir:

cluster     0     1
class               
0         192  4016
1        1744  2172

## Sonuç
Logistic Regression modelimiz yüksek doğruluk sağlamıştır ve K-Means kümeleme sonuçları, sınıfların belirli kümelere nasıl ayrıldığını göstermektedir. Her iki teknik de veri setindeki mantarları analiz etmede etkili olmuştur.

