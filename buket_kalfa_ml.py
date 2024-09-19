# -*- coding: utf-8 -*-
"""Buket_Kalfa_ML

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/buket-kalfa-ml-cb87c9c4-b590-4bea-b86a-a6eaad814e61.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240919/auto/storage/goog4_request%26X-Goog-Date%3D20240919T090008Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D5b25f8459942758db084208eef888751a0df48c7056401f96f543c6e1df024d7fa7fa5b05b7d03b1f3fa25b1a61f985a35b2f3abfc2b7e44471801b8a6ee6f6237fb6e2fa4e4b00e3f0ffc6827ef47b8e2f4a059111a93e409400efb930a67f78cbcbe01af8dbfa0a31c197bfcd2861d7195ef8043eda8acd29ae18b5a3156464026bfd0aed4237e134212d19f56905aac6de4ac4ad4ecc895784acbdbc4330146bfca913210ee8180c9564fbb70574e66749b4a27b2a1bb408bde661ff5f299ac25fdbcf7f5c6da3d9c1b325c8a0e7b8411829fd1972909bca244e27c1c9835e51f9a5dec0895e604689afd61d77a383f672bbf1b8c73b58071e51dced7192d
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'mushroom-classification:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F478%2F974%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240919%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240919T090007Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D9a38443977e751c2b3a4f9ed80f00af3a046d9693eb77d2ac06b504906bed44bf784e3c23a7ea5a5fd45e2ea2a27fb90b429666a421025fea387dc62432d5f03b3a240c7423fa2a9d840cf5bdeb4007a1cd5cc1cd92e73f91276f63abaff1b79be7d91bb21a6f798cc605d884fe5ce1ff6d15d836c88b00debfc0acee52b16baedc19803bec754cb4afc8f9fa42e51aea8b4f774b73e0aee151a0122f8f1965e1eaf1453aecc5cb6d1698b8d3ae48409f77dbc246c382ba332752a5ee4e1c72963e59bc98b5100527177ea09154815c6a7c2ad58a590223979743f5672ce69af6e0a9655442d2cc3e26ef2043f85480b71e66196472e4d23c97b61f888ba4b6a'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

# Veri setinin ilk 5 satırını görüntüleyelim
df.head()

# Veri seti hakkında genel bilgi alalım
df.info()

# Veri setindeki özet istatistikleri inceleyelim
df.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Bağımlı değişkenin (class) dağılımı
sns.countplot(x='class', data=df)
plt.title('Sınıf Dağılımı (Yenilebilir/Zehirli)')
plt.show()

# Diğer kategorik değişkenlerin bazılarını inceleyelim
plt.figure(figsize=(12,6))
sns.countplot(x='odor', data=df, hue='class')
plt.title('Koku Özelliği ile Sınıf Dağılımı')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Tüm kategorik değişkenleri sayısal hale getirmek için Label Encoding
label_encoder = LabelEncoder()

for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

# Veri kümesini eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split

X = df.drop('class', axis=1)  # Bağımsız değişkenler
y = df['class']  # Bağımlı değişken (yenilebilir/zehirli)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression  # LogisticRegression sınıfını import etmelisiniz
from sklearn.metrics import accuracy_score, classification_report  # Doğruluk skoru ve sınıflandırma raporu için gerekli metrikler

# İterasyon sayısını artırarak uyarıyı çözebiliriz
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Model performansı
y_pred = model.predict(X_test)
print('Doğruluk Skoru:', accuracy_score(y_test, y_pred))
print('Sınıflandırma Raporu:')
print(classification_report(y_test, y_pred))

from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# K-Means kümeleme (n_clusters=2 çünkü mantarlar iki sınıfa ayrılıyor)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Küme etiketlerini ekleyelim
df['cluster'] = kmeans.labels_

# Kümeleme sonuçlarını inceleyelim
sns.countplot(x='cluster', data=df)
plt.title('K-Means Kümeleme Sonuçları')
plt.show()

# Kümelerin sınıflarla nasıl eşleştiğini görelim
ct = pd.crosstab(df['class'], df['cluster'])
print(ct)