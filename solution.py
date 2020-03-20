from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
import csv, copy, random
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def printNumbers(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if (index == 0):
                continue
            for i, item in enumerate(row[:-1]):
                if item == '1':
                    print(" ", end=" ")
                elif item == '0':
                    print("#", end=" ")
                if (i+1)%9 == 0:
                    print("")
            print("")

# for each number, generate 50 versions with noise
def generateDataset():
    with open('./data.csv', 'w') as d:
        with open('./numbers.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for i in range(0, 1000):
                    while(True):
                        temp = copy.copy(row)
                        a, b = random.randint(0, len(temp)-2), random.randint(0, len(temp)-2)
                        temp[a], temp[b] = temp[b], temp[a]
                        if (temp != row):
                            break
                    d.write(str(temp).replace("[", "").replace("]", "").replace("'", "").replace(" ", "") + '\n')

# convert dataframe to tf dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# generateDataset()
# printNumbers("./data.csv")

URL = './data.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=30)
train, val = train_test_split(train, test_size=50)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

feature_columns = []
for header in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45']:
    feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)