import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wfdb
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_DIR = './physionet_mitdb'
SCALO_DIR = './scalograms'
os.makedirs(SCALO_DIR, exist_ok=True)


def load_and_segment(record_name, seg_length=500, num_per_class=30):
    
    record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    sig = record.p_signal[:,0] 
    ann = wfdb.rdann(os.path.join(DATA_DIR, record_name), 'atr')
    X, y = [], []
    
    for s, sym in zip(ann.sample, ann.symbol):
        start = s - seg_length//2
        if start < 0 or start+seg_length > len(sig):
            continue
        beat = sig[start:start+seg_length]
        X.append(beat)
      
        if sym == 'N':       lab = 0
        elif sym == 'A':     lab = 1
        elif sym == 'V':     lab = 2
        else:                continue
        y.append(lab)
       
        if y.count(lab) >= num_per_class:
            continue
    return np.array(X), np.array(y)


def make_scalogram(signal1d, widths=np.arange(1, 129), wavelet='morl'):
   
    cwtm = signal.cwt(signal1d, signal.morlet2, widths, w=5.0)
    
    norm = (cwtm - cwtm.min()) / (cwtm.max() - cwtm.min())
    return norm


def prepare_dataset(record_list):
    X_all, y_all = [], []
    for rec in record_list:
        X, y = load_and_segment(rec)
        for i, beat in enumerate(X):
            scalo = make_scalogram(beat)
            # resize to 224x224
            img = plt.imshow(scalo, cmap='jet')
            plt.axis('off')
            fname = f'{rec}_{i}.png'
            plt.savefig(os.path.join(SCALO_DIR, fname), bbox_inches='tight', pad_inches=0)
            plt.close()
            X_all.append(fname)
            y_all.append(y[i])
    return np.array(X_all), np.array(y_all)


records = ['100', '101', '102']  
X_files, y_labels = prepare_dataset(records)


df =  np.vstack((X_files, y_labels)).T
import pandas as pd
df = pd.DataFrame(df, columns=['filename','class'])
df['class'] = df['class'].astype(int)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

train_gen = ImageDataGenerator(rescale=1./255,
                               horizontal_flip=True,
                               rotation_range=15)
val_gen   = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_dataframe(train_df,
                                           directory=SCALO_DIR,
                                           x_col='filename', y_col='class',
                                           target_size=(224,224),
                                           class_mode='categorical',
                                           batch_size=16)
val_flow   = val_gen.flow_from_dataframe(val_df,
                                         directory=SCALO_DIR,
                                         x_col='filename', y_col='class',
                                         target_size=(224,224),
                                         class_mode='categorical',
                                         batch_size=16)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)   
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_flow,
          epochs=10,
          validation_data=val_flow)


for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_flow,
          epochs=5,
          validation_data=val_flow)


model.save('ecg_arrhythmia_resnet50.h5')
