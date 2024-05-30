import numpy as np
import tensorflow as tf
import zlib
import csv
from scapy.all import *
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA

from codecarbon import OfflineEmissionsTracker
import time
import resource
import os
from memory_profiler import profile

@profile
def predict_embeddings(strategy, max_sequence_length, first_dense_model):
    new_num_bytes = [list(bytes.fromhex(byte_string)) for byte_string in strategy[0:10000]]
    long_sequences = pad_sequences(new_num_bytes, maxlen=max_sequence_length, padding='post')
    final = first_dense_model.predict(long_sequences)
    return final

tf_strategy = tf.distribute.MirroredStrategy() 

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )

def largest_factor_pair(n):
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i, n // i
    return None

def file_to_list(file_path):
    with open(file_path, 'r') as file:
        content_list = [int(line.strip()) for line in file]
    return content_list

def read_file_to_list_of_lists(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            inner_list = [str(item) for item in line.strip().split(',')]
            data.append(inner_list)
    return data

MAC_list={
0:"Smart Things",
1:"Amazon Echo",
2:"Netatmo Welcome",
3:"TP-Link Day Night Cloud camera",
4:"Samsung SmartCam",
5:"Dropcam",
6:"Insteon Camera",
7:"Withings Smart Baby Monitor",
8:"Belkin Wemo switch",
9:"TP-Link Smart plug",
10:"iHome",
11:"Belkin wemo motion sensor",
12:"NEST Protect smoke alarm",
13:"Netatmo weather station",
14:"Withings Smart scale",
15:"Blipcare Blood Pressure meter",
16:"Withings Aura smart sleep sensor",
17:"Light Bulbs LiFX Smart Bulb",
18:"Triby Speaker",
19:"PIX-STAR Photo-frame",
20:"HP Printer",
21:"Nest Dropcam"
}


inputs = read_file_to_list_of_lists("../Embeddings/byte_data_set.txt")
Whole_labels = file_to_list("../Embeddings/labels.txt")
IP_labels = file_to_list("../Embeddings/IP_labels.txt")
transport_labels = file_to_list("../Embeddings/transport_labels.txt")
raw_labels = file_to_list("../Embeddings/Raw_labels.txt")
index = 0
dataset = ["Whole_packet", "Whole_packet_no_dst", "Eth_payloads", "Eth_no_dst_payloads", "IP_headers", "IP_no_dst_headers", "IP_payloads", "Transport_payload", "Raw_data"]
nam = {"Whole_packet": "", "Whole_packet_no_dst": "no_dst", "Eth_payloads": "Eth", "Eth_no_dst_payloads": "Eth_no_dst", "IP_headers": "IP_Header", "IP_no_dst_headers": "IP_Header_no_dst", "IP_payloads": "IP", "Transport_payload": "transport", "Raw_data": "raw"}
tracker = OfflineEmissionsTracker(country_iso_code="AUS")
strategy = inputs[0]

labels = Whole_labels
y_length = len(Whole_labels)
split_index = int(y_length * 0.7)
num_classes = len(set(Whole_labels))


print("#############################################")
print("LSTM")
print("#############################################")


num_bytes = [list(bytes.fromhex(byte_string)) for byte_string in strategy]
max_sequence_length = max(len(vector) for vector in num_bytes)
print(max_sequence_length)
padded_sequences = pad_sequences(num_bytes, maxlen=max_sequence_length, padding='post')
x_first_70, x_last_30, first_70_labels, last_30_labels = train_test_split(padded_sequences, labels, test_size=0.3, stratify=labels, random_state=42)

    
# LSTM
Technique = "LSTM"
with tf_strategy.scope():
    model = Sequential()
    model.add(Embedding(256, output_dim=64))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(64, name="embedd_dense_output"))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_first_70,  np.array(first_70_labels), epochs=2, batch_size=512)
model.predict(x_first_70[:4])
first_dense_output = model.get_layer("embedd_dense_output").output
first_dense_model = Model(inputs=model.layers[0].input, outputs=first_dense_output)
print(first_dense_model.summary())
print(model.summary())
classify = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)
embedding = predict_embeddings(strategy, max_sequence_length, first_dense_model)
classify.fit(embedding, labels[0:10000]) 
tracker.start()
start_time = time.time()
final = predict_embeddings(strategy, max_sequence_length, first_dense_model)
end_time = time.time()
classify.predict(final)
end_time2 = time.time()
tracker.stop()
execution_time = end_time - start_time
inference_time = end_time2 - start_time
print(f"The embedding generation time of the code block is {execution_time} seconds")
print(f"The inference time of the code block is {inference_time} seconds")