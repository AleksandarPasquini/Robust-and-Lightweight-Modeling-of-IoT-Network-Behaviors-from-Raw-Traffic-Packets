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
def predict_embeddings(strategy, compressor, pca):
    new_compress = []
    max_seq_length = 0
    for byte_string in strategy[0:10000]:
        compressor.compress(bytes.fromhex(byte_string))
        compressed = list(compressor.flush(zlib.Z_FULL_FLUSH))
        if len(compressed) > max_seq_length:
            max_seq_length = len(compressed)
        new_compress.append(compressed)
    long_sequences = pad_sequences(new_compress, maxlen=max_seq_length, padding='post')
    seen_embeddings = pca.fit_transform(long_sequences)
    return seen_embeddings



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
print("Compression")
print("#############################################")
compressor = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, wbits=-15)
embedding = predict_embeddings(strategy, compressor, PCA(n_components=64))
classify = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)
classify.fit(embedding, labels[0:10000]) 
tracker.start()
start_time = time.time()
seen_embeddings = predict_embeddings(strategy, compressor, PCA(n_components=64))
end_time = time.time()
classify.predict(embedding)
end_time2 = time.time()
tracker.stop()
execution_time = end_time - start_time
inference_time = end_time2 - start_time
print(f"The embedding generation time of the code block is {execution_time} seconds")
print(f"The inference time of the code block is {inference_time} seconds")