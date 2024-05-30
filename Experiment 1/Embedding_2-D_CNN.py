import numpy as np
import tensorflow as tf
import csv
from scapy.all import *
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from codecarbon import OfflineEmissionsTracker
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

def write_vectors_to_tsv(file_path, vectors, labels):
    with open(file_path + ".tsv", 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for line in zip(vectors, labels):
            writer.writerow(line)


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

def similarity_calculator(vectors, class_index, Technique):
    print("#############################################")
    print(Technique)
    print("#############################################")
    random.seed(42)
    avg_similarities_per_class = []
    limit = 10000
    for j, sim_class in enumerate(class_index):
        name = MAC_list[j]
        target_vectors = [vectors[i] for i in sim_class]
        if len(target_vectors) > limit:
            target_vectors = random.sample(target_vectors, limit)
        if target_vectors != []:
            similarity_matrix = cosine_similarity(target_vectors)
            avg_similarity = np.mean(similarity_matrix)
            avg_similarities_per_class.append(avg_similarity)
            std_deviation_per_class = np.std(similarity_matrix)
            print(f"Class {name} Average Similarity: {avg_similarity:.3f}±{std_deviation_per_class:.3f}")
        else:
            print(f"Class {name} has no embeddings")
    
    avg_similarity_across_all_classes = np.mean(avg_similarities_per_class)
    std_deviation_across_all_classes = np.std(avg_similarities_per_class)
    print(f"Average Similarity across all classes: {avg_similarity_across_all_classes:.3f}±{std_deviation_across_all_classes:.3f}")
    
inputs = read_file_to_list_of_lists("../Embeddings/byte_dataset.txt")
Whole_labels = file_to_list("../Embeddings/labels.txt")
IP_labels = file_to_list("../Embeddings/IP_labels.txt")
transport_labels = file_to_list("../Embeddings/transport_labels.txt")
raw_labels = file_to_list("../Embeddings/Raw_labels.txt")
index = 0
dataset = ["Whole_packet", "Eth_payloads", "Eth_no_dst_payloads", "IP_payloads", "Transport_payload", "Raw_data"]
nam = {"Whole_packet": "", "Eth_payloads": "Eth", "Eth_no_dst_payloads": "Eth_no_dst", "IP_payloads": "IP", "Transport_payload": "transport", "Raw_data": "raw"}

tracker = OfflineEmissionsTracker(country_iso_code="AUS")
for strategy in inputs:
    Technique = "2-D CNN"
    prefix = nam[dataset[index]]
    file_name = Technique + "_" + prefix
    
    if dataset[index] == "IP_headers" or dataset[index] ==  "IP_payloads" or dataset[index] ==  "IP_no_dst_headers":
        y_length = len(IP_labels)
        split_index = int(y_length * 0.7)
        label = IP_labels
        num_classes = len(set(IP_labels))

    elif dataset[index] == "Transport_payload":
        y_length = len(transport_labels)
        split_index = int(y_length * 0.7)
        label = transport_labels
        num_classes = len(set(transport_labels))

    elif dataset[index] == "Raw_data":
        y_length = len(raw_labels)
        split_index = int(y_length * 0.7)
        label = raw_labels
        num_classes = len(set(raw_labels))
    else:
        y_length = len(Whole_labels)
        split_index = int(y_length * 0.7)
        label = Whole_labels
        num_classes = len(set(Whole_labels))
    
    
    num_bytes = [list(bytes.fromhex(byte_string)) for byte_string in strategy]
    max_sequence_length = max(len(vec) for vec in num_bytes)
    print(max_sequence_length)
    padded_sequences = pad_sequences(num_bytes, maxlen=max_sequence_length, padding='post')
    desired_shape = largest_factor_pair(max_sequence_length)
    inputs = np.reshape(padded_sequences, (len(padded_sequences), *desired_shape))
    inputs = inputs.reshape(inputs.shape + (1,))

    x_first_70, x_last_30, first_70_labels, last_30_labels = train_test_split(inputs, label, test_size=0.3, stratify=label, random_state=42)

    # Calculate the y-indexs
    class_index_70 = []
    class_index_30 = []
    for value in list(MAC_list.keys()):
        j = [ind for ind, target in enumerate(first_70_labels) if target == value]
        class_index_70.append(j)
        i = [ind for ind, target in enumerate(last_30_labels) if target == value]
        class_index_30.append(i)


    tf_strategy = tf.distribute.MirroredStrategy()
    #2-D CNN

    with tf_strategy.scope():
        model = Sequential()

        # Block #1
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'))

        # Block #2
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))

        # Block #3
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2)))

        # Block #4
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(64, activation='relu', name="embed_dense_output"))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if index == 0:  
        tracker.start()
    model.fit(x_first_70, np.array(first_70_labels), epochs=2, batch_size=512)
    if index == 0:
        tracker.stop()
    model.predict(x_first_70[:4])
    print(model.summary())
    first_dense_output = model.get_layer("embed_dense_output").output
    first_dense_model = Model(inputs=model.layers[0].input, outputs=first_dense_output)
    seen_embeddings = first_dense_model.predict(x_first_70)
    unseen_embeddings = first_dense_model.predict(x_last_30)
    if index == 0:
        print(model.summary())
        labels = np.array(first_70_labels) 
        features_sample, _, labels_sample, _ = train_test_split(seen_embeddings, labels, test_size=0.985, stratify=labels, random_state=42)
        print(metrics.homogeneity_completeness_v_measure(labels_sample, AgglomerativeClustering(n_clusters=22).fit_predict(np.array(features_sample).astype(np.float16)))) 
        features_sample, _, labels_sample, _ = train_test_split(unseen_embeddings, np.array(last_30_labels), test_size=0.965, stratify=np.array(last_30_labels), random_state=42)
        print(metrics.homogeneity_completeness_v_measure(labels_sample , AgglomerativeClustering(n_clusters=22).fit_predict(np.array(features_sample).astype(np.float16))))
        similarity_calculator(seen_embeddings, class_index_70, dataset[index])
        similarity_calculator(unseen_embeddings, class_index_30, "UNSEEN")
    embeddings = np.concatenate((seen_embeddings, unseen_embeddings))
    write_vectors_to_tsv(file_name, embeddings.tolist(), first_70_labels+last_30_labels)
    index += 1
