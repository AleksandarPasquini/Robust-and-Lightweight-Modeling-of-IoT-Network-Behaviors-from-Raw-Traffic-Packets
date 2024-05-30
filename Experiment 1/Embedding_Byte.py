import numpy as np
import tensorflow as tf
import csv
from scapy.all import *
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
        for vector, label in zip(vectors, labels):
            row = list(vector) + [label]
            writer.writerow(row)


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

def pairwise_hamming_distance(arr):
    arr = np.array(arr)
    num_samples = arr.shape[0]
    distances = np.zeros((num_samples, num_samples), dtype=int)

    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                distance = hamming_distance(arr[i], arr[j])
                distances[i][j] = distance

    return distances

def hamming_distance(s1, s2):
    # Ensure the bit streams are of equal length
    if len(s1) != len(s2):
        raise ValueError("Bit streams must have equal length")
    
    # Convert to integer type if necessary
    s1 = np.asarray(s1, dtype=int)
    s2 = np.asarray(s2, dtype=int)

    # Calculate Hamming distance
    distance = sum(bin(byte1 ^ byte2).count('1') for byte1, byte2 in zip(s1, s2))
    return distance/len(s1)

def similarity_distance(vectors, class_index, Technique):
    print("#############################################")
    print(Technique)
    print("#############################################")
    avg_similarities_per_class = []
    limit = 10000
    for j, sim_class in enumerate(class_index):
        name = MAC_list[j]
        target_vectors = [vectors[i] for i in sim_class]
        if len(target_vectors) > limit:
            target_vectors = random.sample(target_vectors, limit)
        if target_vectors != []:
            similarity_matrix = pairwise_hamming_distance(target_vectors)
            avg_similarity = np.mean(similarity_matrix)
            avg_similarities_per_class.append(avg_similarity)
            print(f"Class {name} Average Similarity: {avg_similarity}")
        else:
            print(f"Class {name} has no vectors")
    
    avg_similarity_across_all_classes = np.mean(avg_similarities_per_class)
    print(f"Average Similarity across all classes: {avg_similarity_across_all_classes}")

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
labels = file_to_list("../Embeddings/labels.txt")
index = 0
dataset = ["Whole packet"]

for strategy in inputs:
    # --- Character-based Embeddings ---
    
    # Calculate the y-indexs
    num_classes = len(set(labels))
    class_index = []
    split_index = int(len(labels) * 0.7)
    for value in list(MAC_list.keys()):
        j = [ind for ind, target in enumerate(labels) if target == value]
        class_index.append(j)
    if not os.path.exists("metadata.tsv"):
        with open("metadata.tsv", 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            #writer.writerow("Device_Type")
            for value in labels:
                writer.writerow([MAC_list[value]]) 
    
    
    # Byte values
    Technique = "Byte"
    list_bytes = [list(bytes.fromhex(byte_string)) for byte_string in strategy]
    max_sequence_length = max(len(vector) for vector in list_bytes)
    print(max_sequence_length)
    embeddings = np.array(pad_sequences(list_bytes, maxlen=max_sequence_length, padding='post'))
    labels = np.array(labels) 
    features_sample, _, labels_sample, _ = train_test_split(embeddings, labels, test_size=0.99, stratify=labels, random_state=42)
    print(metrics.homogeneity_completeness_v_measure(labels_sample, AgglomerativeClustering(n_clusters=22).fit_predict(np.array(features_sample).astype(np.float16)))) 
    similarity_calculator(embeddings, class_index, Technique)
    write_vectors_to_tsv(Technique, embeddings, labels)
    break
    #similarity_distance(seen_embeddings, class_index_90, Technique)
    #similarity_distance(unseen_embeddings, class_index_10, "UNSEEN")
    