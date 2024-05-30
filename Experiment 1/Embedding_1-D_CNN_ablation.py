import numpy as np
import tensorflow as tf
import csv
from scapy.all import *
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
random.seed(42)

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


def create_model(embedding_dim, kernel_size, filters, add_activation=None, remove_conv=False):
    tf_strategy = tf.distribute.MirroredStrategy()
    with tf_strategy.scope():
        model = Sequential()
        model.add(Embedding(256, embedding_dim))
        if not remove_conv:
            if add_activation:
                model.add(Conv1D(filters, kernel_size=kernel_size, activation=add_activation))
            else:
                model.add(Conv1D(filters, kernel_size=kernel_size))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, name="first_dense_output"))
        model.add(Dense(22, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


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
labels = file_to_list("../Embeddings/labels.txt")
index = 0
dataset = ["Whole packet"]

embedding_dim = 64

models = {
    "base_model": create_model(embedding_dim, 15, 128),
    "kernel_size_12": create_model(embedding_dim, 12, 128),
    "kernel_size_18": create_model(embedding_dim, 18, 128),
    "kernel_size_21": create_model(embedding_dim, 21, 128),
    "kernel_size_3": create_model(embedding_dim, 3, 128),
    "kernel_size_5": create_model(embedding_dim, 5, 128),
    "kernel_size_9": create_model(embedding_dim, 9, 128),
    "kernel_size_7": create_model(embedding_dim, 7, 128),
    "embedding_dim_32": create_model(32, 15, 128),
    "embedding_dim_128": create_model(128, 15, 128),
    "filters_96": create_model(embedding_dim, 15, 96),
    "filters_256": create_model(embedding_dim, 15, 256),
    "activation_relu": create_model(embedding_dim, 15, 128, add_activation='relu'),
    "activation_sig": create_model(embedding_dim, 15, 128, add_activation='sigmoid'),
    "activation_swish": create_model(embedding_dim, 15, 128, add_activation='swish'),
    "no_convolution": create_model(embedding_dim, 15, 128, remove_conv=True)
}


# --- Character-based Embeddings ---

# Calculate the y-indexs
num_classes = len(set(labels))
total_elements = len(labels)
    
characters = [list(bytes.fromhex(byte_string)) for byte_string in inputs[0]]
max_sequence_length = max(len(vec) for vec in characters)
print(max_sequence_length)
padded_sequences = pad_sequences(characters, maxlen=max_sequence_length, padding='post')
x_first_70, x_last_30, first_70_labels, last_30_labels = train_test_split(padded_sequences, labels, test_size=0.3, stratify=labels, random_state=42)

# Calculate the y-indexs
class_index_70 = []
class_index_30 = []
for value in list(MAC_list.keys()):
    j = [ind for ind, target in enumerate(first_70_labels) if target == value]
    class_index_70.append(j)
    i = [ind for ind, target in enumerate(last_30_labels) if target == value]
    class_index_30.append(i)
    # 1-D CNN

for name, model in models.items():
    model.fit(x_first_70, np.array(first_70_labels), epochs=2, batch_size=512)
    first_dense_output = model.get_layer("first_dense_output").output
    first_dense_model = Model(inputs=model.layers[0].input, outputs=first_dense_output)
    embeddings = first_dense_model.predict(padded_sequences)
    seen_embeddings = first_dense_model.predict(x_first_70)
    unseen_embeddings = first_dense_model.predict(x_last_30)
    similarity_calculator(seen_embeddings, class_index_70, name)
    similarity_calculator(unseen_embeddings, class_index_30, "UNSEEN")