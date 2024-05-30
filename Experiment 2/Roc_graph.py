import os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from collections import defaultdict
import numpy as np
import statistics
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from matplotlib import rcParams

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
inv_dict = {v: k for k, v in MAC_list.items()}
def find_tsv_files(directory):
    tsv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tsv"):
                tsv_files.append(os.path.join(root, file))
    return tsv_files

def file_to_list(file_path):
    with open(file_path, 'r') as file:
        content_list = [int(line.strip()) for line in file]
    return content_list

def load_tsv_file(file_path):
    vectors = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split('\t')
            # All values except the last one are part of the vector
            vector = [float(val) for val in values[0].strip('[]').split(', ')]
            # The last value is the label
            label = int(values[-1].strip('\n'))
            vectors.append(vector)
            labels.append(label)
    return np.array(vectors), labels

inputs = find_tsv_files('../Intrinsic')
print(inputs)
Whole_labels = file_to_list("../Embeddings/labels.txt")
IP_labels = file_to_list("../Embeddings/IP_labels.txt")
transport_labels = file_to_list("../Embeddings/transport_labels.txt")
raw_labels = file_to_list("../Embeddings/Raw_labels.txt")


original_dataset=pd.read_csv("../../First_Exp/IoTSense_FP_MAIN.csv")
Non_IoT = original_dataset[original_dataset['Label'].isin(['MacBook/Iphone', 'IPhone', 'Android Phone', 'MacBook', 'Laptop', 'unknown maybe cam', 'Samsung Galaxy Tab', 'TPLink Router Bridge LAN (Gateway)'])].index
original_dataset.drop(Non_IoT, inplace=True)
if "MAC" in original_dataset.columns:
    del original_dataset["MAC"]
X = original_dataset.loc[:, original_dataset.columns != "Label"]
Y = original_dataset["Label"].map(inv_dict)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
classify = OneVsRestClassifier(HistGradientBoostingClassifier(random_state=1), n_jobs=-1)

classify.fit(X_train, y_train)
y_pred = classify.predict(X_test)
y_proba = classify.predict_proba(X_test)

interest_class = 14
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

display = RocCurveDisplay.from_predictions(
    y_onehot_test[:, interest_class],
    y_proba[:, interest_class],
    name="Withings Smart Scale vs the rest",
    color="red",
    plot_chance_level=True,
)
display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
rcParams['pdf.fonttype'] = 42
plt.savefig('roc_curve_class_{}.pdf'.format("Withings Smart Scale"))
plt.clf()

main_datasets = ['../Intrinsic/LSTM_.tsv', '../Intrinsic/2-D CNN_.tsv', '../Intrinsic/Word2Vec_.tsv', '../Intrinsic/Compression_.tsv', '../Intrinsic/1-D CNN_.tsv', '../Intrinsic/Compression_.tsv', '../Intrinsic/Transformer_.tsv']
index = 0
names = [' LSTM', ' 2-D CNN', ' Word2Vec', ' Compression', ' 1-D CNN', ' Compression', ' Transformer']
for input in inputs:
    roc_auc_scores = defaultdict(list)
    precision_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    class_accuracy_scores = defaultdict(list)
    if input not in main_datasets:
        continue
    if "IP" in input:
        labels = IP_labels
    elif "transport" in input:
        labels = transport_labels
    elif "raw" in input:
        labels = raw_labels
    else:
        labels = Whole_labels
    vectors, lab = load_tsv_file(input)
    x_length = len(vectors)
    n = os.path.basename(input).split('_')[0]
    print("#############################################")
    print(n)
    print("#############################################")
    classify = OneVsRestClassifier(HistGradientBoostingClassifier(random_state=1), n_jobs=-1)
    split_index = int(0.7 * x_length)
    X_train, X_test = vectors[:split_index], vectors[split_index:]
    y_train, y_test = lab[:split_index], lab[split_index:]
    classify.fit(X_train, y_train)
    y_pred = classify.predict(X_test)
    y_proba = classify.predict_proba(X_test)
    
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, interest_class],
        y_proba[:, interest_class],
        name="Withings Smart Scale vs the rest" ,
        color="green",
        plot_chance_level=True,
    )
    display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    rcParams['pdf.fonttype'] = 42
    plt.savefig('roc_curve_class_{}.pdf'.format(n+"_Withings Smart Scale"))
    plt.clf()
    index += 1 

       