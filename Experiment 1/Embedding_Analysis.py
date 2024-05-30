import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import rcParams

cl = ["Smart Things", "Amazon Echo", "Netatmo Welcome", "TP-Link Day Night Cloud camera", "Samsung SmartCam", "Dropcam", "Insteon Camera", "Withings Smart Baby Monitor", "Belkin Wemo switch",
"TP-Link Smart plug", "iHome", "Belkin wemo motion sensor", "NEST Protect smoke alarm", "Netatmo weather station", "Withings Smart scale", "Blipcare Blood Pressure meter",
"Withings Aura smart sleep sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "PIX-STAR Photo-frame", "HP Printer", "Nest Dropcam"]

MAC_list={
0:"Smart Things",
1:"Amazon Echo",
2:"Netatmo Welcome",
3:"TP-Link Day Night Cloud camera",
4:"Samsung SmartCam",
5:"Dropcam",
6:"Insteon Camera",
7:"Withings Smart Baby Monitor",
8:"Belkin Wemo Switch",
9:"TP-Link Smart Plug",
10:"iHome",
11:"Belkin Wemo Motion Sensor",
12:"NEST Protect Smoke Alarm",
13:"Netatmo Weather Station",
14:"Withings Smart Scale",
15:"Blipcare Blood Pressure Meter",
16:"Withings Aura Smart Sleep Sensor",
17:"Light Bulbs LiFX Smart Bulb",
18:"Triby Speaker",
19:"PIX-STAR Photo-Frame",
20:"HP Printer",
21:"Nest Dropcam"
}


mini_cl = ["Amazon Echo", "Belkin Wemo Motion Sensor", "Samsung SmartCam", "Dropcam", "Insteon Camera", "Netatmo Weather Station", "HP Printer"]
# Replace these file paths with which tsv you want to analyse
vectors_file = ['Word2Vec_.tsv','LSTM_.tsv']
# index = 0
for vector_file in vectors_file:
    if 'Byte.tsv' in vector_file:
        n = vector_file.split('.')[0]
        less = True
    else:
        n = vector_file.split('_')[0]
        less = False
    output_vectors_file = '../TSVs/selected_'+n+'.tsv'
    output_labels_file = '../TSVs/selected_labels_'+n+'.tsv'

    # Read vectors and labels files
    vectors_df = pd.read_csv(vector_file, sep='\t', header=None)
    vectors_df.iloc[:, -1] = vectors_df.iloc[:, -1].map(MAC_list)
    selected_data_df = pd.DataFrame()
    for class_label in mini_cl:  
        class_instances = vectors_df[vectors_df.iloc[:, -1] == class_label]
        if len(class_instances) > 5000:
            class_instances = class_instances.sample(n=5000, random_state=42)
        selected_data_df = pd.concat([selected_data_df, class_instances], ignore_index=True, sort=False)


    # Separate vectors and labels
    selected_vectors = selected_data_df.iloc[:, :-1]  # Assuming the last column is the label
    selected_labels = selected_data_df.iloc[:, -1]  # Assuming the last column is the label
    if not less:
        selected_vectors.iloc[:, 0] = selected_vectors.iloc[:, 0].str.strip('[]')
        selected_vectors = selected_vectors.iloc[:, 0].str.split(',', expand=True)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(selected_vectors)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['label'] = selected_labels.values

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)

    fig, ax = plt.figure(figsize=(16,14), dpi=300)
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='label',
        palette=sns.hls_palette(len(tsne_df['label'].unique()), s=0.9, l=0.5),
        legend=False,
        data=tsne_df,
        alpha=0.6
    )
    #if index % 2 == 0:
    #    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=18)
    #index += 1     
    rcParams['pdf.fonttype'] = 42
    plt.savefig(n+'_2D_tsne_projection.pdf', dpi=300, bbox_inches="tight")
    plt.clf()
    
    '''
    # 3D
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(selected_vectors)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    tsne_df['label'] = selected_labels.values
    unique_labels = tsne_df['label'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))
    tsne_df['color'] = tsne_df['label'].map(color_map)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        tsne_df['TSNE1'], 
        tsne_df['TSNE2'], 
        tsne_df['TSNE3'], 
        c=tsne_df['color'], 
        alpha=0.7
    )

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label)
                    for label, color in color_map.items()]

    ax.legend(handles=legend_elements)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(n+'_3D_tsne_projection.png', dpi=300, bbox_inches="tight")
    plt.clf()

    for class_label in MAC_list.values():  
        class_instances = vectors_df[vectors_df.iloc[:, -1] == class_label]
        if not less:
            if len(class_instances) > 1500:
                class_instances = class_instances.sample(n=1500, random_state=42)
        else:
            if len(class_instances) > 100:
                class_instances = class_instances.sample(n=200, random_state=42)
        selected_data_df = pd.concat([selected_data_df, class_instances], ignore_index=True, sort=False)

    selected_vectors = selected_data_df.iloc[:, :-1]  # Assuming the last column is the label
    selected_labels = selected_data_df.iloc[:, -1]  # Assuming the last column is the label

    if not less:
        selected_vectors.iloc[:, 0] = selected_vectors.iloc[:, 0].str.strip('[]')
        selected_vectors = selected_vectors.iloc[:, 0].str.split(',', expand=True)

    # Write selected vectors and labels to new TSV files
    selected_vectors.to_csv(output_vectors_file, sep='\t', index=False, header=False)
    selected_labels.to_csv(output_labels_file, sep='\t', index=False, header=False)
    '''