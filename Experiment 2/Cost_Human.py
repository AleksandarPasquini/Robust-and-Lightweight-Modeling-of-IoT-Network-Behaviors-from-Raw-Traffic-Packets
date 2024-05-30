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
from memory_profiler import profile

from codecarbon import OfflineEmissionsTracker
import time
import resource
import os

Class_list={
"d0:52:a8:00:67:5e":0,
"44:65:0d:56:cc:d3":1,
"70:ee:50:18:34:43":2,
"f4:f2:6d:93:51:f1":3,
"00:16:6c:ab:6b:88":4,
"30:8c:fb:2f:e4:b2":5,
"00:62:6e:51:27:2e":6,
"e8:ab:fa:19:de:4f":6,
"00:24:e4:11:18:a8":7,
"ec:1a:59:79:f4:89":8,
"50:c7:bf:00:56:39":9,
"74:c6:3b:29:d7:1d":10,
"ec:1a:59:83:28:11":11,
"18:b4:30:25:be:e4":12,
"70:ee:50:03:b8:ac":13,
"00:24:e4:1b:6f:96":14,
"74:6a:89:00:2e:25":15,
"00:24:e4:20:28:c6":16,
"d0:73:d5:01:83:08":17,
"18:b7:9e:02:20:44":18,
"e0:76:d0:33:bb:85":19,
"70:5a:0f:e4:9b:c0":20,
#"08:21:ef:3b:fc:e3":"Samsung Galaxy Tab",
"30:8c:fb:b6:ea:45":21,
#"40:f3:08:ff:1e:da":"Android Phone",
#"74:2f:68:81:69:42":"Laptop",
#"ac:bc:32:d4:6f:2f":"MacBook",
#"b4:ce:f6:a7:a3:c2":"Android Phone",
#"d0:a6:37:df:a1:e1":"IPhone",
#"f4:5c:89:93:cc:85":"MacBook/Iphone",
#"14:cc:20:51:33:ea":"TPLink Router Bridge LAN (Gateway)"
}

def filter_packets_by_mac(pcap_file, max_count=10000):
    filtered_packets = []
    for pkts in PcapReader(pcap_file):
        if pkts.haslayer(Ether) and pkts[Ether].src in Class_list:
            filtered_packets.append(pkts)
            # Stop if the list reaches 10,000 packets
            if len(filtered_packets) >= max_count:
                break

    return filtered_packets

FIN = 0x01
ACK = 0x10
#IP
DF= 0x02

IP_fl = {'0': 1, ' ': 2, 'DF': 3, 'MF': 4, '': 1}
BOOTP_fl = {'0': 1, ' ': 2, 'B': 3, '': 1}
def shannon(data):
    LOG_BASE = 2
   # We determine the frequency of each byte
   # in the dataset and if this frequency is not null we use it for the
   # entropy calculation
    dataSize = len(data)
    ent = 0.0
    freq={} 
    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
   # to determine if each possible value of a byte is in the list
    for key in freq.keys():
        f = float(freq[key])/dataSize
        if f > 0: # to avoid an error for log(0)
            ent = ent + f * math.log(f, LOG_BASE)
    return -ent

def pre_entropy(payload):
    
    characters=[]
    for i in payload:
            characters.append(i)
    return shannon(characters)
            

def port_class(port):
    port_list=[0,53,67,68,80,123,443,1900,5353,49153]# private port list (0-Reserved,53-DNS, 67-BOOTP server, 68-BOOTP client...)
    if port in port_list: #Is the port number in the list?
        return port_list.index(port)+1 # return the port's index number in the list (actually with index+1)
    elif 0 <= port <= 1023: # return 11 if the port number is in the range 0 :1023
        return 11
    elif  1024 <= port <= 49151 : # return 12 if the port number is in the range 1024:49151
        return 12
    elif 49152 <=port <= 65535 :# return 13 if the port number is in the range 49152:65535
        return 13
    else:# return 0 if no previous conditions are met
        return 0
    
    
def port_1023(port):
    if 0 <= port <= 1023:
        return port
    elif  1024 <= port <= 49151 :
        return 2
    elif 49152 <=port <= 65535 :
        return 3
    else:
        return 0

def find_pcap_files(directory):
    pcap_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

@profile
def predict_embeddings(strategy):
    embeddings = []
    for j in strategy:
            if j.haslayer(ARP):
                embeddings.append([0] * 30)
            else:
                try:pck_size=j.len
                except:pck_size=0
                if j.haslayer(Ether):
                    Ether_type=j[Ether].type
                else:
                    Ether_type=0    


                if j.haslayer(LLC):
                    LLC_ctrl=j[LLC].ctrl
                else:
                    LLC_ctrl=0            



                if j.haslayer(EAPOL):
                    EAPOL_version=j[EAPOL].version
                    EAPOL_type=j[EAPOL].type

                else:
                    EAPOL_version=0
                    EAPOL_type=0          


                if j.haslayer(IP):
                    IP_DF= 0
                    IP_ihl=j[IP].ihl
                    IP_tos=j[IP].tos
                    IP_len=j[IP].len
                    IP_flags=j[IP].flags
                    IP_ttl=j[IP].ttl
                    IP_options=j[IP].options
                    if "IPOption_Router_Alert" in str(IP_options):
                        IP_options=1
                    else: IP_options=0

                    if IP_flags & DF:IP_DF = 1 


                else:
                    IP_DF= 0
                    IP_ihl=0
                    IP_tos=0
                    IP_len=0
                    IP_flags=0
                    IP_ttl=0
                    IP_options=0     

                if j.haslayer(ICMP):
                    ICMP_code=j[ICMP].code
                else:
                    ICMP_code=0




                if j.haslayer(TCP):
                    TCP_FIN = 0
                    TCP_ACK = 0
                    TCP_dport=j[TCP].dport
                    TCP_dataofs=j[TCP].dataofs
                    TCP_window=j[TCP].window     
                    TCP_flags = j[TCP].flags
                    if TCP_flags & FIN: TCP_FIN = 1
                    if TCP_flags & ACK: TCP_ACK = 1


                else:
                    TCP_dport=0
                    TCP_dataofs=0
                    TCP_window=0
                    TCP_FIN = 0
                    TCP_ACK = 0


                if j.haslayer(UDP):
                    UDP_len=j[UDP].len
                    UDP_dport=j[UDP].dport
                else:
                    UDP_len=0
                    UDP_dport=0

                if j.haslayer(DHCP):
                    DHCP_options=str(j[DHCP].options)
                    DHCP_options=DHCP_options.replace(",","-")
                    if "message" in DHCP_options:
                        x = DHCP_options.find(")")
                        DHCP_options=int(DHCP_options[x-1])
                        
                else:
                    DHCP_options=0            


                if j.haslayer(BOOTP):
                    BOOTP_hlen=j[BOOTP].hlen
                    BOOTP_flags=j[BOOTP].flags

                    BOOTP_sname=str(j[BOOTP].sname)
                    if BOOTP_sname!="0":
                        BOOTP_sname=1
                    else:
                        BOOTP_sname=0
                    BOOTP_file=str(j[BOOTP].file)
                    if BOOTP_file!="0":
                        BOOTP_file=1
                    else:
                        BOOTP_file=0
                    BOOTP_options=str(j[BOOTP].options)
                    BOOTP_options=BOOTP_options.replace(",","-")
                    if BOOTP_options!="0":
                        BOOTP_options=1
                    else:
                        BOOTP_options=0
                else:
                    BOOTP_hlen=0
                    BOOTP_flags=0
                    BOOTP_sname=0
                    BOOTP_file=0
                    BOOTP_options=0

                if j.haslayer(DNS):
                    DNS_qr=j[DNS].qr
                    DNS_rd=j[DNS].rd
                    DNS_qdcount=j[DNS].qdcount
                else:
                    DNS_qr=0
                    DNS_rd=0
                    DNS_qdcount=0

                pdata=[]
                if "TCP" in j:            
                    pdata = (j[TCP].payload)
                if "Raw" in j:
                    pdata = (j[Raw].load)
                elif "UDP" in j:            
                    pdata = (j[UDP].payload)
                elif "ICMP" in j:            
                    pdata = (j[ICMP].payload)
                pdata=list(memoryview(bytes(pdata)))            
        
                if pdata!=[]:
                    entropy=shannon(pdata)        
                else:
                    entropy=0
                payload_bytes=len(pdata)

                dport_class=port_class(TCP_dport+UDP_dport)
            
                embeddings.append([pck_size,
                Ether_type,
                LLC_ctrl,
                EAPOL_type,
                EAPOL_version,
                IP_ihl,
                IP_tos,
                IP_len,
                IP_fl[str(IP_flags)],
                IP_DF,
                IP_ttl,
                IP_options,
                ICMP_code,
                TCP_dataofs,
                TCP_FIN,
                TCP_ACK, 
                TCP_window,
                UDP_len,
                DHCP_options,
                BOOTP_hlen,
                BOOTP_fl[str(BOOTP_flags)],
                BOOTP_sname,
                BOOTP_file,
                BOOTP_options,
                DNS_qr,
                DNS_rd,
                DNS_qdcount,
                payload_bytes,
                dport_class,
                entropy,
                ])
    print(len(embeddings))            
    return embeddings

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
print("Human")
print("#############################################")
for pcap_file in find_pcap_files('../data/'):
    result_packets = filter_packets_by_mac(pcap_file)
    if len(result_packets) > 10000:
        break
embedding = predict_embeddings(result_packets)
classify = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)
classify.fit(embedding, labels[0:10000]) 
tracker.start()
start_time = time.time()
embeddings = predict_embeddings(result_packets)
end_time = time.time()
classify.predict(embeddings)
end_time2 = time.time()
tracker.stop()
execution_time = end_time - start_time
inference_time = end_time2 - start_time
print(f"The embedding generation time of the code block is {execution_time} seconds")
print(f"The inference time of the code block is {inference_time} seconds")

