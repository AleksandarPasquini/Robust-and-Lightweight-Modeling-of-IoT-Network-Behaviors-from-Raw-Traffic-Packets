import scapy.all as scapy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np

MAC_list={
"d0:52:a8:00:67:5e":"Smart Things",
"44:65:0d:56:cc:d3":"Amazon Echo",
"70:ee:50:18:34:43":"Netatmo Welcomes",
"f4:f2:6d:93:51:f1":"TP-Link Day Night Cloud camera",
"00:16:6c:ab:6b:88":"Samsung SmartCam",
"30:8c:fb:2f:e4:b2":"Dropcam",
"00:62:6e:51:27:2e":"Insteon Camera",
"e8:ab:fa:19:de:4f":"Insteon Camera",
"00:24:e4:11:18:a8":"Withings Smart Baby Monitor",
"ec:1a:59:79:f4:89":"Belkin Wemo switch ",
"50:c7:bf:00:56:39":"TP-Link Smart plug",
"74:c6:3b:29:d7:1d":"iHome",
"ec:1a:59:83:28:11":"Belkin wemo motion sensor",
"18:b4:30:25:be:e4":"NEST Protect smoke alarm",
"70:ee:50:03:b8:ac":"Netatmo weather station",
"00:24:e4:1b:6f:96":"Withings Smart scale ",
"74:6a:89:00:2e:25":"Blipcare Blood Pressure Meter",
"00:24:e4:20:28:c6":"Withings Aura smart sleep sensor",
"d0:73:d5:01:83:08":"Light Bulbs LiFX Smart Bulb",
"18:b7:9e:02:20:44":"Triby Speaker",
"e0:76:d0:33:bb:85":"PIX-STAR Photo-Frame ",
"70:5a:0f:e4:9b:c0":"HP Printer",
"30:8c:fb:b6:ea:45":"Nest Dropcam",
}

# Function to read packets from a pcap file
def read_pcap_first_hour(file_name):
    packets = scapy.rdpcap(file_name)
    first_packet_time = packets[0].time
    one_hour_later = first_packet_time + 3600  # One hour later in seconds
    first_hour_packets = []
    for packet in packets:
        if packet.time <= one_hour_later:
            first_hour_packets.append(packet)
        else:
            break
    return first_hour_packets

# Function to classify packets according to the TCP/IP model
def classify_packet(packet):
    labels = []
    if scapy.Raw in packet:
        labels.append("Raw Data")
    if scapy.IP in packet or scapy.IPv6 in packet:
        labels.append("Network (IP/IPv6) Layer")
    if scapy.TCP in packet and len(packet[scapy.TCP].payload) != 0:
        labels.append("Transport (TCP/UDP) Layer")
    elif scapy.UDP in packet and len(packet[scapy.UDP].payload) != 0:
        labels.append("Transport (TCP/UDP) Layer")
    labels.append("Whole Packet")
    return labels

# Function to plot packets over time
def plot_packets(packets, mac_addresses):
    fig, ax = plt.subplots(figsize=(20, 10))
    label_order = ["Raw Data", "Transport (TCP/UDP) Layer", "Network (IP/IPv6) Layer", "Whole Packet",]
    colors = plt.cm.tab10(range(len(label_order)))   # Distinct colors for each label

    data = {mac: {label: [] for label in label_order} for mac in mac_addresses}
    legend_added = set()

    for packet in packets:
        if scapy.Ether in packet and packet[scapy.Ether].src in mac_addresses:
            timestamp = float(packet.time)
            packet_labels = classify_packet(packet)
            src_mac = packet[scapy.Ether].src
            for label in packet_labels:
                if label in data[src_mac]:
                    data[src_mac][label].append(timestamp)

    for i, mac in enumerate(mac_addresses):
        for j, label in enumerate(label_order):
            if data[mac][label]:
                times = pd.to_datetime(data[mac][label], unit='s')
                plot_label = label if label not in legend_added else ""
                ax.plot(times, [i * len(label_order) + j + 1] * len(times),
                        label=plot_label, marker='o', linestyle='', color=colors[j])
                legend_added.add(label)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    #yticks = [i + 1 + 4 * j for j in range(len(mac_addresses)) for i in range(len(label_order))]
    yticklabels = [f"{MAC_list[mac]}" for mac in mac_addresses]
    mid_points = [i * len(label_order) + 2.5 for i in range(len(mac_addresses))]
    plt.yticks(ticks=mid_points, labels=yticklabels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=True)
    #plt.yticks(ticks=yticks, labels=yticklabels)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({'font.size': 32})
    plt.savefig("Packets_Over_Time.pdf", bbox_inches="tight")




file_name = '16-09-23.pcap'  
mac_addresses = ["44:65:0d:56:cc:d3", "e0:76:d0:33:bb:85", "70:5a:0f:e4:9b:c0", "74:6a:89:00:2e:25",  "30:8c:fb:2f:e4:b2"]
packets = read_pcap_first_hour(file_name)
plot_packets(packets, mac_addresses)

