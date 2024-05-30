from scapy.all import *
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

def list_to_file(input_list, file_path):
    with open(file_path, 'w') as file:
        for item in input_list:
            file.write(str(item) + '\n')
    print(f"List has been written to {file_path}")

def write_list_of_lists_to_file(data, filename):
    with open(filename, 'w') as file:
        for inner_list in data:
            line = ','.join(map(str, inner_list))  # Convert inner list elements to strings and join with commas
            file.write(line + '\n')  # Write the line to the file

def find_pcap_files(directory):
    pcap_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

pcap_dir = '../data/'
files = find_pcap_files(pcap_dir)
print(files)
raw_data = []
raw_labels = []
transport_payload = []
transport_labels = []
IP_payloads = []
IP_labels = []
labels = []
eth_payloads = []
eth_no_dst_payloads = []
whole_packet = []
for pcap_file in files:
    # Convert byte string to characters
    print(pcap_file)
    for byte_string in PcapReader(pcap_file):
        if byte_string.haslayer(Ether) and byte_string[Ether].src in Class_list:
            labels.append(Class_list[byte_string[Ether].src])

            if byte_string.haslayer(Raw):
                raw_labels.append(Class_list[byte_string[Ether].src])
                raw_data.append(bytes(byte_string[Raw].load).hex(' '))

            if byte_string.haslayer(TCP):
                if len(byte_string[TCP].payload) != 0:
                    transport_labels.append(Class_list[byte_string[Ether].src])
                    transport_payload.append(bytes(byte_string[TCP].payload).hex(' '))
            if byte_string.haslayer(UDP):
                if len(byte_string[UDP].payload) != 0:
                    transport_labels.append(Class_list[byte_string[Ether].src])
                    transport_payload.append(bytes(byte_string[UDP].payload).hex(' '))

            if byte_string.haslayer(IP):
                IP_labels.append(Class_list[byte_string[Ether].src])

                IP_payload = bytes(byte_string[IP].payload).hex(' ')

                IP_payloads.append(IP_payload)

                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:78]+bytes(byte_string).hex(' ')[90:]
                no_dst_payload = bytes(byte_string.payload).hex(' ')[:36]+bytes(byte_string.payload).hex(' ')[60:]
                payload = bytes(byte_string.payload).hex(' ')[:36]+bytes(byte_string.payload).hex(' ')[48:]

                whole_packet.append(string)
                eth_payloads.append(payload)
                eth_no_dst_payloads.append(no_dst_payload)
            
            elif byte_string.haslayer(ICMPv6ND_RS):
                IP_labels.append(Class_list[byte_string[Ether].src])

                IP_payload = bytes(byte_string[IPv6].payload).hex(' ')[:24]

                IP_payloads.append(IP_payload)

                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:66]+bytes(byte_string).hex(' ')[114:234]
                no_dst_payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[120:144]
                payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[72:192]

                whole_packet.append(string)
                eth_payloads.append(payload)
                eth_no_dst_payloads.append(no_dst_payload)
            
            elif byte_string.haslayer(ICMPv6ND_NS) or byte_string.haslayer(ICMPv6ND_NA):
                IP_labels.append(Class_list[byte_string[Ether].src])

                IP_payload = bytes(byte_string[IPv6].payload).hex(' ')[:72]

                IP_payloads.append(IP_payload)

                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:66]+bytes(byte_string).hex(' ')[114:186]
                no_dst_payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[120:144]
                payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[72:144]

                whole_packet.append(string)
                eth_payloads.append(payload)
                eth_no_dst_payloads.append(no_dst_payload)

            elif byte_string.haslayer(IPv6):
                IP_labels.append(Class_list[byte_string[Ether].src])

                IP_payload = bytes(byte_string[IPv6].payload).hex(' ')

                IP_payloads.append(IP_payload)

                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:66]+bytes(byte_string).hex(' ')[114:]
                no_dst_payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[120:]
                payload = bytes(byte_string.payload).hex(' ')[:24]+bytes(byte_string.payload).hex(' ')[72:]

                whole_packet.append(string)
                eth_payloads.append(payload)
                eth_no_dst_payloads.append(no_dst_payload)
                
            elif byte_string.haslayer(ARP):
                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:66]+bytes(byte_string).hex(' ')[96:]
                no_dst_string = bytes(byte_string.payload).hex(' ')[:24]
        
                whole_packet.append(string)
                eth_payloads.append(string)
                eth_no_dst_payloads.append(no_dst_string)
            else:
                string = bytes(byte_string).hex(' ')[:18]+bytes(byte_string).hex(' ')[36:]
                payload = bytes(byte_string.payload).hex(' ')
        
                whole_packet.append(string)
                eth_payloads.append(payload)
                eth_no_dst_payloads.append(payload)


write_list_of_lists_to_file([whole_packet, eth_payloads, eth_no_dst_payloads, IP_payloads, transport_payload, raw_data], "byte_dataset.txt")
if not os.path.exists("labels.txt"):
    list_to_file(labels, "labels.txt")
if not os.path.exists("IP_labels.txt"):
    list_to_file(IP_labels, "IP_labels.txt")
if not os.path.exists("transport_labels.txt"):
    list_to_file(transport_labels, "transport_labels.txt")
if not os.path.exists("Raw_labels.txt"):
    list_to_file(raw_labels, "Raw_labels.txt")