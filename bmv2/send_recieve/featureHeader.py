from scapy.all import *

TYPE_FeatureHeader = 9999

class FeatureHeader(Packet):
    name = "FeatureHeader"
    fields_desc = [
        ByteField("ip_len", 0),
        ShortField("ip_total_len", 0),
        ByteField("protocol", 0),
        ByteField("tos", 0),
        ShortField("offset", 0),
        IntField("max_byte", 0),
        ShortField("min_byte", 0),
        IntField("max_ipd", 0),
        IntField("min_ipd", 0),
        IntField("ipd", 0)
    ]
    def mysummary(self):
        return self.sprintf("ip_len=%ip_len%, ip_total_len=%ip_total_len%, protocol=%protocol%, tos=%tos%, offset=%offset%, max_byte=%max_byte%, min_byte=%min_byte%, max_ipd=%max_ipd%, min_ipd=%min_ipd%, ipd=%ipd%")
class Label(Packet):
    name = "Label"
    fields_desc = [
        IntField("label", 5)
    ]
bind_layers(TCP, FeatureHeader, dport = TYPE_FeatureHeader)
bind_layers(FeatureHeader, Label)

