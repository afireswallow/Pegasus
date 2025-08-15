from scapy.all import *

TYPE_FeatureHeader = 9999

class SignedIntField(IntField):
    def i2repr(self, pkt, x):
        t = (x >> 31) & 1
        x = (x & 0x7fffffff) - (t << 31)
        return f"{x}"

class ResultHeader(Packet):
    name = "ResultHeader"
    fields_desc = [
        SignedIntField("l32_1", 0),
        SignedIntField("l32_2", 0),
        SignedIntField("l32_3", 0),
        IntField("l32_4", 0)
    ]
    def mysummary(self):
        return self.sprintf("l32_1=%l32_1%, l32_2=%l32_2%, l32_3=%l32_3%, l32_4=%l32_4%")
    

class ReservedResultHeader(Packet):
    name = "ReservedResultHeader"
    fields_desc = [
        IntField("l32_5", 0),
        IntField("l32_6", 0),
        IntField("l32_7", 0),
        IntField("l32_8", 0)
    ]
    def mysummary(self):
        return self.sprintf("l32_5=%l32_5%, l32_6=%l32_6%, l32_7=%l32_7%, l32_8=%l32_8%")


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
bind_layers(FeatureHeader, ResultHeader)
# bind_layers(ResultHeader, ReservedResultHeader)
# bind_layers(ReservedResultHeader, Label)
bind_layers(ResultHeader, Label)  # Bind directly to Label, skip ReservedResultHeader


