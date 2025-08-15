/* -*- P4_16 -*- */

#ifndef _HEADERS_
#define _HEADERS_

typedef bit<48> mac_addr_t;
typedef bit<32> ipv4_addr_t;
typedef bit<128> ipv6_addr_t;
typedef bit<12> vlan_id_t;


typedef bit<16> ether_type_t;
const ether_type_t ETHERTYPE_IPV4 = 16w0x0800;
const ether_type_t ETHERTYPE_ARP = 16w0x0806;
const ether_type_t ETHERTYPE_IPV6 = 16w0x86dd;
const ether_type_t ETHERTYPE_VLAN = 16w0x8100;

typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_ICMP = 1;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;

// #define egress_port 0x120

#define Register_Index_Size 8    //register array index bits
#define Register_Table_Size (1 << Register_Index_Size) //register array size
#define WIDTH 32
#define size_of_pkt_count_mod_7 8


header ethernet_h {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    ipv4_addr_t src_addr;
    ipv4_addr_t dst_addr;
}

header tcp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4>  data_offset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
}

struct pkt_t {
    bit<32> l32_1;
    bit<32> l32_2;
}

struct metadata_x_t {
    bit<16> l16_1;
    bit<16> l16_2;
    bit<16> l16_3;
    bit<16> l16_4; 
}

struct metadata_linear_t {
    bit<32> l32_1;
    bit<32> l32_2;
    bit<32> l32_3;
    bit<32> l32_4;
}

struct metadata_temp_t {
    bit<32> l32_1;
}
struct metadata_short_t {
    bit<16> l16_1;
    bit<16> l16_2;
}
struct metadata_temp4_t {
    bit<8> l8_1;
    bit<8> l8_2;
    bit<8> l8_3;
    bit<8> l8_4;
}
header feature_h {
    bit<16> pkl;
    bit<16> ipd;
}

header output_h {
        // metadata_x_t x_1;
        // metadata_x_t x_2;
        // metadata_x_t x_3;
        // metadata_x_t x_4;
        // metadata_x_t x_5;
        // metadata_x_t x_6;
        // metadata_x_t x_7;
    metadata_x_t x_8;
    metadata_temp_t t_1;
    metadata_temp_t t_2;
    metadata_temp_t t_3;
    metadata_temp_t t_4;
    metadata_temp_t t_5;
    metadata_temp_t t_6;
    metadata_temp_t t_7;
    metadata_temp_t t_8;
    metadata_short_t s_1;
    metadata_short_t s_2;
    metadata_short_t s_3;
    metadata_short_t s_4;
    metadata_short_t s_5;
    metadata_short_t s_6;
    metadata_short_t s_7;
    metadata_short_t s_8;
    metadata_temp4_t t4_8;
    metadata_linear_t linear1;
}

struct header_t {
    ethernet_h ethernet;//112
    ipv4_h ipv4;//160
    tcp_h tcp;//160
    udp_h udp;//64
    feature_h feature;//512
    output_h output;//64
}

struct empty_header_t {}

struct empty_metadata_t {}

struct metadata_t {
    metadata_temp_t t_1;
    metadata_temp_t t_2;
    metadata_temp_t t_3;
    metadata_temp_t t_4;
    metadata_temp_t t_5;
    metadata_temp_t t_6;
    metadata_temp_t t_7;
    metadata_linear_t linear0;
    metadata_linear_t linear1;
    metadata_linear_t linear2;
    metadata_linear_t linear3;
    metadata_linear_t linear4;
    metadata_x_t x_temp;
    bit<16> idx;
    bit<32> flow_hash;
    bit<32> flow_hash1;
    bit<16> src_port;
    bit<16> dst_port;
    bit<Register_Index_Size> flow_index;
    bit<32> pkt_count;
    bit<size_of_pkt_count_mod_7> pkt_count_mod_7;
}

struct eg_metadata_t {
    metadata_linear_t linear0;
    metadata_linear_t linear1;
    metadata_linear_t linear2;
    metadata_linear_t linear3;
    metadata_linear_t linear4;
}


#endif /* _HEADERS_ */