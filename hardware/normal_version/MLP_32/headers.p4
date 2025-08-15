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

#define Register_Table_Size 65536 //register array size
#define Register_Index_Size 16    //register array index bits


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

struct metadata_temp_t {
    bit<4> l4_1;
    bit<4> l4_2;
    bit<4> l4_3;
    bit<4> l4_4;
    bit<4> l4_5;
    bit<4> l4_6;
    bit<4> l4_7;
    bit<4> l4_8;
    bit<4> l4_9;
    bit<4> l4_10;
    bit<4> l4_11;
    bit<4> l4_12;
    bit<4> l4_13;
    bit<4> l4_14;
    bit<4> l4_15;
    bit<4> l4_16;
}

struct metadata_linear_t {
    bit<32> l32_1;
    bit<32> l32_2;
    bit<32> l32_3;
    bit<32> l32_4;
    bit<32> l32_5;
    bit<32> l32_6;
    bit<32> l32_7;
    bit<32> l32_8;
}

header feature_h {
    bit<8> ip_len;
    bit<16> ip_total_len;
    bit<8> protocol;
    bit<8> tos;
    bit<8> ttl;
    bit<16> offset;
    bit<32> max_byte;
    bit<16> min_byte;
    bit<16> max_ipd;
    bit<16> min_ipd;
    bit<16> ipd;
}

header output_h {
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
    metadata_linear_t linear1;
    metadata_linear_t linear2;
    metadata_linear_t linear3;
    metadata_linear_t linear4;
    metadata_temp_t temp;
}

#endif /* _HEADERS_ */