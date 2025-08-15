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
struct metadata_byte_t {
    bit<8> l8_1;
    bit<8> l8_2;
    bit<8> l8_3;
    bit<8> l8_4;
    bit<8> l8_5;
    bit<8> l8_6;
}

struct metadata_x_t {
    bit<8> l8_1;
    bit<8> l8_2;
    bit<8> l8_3;
    bit<8> l8_4;
}
struct metadata_difference_t {
    bit<8> l8_1;
    bit<8> l8_2;
    bit<8> l8_3;
    bit<8> l8_4;
    bit<8> l8_5;
    bit<8> l8_6;
    bit<8> l8_7;
    bit<8> l8_8;
    bit<8> l8_9;
    bit<8> l8_10;
    bit<8> l8_11;
    bit<8> l8_12;
    bit<8> l8_13;
    bit<8> l8_14;
    bit<8> l8_15;
    bit<8> l8_16;
    bit<8> l8_17;
    bit<8> l8_18;
    bit<8> l8_19;
    bit<8> l8_20;
    bit<8> l8_21;
    bit<8> l8_22;
    bit<8> l8_23;
    bit<8> l8_24;
    bit<8> l8_25;
    bit<8> l8_26;
    bit<8> l8_27;
    bit<8> l8_28;
    bit<8> l8_29;
    bit<8> l8_30;
    bit<8> l8_31;
    bit<8> l8_32;
}
struct metadata_short_t {
    bit<16> l16_1;
    bit<16> l16_2;
    bit<16> l16_3;
    bit<16> l16_4;
    bit<16> l16_5;
    bit<16> l16_6;
    bit<16> l16_7;
    bit<16> l16_8;
    bit<16> l16_9;
    bit<16> l16_10;
    bit<16> l16_11;
    bit<16> l16_12;
    bit<16> l16_13;
    bit<16> l16_14;
    bit<16> l16_15;
    bit<16> l16_16;
    bit<16> l16_17;
    bit<16> l16_18;
    bit<16> l16_19;
    bit<16> l16_20;
    bit<16> l16_21;
    bit<16> l16_22;
    bit<16> l16_23;
    bit<16> l16_24;
    bit<16> l16_25;
    bit<16> l16_26;
    bit<16> l16_27;
    bit<16> l16_28;
    bit<16> l16_29;
    bit<16> l16_30;
    bit<16> l16_31;
    bit<16> l16_32;
}
header feature_h {
    bit<16> pkl;
    bit<16> ipd;
    metadata_x_t x_1;
    metadata_x_t x_2;
    metadata_x_t x_3;
    metadata_x_t x_4;
    metadata_x_t x_5;
    metadata_x_t x_6;
    metadata_x_t x_7;
    metadata_x_t x_8;
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

header output_h {
    bit<16> l16_1;
}
struct storage_t {
    //metadata_difference_t d1;
    metadata_short_t s_1;
}
struct header_t {
    ethernet_h ethernet;//112
    ipv4_h ipv4;//160
    tcp_h tcp;//160
    udp_h udp;//64
    feature_h feature;//512
    output_h output;//64
    storage_t storage;
}

struct empty_header_t {}

struct empty_metadata_t {}

struct metadata_t {
    metadata_linear_t linear1;
    metadata_linear_t linear2;
    metadata_linear_t linear3;
    metadata_linear_t linear4;
    metadata_byte_t b1;
    metadata_byte_t b2;
    metadata_byte_t b3;
    metadata_byte_t b4;
    metadata_byte_t b5;
    metadata_byte_t b6;
    metadata_byte_t b7;
    metadata_byte_t b8;
    metadata_difference_t d1;
    metadata_temp_t temp;
}
struct eg_metadata_t {}
#endif /* _HEADERS_ */