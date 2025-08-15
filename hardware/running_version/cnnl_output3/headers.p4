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
#define Width 8
#define Register_Table_Size (1<<16) //register array size
#define Register_Index_Size 16    //register array index bits
// #define Register_Table_Size_ipd 65536 //register array size
// #define Register_Index_Size_ipd 16    //register array index bits
#define size_of_pkt_count_mod_8 8

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
}
struct metadata_byte_t {
    bit<8> l8_1;
    bit<8> l8_2;
}
struct metadata_single_byte_t {
    bit<8> l8_1;
}
struct metadata_x_t {
    bit<16> l16_1;
    bit<16> l16_2;
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
}
header feature_h {
    // metadata_byte_t b_1;
    // metadata_byte_t b_2;
    // metadata_byte_t b_3;
    // metadata_byte_t b_4;
    // metadata_byte_t b_5;
    // metadata_byte_t b_6;
    // metadata_byte_t b_7;
    // metadata_byte_t b_8;
    // metadata_byte_t b_9;
    // metadata_byte_t b_10;
    metadata_byte_t b_11;
    metadata_byte_t b_12;
    metadata_byte_t b_13;
    metadata_byte_t b_14;
    metadata_byte_t b_15;
    metadata_byte_t b_16;
    metadata_byte_t b_17;
    metadata_byte_t b_18;
    metadata_byte_t b_19;
    metadata_byte_t b_20;
    metadata_byte_t b_21;
    metadata_byte_t b_22;
    metadata_byte_t b_23;
    metadata_byte_t b_24;
    metadata_byte_t b_25;
    metadata_byte_t b_26;
    metadata_byte_t b_27;
    metadata_single_byte_t b_28;
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
    bit<4> l4_17;
    bit<4> l4_18;
    bit<4> l4_19;
    bit<4> l4_20;
    bit<4> l4_21;
    bit<4> l4_22;
    bit<4> l4_23;
    bit<4> l4_24;
    bit<4> l4_25;
    bit<4> l4_26;
    bit<4> l4_27;
    bit<4> l4_28;
    bit<4> l4_29;
    bit<4> l4_30;
}
struct reg_t {
    bit<8> l8_1;
    bit<8> l8_2;
    bit<8> l8_3;
    bit<8> l8_4;
    bit<8> l8_5;
    bit<8> l8_6;
    bit<8> l8_7;
    bit<8> l8_8;
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
    metadata_byte_t b_1;
    metadata_byte_t b_2;
    metadata_byte_t b_3;
    metadata_byte_t b_4;
    metadata_byte_t b_5;
    metadata_byte_t b_6;
    metadata_byte_t b_7;
    metadata_byte_t b_8;
    metadata_byte_t b_9;
    metadata_byte_t b_10;
    
    metadata_linear_t linear1;
    metadata_short_t s_1;
    metadata_short_t s_2;
    metadata_short_t s_3;
    metadata_short_t s_4;
    metadata_temp_t temp;
    reg_t reg;
    reg_t reg1;
    bit<16> idx;
    bit<32> flow_hash;
    bit<32> flow_hash1;
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> timestamp;
    bit<16> last_timestamp;
    bit<16> interval;
    bit<4> data_offset;
    bit<Register_Index_Size> flow_index;
//     bit<Register_Index_Size_ipd> flow_index_ipd;
    bit<32> pkt_count;
    bit<8> real_pkt_count;
    bit<size_of_pkt_count_mod_8> pkt_count_mod_8;
}

struct eg_metadata_t {
    metadata_linear_t linear1;
    metadata_temp_t temp;
}
#endif /* _HEADERS_ */