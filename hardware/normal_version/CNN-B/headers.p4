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
#define WIDTH 8
#define size_of_pkt_count_mod_7 8

#define stage_emb @stage(0)
#define stage_cal_hash @stage(0)
#define stage_ether_type_detect @stage(0)
#define stage_emb_fuse @stage(1)
#define stage_get_index @stage(1)
#define stage_get_last_hash @stage(1)
#define stage_cmp_hash @stage(2)
#define stage_pkt_co4 @stage(3)

#define stage_tab_swap @stage(9)
#define stage_x1_1_2 @stage(0)
#define stage_x1_3_4 @stage(0)
#define stage_x2_1_2 @stage(0)
#define stage_x2_3_4 @stage(0)
#define stage_x3_1_2 @stage(1)
#define stage_x3_3_4 @stage(1)
#define stage_x4_1_2 @stage(1)
#define stage_x4_3_4 @stage(1)
#define stage_x5_1_2 @stage(2)
#define stage_x5_3_4 @stage(2)
#define stage_x6_1_2 @stage(2)
#define stage_x6_3_4 @stage(2)
#define stage_x7_1_2 @stage(0)
#define stage_x7_3_4 @stage(0)

#define stage_x1_1_4 @stage(2)
#define stage_x2_1_4 @stage(3)
#define stage_x3_1_4 @stage(5)
#define stage_x4_1_4 @stage(7)
#define stage_x5_1_4 @stage(6)
#define stage_x6_1_4 @stage(4)
#define stage_x7_1_4 @stage(2)

#define stage_cnn_1_1_1_2 @stage(7)
#define stage_cnn_2_1_1_2 @stage(7)
#define stage_cnn_1_2_1_2 @stage(7)
#define stage_cnn_2_2_1_2 @stage(8)
#define stage_cnn_1_1_3_4 @stage(8)
#define stage_cnn_1_1_5_6 @stage(8)
#define stage_cnn_1_2_3_4 @stage(8)
#define stage_cnn_1_3_3_4 @stage(8)
#define stage_cnn_2_1_3_4 @stage(8)
#define stage_cnn_2_1_5_6 @stage(9)
#define stage_cnn_2_3_1_2 @stage(9)
#define stage_cnn_2_3_3_4 @stage(9)
#define stage_cnn_1_2_2_5 @stage(10)
#define stage_cnn_1_3_1_2 @stage(10)
#define stage_cnn_2_2_3_4 @stage(10)

#define stage_get_fc_1 @stage(10)
#define stage_get_fc_2 @stage(11)
#define stage_get_fc_3 @stage(12)
#define stage_get_fc_4 @stage(13)
#define stage_get_fc_5 @stage(14)

#define pkt_count_mod_7_0_0 pkt_count_mod_7
#define pkt_count_mod_7_1_0 pkt_count_mod_7
#define pkt_count_mod_7_2_0 pkt_count_mod_7
#define pkt_count_mod_7_3_0 pkt_count_mod_7
#define pkt_count_mod_7_4_0 pkt_count_mod_7
#define pkt_count_mod_7_5_0 pkt_count_mod_7
#define pkt_count_mod_7_6_0 pkt_count_mod_7
#define pkt_count_mod_7_7_0 pkt_count_mod_7
#define pkt_count_mod_7_0_1 pkt_count_mod_7
#define pkt_count_mod_7_1_1 pkt_count_mod_7
#define pkt_count_mod_7_2_1 pkt_count_mod_7
#define pkt_count_mod_7_3_1 pkt_count_mod_7
#define pkt_count_mod_7_4_1 pkt_count_mod_7
#define pkt_count_mod_7_5_1 pkt_count_mod_7
#define pkt_count_mod_7_6_1 pkt_count_mod_7
#define pkt_count_mod_7_7_1 pkt_count_mod_7


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

struct metadata_x_t {
    bit<16> l16_1;
    bit<16> l16_2;
}

struct metadata_cnn_1_t {
    bit<16> l16_1;
    bit<16> l16_2;
    bit<16> l16_3;
    bit<16> l16_4;
    bit<16> l16_5;
    bit<16> l16_6;
}

struct metadata_cnn_2_t {
    bit<16> l16_1;
    bit<16> l16_2;
    bit<16> l16_3;
    bit<16> l16_4;
    bit<16> l16_5;
}

struct metadata_cnn_3_t {
    bit<16> l16_1;
    bit<16> l16_2;
    bit<16> l16_3;
    bit<16> l16_4;
}
struct embed_t {
    bit<16> l16_1;
    bit<16> l16_2;
}
header feature_h {
    embed_t hidden_0;
    embed_t hidden_1;
    embed_t hidden_2;
    embed_t hidden_3;
    embed_t hidden_4;
    embed_t hidden_5;
    embed_t hidden_6;
    embed_t hidden_7;
}
header input_h {
    bit<16> pkl;
    bit<16> ipd;
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
    metadata_linear_t linear1;
}

struct header_t {
    ethernet_h ethernet;//112
    ipv4_h ipv4;//160
    tcp_h tcp;//160
    udp_h udp;//64
    input_h input_feature;
    feature_h feature;//512
    output_h output;//64
}

struct empty_header_t {}

struct empty_metadata_t {}
struct temp_t {
    bit<8> l16_1;
}
struct metadata_t {
    temp_t pkt_embeded_0;
    temp_t pkt_embeded_1;
    temp_t pkt_embeded_2;
    temp_t pkt_embeded_3;
    temp_t pkt_embeded_4;
    temp_t pkt_embeded_5;
    temp_t pkt_embeded_6;
    //embed_t pkt_embeded_7;

    embed_t hidden_0;
    embed_t hidden_1;
    embed_t hidden_2;
    embed_t hidden_3;
    embed_t hidden_4;
    embed_t hidden_5;
    embed_t hidden_6;
    embed_t hidden_7;

    bit<8> v;
    bit<16> idx;
    bit<32> flow_hash;
    bit<32> flow_hash1;
    bit<16> src_port;
    bit<16> dst_port;
    bit<Register_Index_Size> flow_index;
    bit<Register_Index_Size> flow_index0;
    bit<Register_Index_Size> flow_index1;
    bit<Register_Index_Size> flow_index2;
    bit<Register_Index_Size> flow_index3;
    bit<Register_Index_Size> flow_index4;
    bit<Register_Index_Size> flow_index5;
    bit<Register_Index_Size> flow_index6;
    // bit<16> packet_pkl;
    // bit<16> packet_ipd;

    bit<32> pkt_count;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_2;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_3;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_4;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_5;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_6;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_7;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_1_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_2_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_3_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_4_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_5_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_6_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_7_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_0_0;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_1_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_2_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_3_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_4_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_5_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_6_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_7_1;
    // bit<size_of_pkt_count_mod_7> pkt_count_mod_7_0_1;
    bit<size_of_pkt_count_mod_7> pkt_count_mod_7;



    metadata_linear_t linear1;
    metadata_linear_t linear2;
    metadata_linear_t linear3;
    metadata_linear_t linear4;
    metadata_cnn_1_t cnn_1_1;
    metadata_cnn_2_t cnn_1_2;
    metadata_cnn_3_t cnn_1_3;
    metadata_cnn_1_t cnn_2_1;
    metadata_cnn_2_t cnn_2_2;
    metadata_cnn_3_t cnn_2_3;
    metadata_temp_t temp;
}
struct eg_metadata_t {

}

#endif /* _HEADERS_ */