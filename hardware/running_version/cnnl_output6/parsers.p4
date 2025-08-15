/* -*- P4_16 -*- */

/*******************************************************************************
 * BAREFOOT NETWORKS CONFIDENTIAL & PROPRIETARY
 *
 * Copyright (c) Intel Corporation
 * SPDX-License-Identifier: CC-BY-ND-4.0
 */

#ifndef _PARSERS_
#define _PARSERS_


#include "headers.p4"

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;
    // Checksum() ipv4_checksum;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);

        transition parse_ethernet;
    }
    

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        //pkt feature
        transition select(hdr.ipv4.ihl, hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
            (5, 0, IP_PROTOCOLS_TCP) : parse_tcp;
            (5, 0, IP_PROTOCOLS_UDP)  : parse_udp;
            default: accept;
        }
    }

    // state parse_tcp {
    //     pkt.extract(hdr.tcp);
    //     transition parse_feature;
    // }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        ig_md.src_port = hdr.tcp.src_port;
        ig_md.dst_port = hdr.tcp.dst_port;
        // transition select(hdr.tcp.data_offset) {
        //     5: parse_feature;    // 无 options
        //     6: advance_4;
        //     7: advance_8;
        //     8: advance_12;
        //     9: advance_16;
        //     10: advance_20;
        //     11: advance_24;
        //     12: advance_28;
        //     13: advance_32;
        //     14: advance_36;
        //     15: advance_40;
        //     default: reject;
        // }
        transition parse_feature;
    }
    
    // state advance_4 { pkt.advance(32); transition parse_feature; } // 4 bytes = 32 bits
    // state advance_8 { pkt.advance(64); transition parse_feature; }
    // state advance_12 { pkt.advance(96); transition parse_feature; }
    // state advance_16 { pkt.advance(128); transition parse_feature; }
    // state advance_20 { pkt.advance(160); transition parse_feature; }
    // state advance_24 { pkt.advance(192); transition parse_feature; }
    // state advance_28 { pkt.advance(224); transition parse_feature; }
    // state advance_32 { pkt.advance(256); transition parse_feature; }
    // state advance_36 { pkt.advance(288); transition parse_feature; }
    // state advance_40 { pkt.advance(320); transition parse_feature; }

    state parse_udp {
        pkt.extract(hdr.udp);
        ig_md.src_port = hdr.udp.src_port;
        ig_md.dst_port = hdr.udp.dst_port;
        transition parse_feature;
    }

    state parse_feature {
        pkt.extract(hdr.feature);
        transition accept;
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.feature);
        pkt.emit(hdr.output);
    }
}


#endif /* _PARSERS_ */