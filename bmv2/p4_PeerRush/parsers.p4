/* -*- P4_16 -*- */
#ifndef _PARSERS_
#define _PARSERS_


#include "headers.p4"


// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------

parser MyParser(
        packet_in pkt,
        out header_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {

    state start {
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
        transition select(hdr.ipv4.ihl, hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
            (5, 0, IP_PROTOCOLS_TCP) : parse_tcp;  
            (5, 0, IP_PROTOCOLS_UDP)  : parse_udp;  
            default: accept;  
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition parse_feature;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_feature;
    }

    state parse_feature {
        pkt.extract(hdr.feature);
        transition accept;
    }

}



control MyVerifyChecksum(inout header_t hdr, inout metadata_t meta) {   
    apply {  }
}



control MyComputeChecksum(inout header_t hdr, inout metadata_t meta) {
     apply {}
}



// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------


control MyDeparser(
        packet_out pkt,
        in header_t hdr) {

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
