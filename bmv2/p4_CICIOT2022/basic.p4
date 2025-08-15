/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>
#include "headers.p4"
#include "parsers.p4"

control MyIngress(
        inout header_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) 
{


    action noaction(){}

// ****cal h1 

    action set_ip_len_total_len(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        meta.linear1.l32_1 = in01;
        meta.linear1.l32_2 = in02;
        meta.linear1.l32_3 = in03;
        meta.linear1.l32_4 = in04;
        meta.linear1.l32_5 = in05;
        meta.linear1.l32_6 = in06;
        meta.linear1.l32_7 = in07;
        meta.linear1.l32_8 = in08;

        meta.linear2.l32_1 = in11;
        meta.linear2.l32_2 = in12;
        meta.linear2.l32_3 = in13;
        meta.linear2.l32_4 = in14;
        meta.linear2.l32_5 = in15;
        meta.linear2.l32_6 = in16;
        meta.linear2.l32_7 = in17;
        meta.linear2.l32_8 = in18;
    }
    
    action set_protocol_tos(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        meta.linear3.l32_1 = in01;
        meta.linear3.l32_2 = in02;
        meta.linear3.l32_3 = in03;
        meta.linear3.l32_4 = in04;
        meta.linear3.l32_5 = in05;
        meta.linear3.l32_6 = in06;
        meta.linear3.l32_7 = in07;
        meta.linear3.l32_8 = in08;

        meta.linear4.l32_1 = in11;
        meta.linear4.l32_2 = in12;
        meta.linear4.l32_3 = in13;
        meta.linear4.l32_4 = in14;
        meta.linear4.l32_5 = in15;
        meta.linear4.l32_6 = in16;
        meta.linear4.l32_7 = in17;
        meta.linear4.l32_8 = in18;
    }
    
    action set_offset_max_byte(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        meta.linear1.l32_1 = in01 + meta.linear1.l32_1;
        meta.linear1.l32_2 = in02 + meta.linear1.l32_2;
        meta.linear1.l32_3 = in03 + meta.linear1.l32_3;
        meta.linear1.l32_4 = in04 + meta.linear1.l32_4;
        meta.linear1.l32_5 = in05 + meta.linear1.l32_5;
        meta.linear1.l32_6 = in06 + meta.linear1.l32_6;
        meta.linear1.l32_7 = in07 + meta.linear1.l32_7;
        meta.linear1.l32_8 = in08 + meta.linear1.l32_8;

        meta.linear2.l32_1 = in11 + meta.linear2.l32_1;
        meta.linear2.l32_2 = in12 + meta.linear2.l32_2;
        meta.linear2.l32_3 = in13 + meta.linear2.l32_3;
        meta.linear2.l32_4 = in14 + meta.linear2.l32_4;
        meta.linear2.l32_5 = in15 + meta.linear2.l32_5;
        meta.linear2.l32_6 = in16 + meta.linear2.l32_6;
        meta.linear2.l32_7 = in17 + meta.linear2.l32_7;
        meta.linear2.l32_8 = in18 + meta.linear2.l32_8;
    }
    
    action set_min_byte_max_ipd(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        meta.linear3.l32_1 = in01 + meta.linear3.l32_1;
        meta.linear3.l32_2 = in02 + meta.linear3.l32_2;
        meta.linear3.l32_3 = in03 + meta.linear3.l32_3;
        meta.linear3.l32_4 = in04 + meta.linear3.l32_4;
        meta.linear3.l32_5 = in05 + meta.linear3.l32_5;
        meta.linear3.l32_6 = in06 + meta.linear3.l32_6;
        meta.linear3.l32_7 = in07 + meta.linear3.l32_7;
        meta.linear3.l32_8 = in08 + meta.linear3.l32_8;

        meta.linear4.l32_1 = in11 + meta.linear4.l32_1;
        meta.linear4.l32_2 = in12 + meta.linear4.l32_2;
        meta.linear4.l32_3 = in13 + meta.linear4.l32_3;
        meta.linear4.l32_4 = in14 + meta.linear4.l32_4;
        meta.linear4.l32_5 = in15 + meta.linear4.l32_5;
        meta.linear4.l32_6 = in16 + meta.linear4.l32_6;
        meta.linear4.l32_7 = in17 + meta.linear4.l32_7;
        meta.linear4.l32_8 = in18 + meta.linear4.l32_8;
    }
    
    action set_min_ipd_ipd(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08,
                bit<32> in11, bit<32> in12, bit<32> in13, bit<32> in14,
                bit<32> in15, bit<32> in16, bit<32> in17, bit<32> in18) {
        meta.linear1.l32_1 = in01 + meta.linear1.l32_1;
        meta.linear1.l32_2 = in02 + meta.linear1.l32_2;
        meta.linear1.l32_3 = in03 + meta.linear1.l32_3;
        meta.linear1.l32_4 = in04 + meta.linear1.l32_4;
        meta.linear1.l32_5 = in05 + meta.linear1.l32_5;
        meta.linear1.l32_6 = in06 + meta.linear1.l32_6;
        meta.linear1.l32_7 = in07 + meta.linear1.l32_7;
        meta.linear1.l32_8 = in08 + meta.linear1.l32_8;

        meta.linear2.l32_1 = in11 + meta.linear2.l32_1;
        meta.linear2.l32_2 = in12 + meta.linear2.l32_2;
        meta.linear2.l32_3 = in13 + meta.linear2.l32_3;
        meta.linear2.l32_4 = in14 + meta.linear2.l32_4;
        meta.linear2.l32_5 = in15 + meta.linear2.l32_5;
        meta.linear2.l32_6 = in16 + meta.linear2.l32_6;
        meta.linear2.l32_7 = in17 + meta.linear2.l32_7;
        meta.linear2.l32_8 = in18 + meta.linear2.l32_8;
    }
    
    

    table ip_len_total_len {
        key = {
            hdr.feature.ip_len: ternary;
            hdr.feature.ip_total_len: ternary;
        }
        actions = {
            set_ip_len_total_len;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table protocol_tos {
        key = {
            hdr.feature.protocol: ternary;
            hdr.feature.tos: ternary;
        }
        actions = {
            set_protocol_tos;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table offset_max_byte {
        key = {
            hdr.feature.offset: ternary;
            hdr.feature.max_byte: ternary;
        }
        actions = {
            set_offset_max_byte;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table min_byte_max_ipd {
        key = {
            hdr.feature.min_byte: ternary;
            hdr.feature.max_ipd: ternary;
        }
        actions = {
            set_min_byte_max_ipd;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table min_ipd_ipd {
        key = {
            hdr.feature.min_ipd: ternary;
            hdr.feature.ipd: ternary;
        }
        actions = {
            set_min_ipd_ipd;
            noaction;
        }
        size = 8192;
        default_action = noaction();
         
    }
    


// ****cal h2 first   

    action set_h1_1(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear3.l32_1 = in01;
        meta.linear3.l32_2 = in02;
        meta.linear3.l32_3 = in03;
        meta.linear3.l32_4 = in04;
        meta.linear3.l32_5 = in05;
        meta.linear3.l32_6 = in06;
        meta.linear3.l32_7 = in07;
        meta.linear3.l32_8 = in08;
    }
    
    action set_h1_2(bit<4> in01) {
        meta.temp.l4_1 = in01;
    }
    
    action set_h1_3(bit<4> in01) {
        meta.temp.l4_2 = in01;
    }
    
    action set_h1_4(bit<4> in01) {
        meta.temp.l4_3 = in01;
    }
    
    action set_h1_5(bit<4> in01) {
        meta.temp.l4_4 = in01;
    }
    
    action set_h1_6(bit<4> in01) {
        meta.temp.l4_5 = in01;
    }
    
    action set_h1_7(bit<4> in01) {
        meta.temp.l4_6 = in01;
    }
    
    action set_h1_8(bit<4> in01) {
        meta.temp.l4_7 = in01;
    }
    
    action set_h1_9(bit<4> in01) {
        meta.temp.l4_8 = in01;
    }
    
    action set_h1_10(bit<4> in01) {
        meta.temp.l4_9 = in01;
    }
    
    action set_h1_11(bit<4> in01) {
        meta.temp.l4_10 = in01;
    }
    
    action set_h1_12(bit<4> in01) {
        meta.temp.l4_11 = in01;
    }
    
    action set_h1_13(bit<4> in01) {
        meta.temp.l4_12 = in01;
    }
    
    action set_h1_14(bit<4> in01) {
        meta.temp.l4_13 = in01;
    }
    
    action set_h1_15(bit<4> in01) {
        meta.temp.l4_14 = in01;
    }
    
    action set_h1_16(bit<4> in01) {
        meta.temp.l4_15 = in01;
    }
    

    table h1_1 {
        key = {
            meta.linear1.l32_1: ternary;
        }
        actions = {
            set_h1_1;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }

    table h1_2 {
        key = {
            meta.linear1.l32_2: ternary;
        }
        actions = {
            set_h1_2;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_3 {
        key = {
            meta.linear1.l32_3: ternary;
        }
        actions = {
            set_h1_3;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }

    table h1_4 {
        key = {
            meta.linear1.l32_4: ternary;
        }
        actions = {
            set_h1_4;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_5 {
        key = {
            meta.linear1.l32_5: ternary;
        }
        actions = {
            set_h1_5;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_6 {
        key = {
            meta.linear1.l32_6: ternary;
        }
        actions = {
            set_h1_6;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_7 {
        key = {
            meta.linear1.l32_7: ternary;
        }
        actions = {
            set_h1_7;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_8 {
        key = {
            meta.linear1.l32_8: ternary;
        }
        actions = {
            set_h1_8;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_9 {
        key = {
            meta.linear2.l32_1: ternary;
        }
        actions = {
            set_h1_9;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_10 {
        key = {
            meta.linear2.l32_2: ternary;
        }
        actions = {
            set_h1_10;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_11 {
        key = {
            meta.linear2.l32_3: ternary;
        }
        actions = {
            set_h1_11;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_12 {
        key = {
            meta.linear2.l32_4: ternary;
        }
        actions = {
            set_h1_12;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_13 {
        key = {
            meta.linear2.l32_5: ternary;
        }
        actions = {
            set_h1_13;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_14 {
        key = {
            meta.linear2.l32_6: ternary;
        }
        actions = {
            set_h1_14;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_15 {
        key = {
            meta.linear2.l32_7: ternary;
        }
        actions = {
            set_h1_15;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    

    table h1_16 {
        key = {
            meta.linear2.l32_8: ternary;
        }
        actions = {
            set_h1_16;
            noaction;
        }
        size = 1024;
        default_action = noaction();
         
    }
    
// ****cal h2 second  ====== MLP second layer action second ======

    action set_h1_2_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear3.l32_1 = in01 + meta.linear3.l32_1;
        meta.linear3.l32_2 = in02 + meta.linear3.l32_2;
        meta.linear3.l32_3 = in03 + meta.linear3.l32_3;
        meta.linear3.l32_4 = in04 + meta.linear3.l32_4;
        meta.linear3.l32_5 = in05 + meta.linear3.l32_5;
        meta.linear3.l32_6 = in06 + meta.linear3.l32_6;
        meta.linear3.l32_7 = in07 + meta.linear3.l32_7;
        meta.linear3.l32_8 = in08 + meta.linear3.l32_8;
    }
    
    action set_h1_5_7(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear4.l32_1 = in01;
        meta.linear4.l32_2 = in02;
        meta.linear4.l32_3 = in03;
        meta.linear4.l32_4 = in04;
        meta.linear4.l32_5 = in05;
        meta.linear4.l32_6 = in06;
        meta.linear4.l32_7 = in07;
        meta.linear4.l32_8 = in08;
    }
    
    action set_h1_8_10(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear3.l32_1 = in01 + meta.linear3.l32_1;
        meta.linear3.l32_2 = in02 + meta.linear3.l32_2;
        meta.linear3.l32_3 = in03 + meta.linear3.l32_3;
        meta.linear3.l32_4 = in04 + meta.linear3.l32_4;
        meta.linear3.l32_5 = in05 + meta.linear3.l32_5;
        meta.linear3.l32_6 = in06 + meta.linear3.l32_6;
        meta.linear3.l32_7 = in07 + meta.linear3.l32_7;
        meta.linear3.l32_8 = in08 + meta.linear3.l32_8;
    }
    
    action set_h1_11_13(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear4.l32_1 = in01 + meta.linear4.l32_1;
        meta.linear4.l32_2 = in02 + meta.linear4.l32_2;
        meta.linear4.l32_3 = in03 + meta.linear4.l32_3;
        meta.linear4.l32_4 = in04 + meta.linear4.l32_4;
        meta.linear4.l32_5 = in05 + meta.linear4.l32_5;
        meta.linear4.l32_6 = in06 + meta.linear4.l32_6;
        meta.linear4.l32_7 = in07 + meta.linear4.l32_7;
        meta.linear4.l32_8 = in08 + meta.linear4.l32_8;
    }
    
    action set_h1_14_16(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04,
                bit<32> in05, bit<32> in06, bit<32> in07, bit<32> in08) {
        meta.linear3.l32_1 = in01 + meta.linear3.l32_1;
        meta.linear3.l32_2 = in02 + meta.linear3.l32_2;
        meta.linear3.l32_3 = in03 + meta.linear3.l32_3;
        meta.linear3.l32_4 = in04 + meta.linear3.l32_4;
        meta.linear3.l32_5 = in05 + meta.linear3.l32_5;
        meta.linear3.l32_6 = in06 + meta.linear3.l32_6;
        meta.linear3.l32_7 = in07 + meta.linear3.l32_7;
        meta.linear3.l32_8 = in08 + meta.linear3.l32_8;
    }
    

    table h1_2_4 {
        key = {
            meta.temp.l4_1: exact;
            meta.temp.l4_2: exact;
            meta.temp.l4_3: exact;
        }
        actions = {
            set_h1_2_4;
            noaction;
        }
        size = 4096;
        default_action = noaction();
         
    }
    

    table h1_5_7 {
        key = {
            meta.temp.l4_4: exact;
            meta.temp.l4_5: exact;
            meta.temp.l4_6: exact;
        }
        actions = {
            set_h1_5_7;
            noaction;
        }
        size = 4096;
        default_action = noaction();
         
    }
    

    table h1_8_10 {
        key = {
            meta.temp.l4_7: exact;
            meta.temp.l4_8: exact;
            meta.temp.l4_9: exact;
        }
        actions = {
            set_h1_8_10;
            noaction;
        }
        size = 4096;
        default_action = noaction();
         
    }
    

    table h1_11_13 {
        key = {
            meta.temp.l4_10: exact;
            meta.temp.l4_11: exact;
            meta.temp.l4_12: exact;
        }
        actions = {
            set_h1_11_13;
            noaction;
        }
        size = 4096;
        default_action = noaction();
         
    }
    

    table h1_14_16 {
        key = {
            meta.temp.l4_13: exact;
            meta.temp.l4_14: exact;
            meta.temp.l4_15: exact;
        }
        actions = {
            set_h1_14_16;
            noaction;
        }
        size = 4096;
        default_action = noaction();
         
    }
    
// ****cal output

    action set_h2_1(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        meta.linear2.l32_1 = in01;
        meta.linear2.l32_2 = in02;
        meta.linear2.l32_3 = in03;
        //meta.linear2.l32_4 = in04;
    }
    
    action set_h2_2(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01;
        hdr.output.linear1.l32_2 = in02;
        hdr.output.linear1.l32_3 = in03;
        //hdr.output.linear1.l32_4 = in04;
    }
    
    action set_h2_3(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        meta.linear2.l32_1 = in01 + meta.linear2.l32_1;
        meta.linear2.l32_2 = in02 + meta.linear2.l32_2;
        meta.linear2.l32_3 = in03 + meta.linear2.l32_3;
        //meta.linear2.l32_4 = in04 + meta.linear2.l32_4;
    }
    
    action set_h2_4(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h2_5(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        meta.linear2.l32_1 = in01 + meta.linear2.l32_1;
        meta.linear2.l32_2 = in02 + meta.linear2.l32_2;
        meta.linear2.l32_3 = in03 + meta.linear2.l32_3;
        //meta.linear2.l32_4 = in04 + meta.linear2.l32_4;
    }
    
    action set_h2_6(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    
    action set_h2_7(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        meta.linear2.l32_1 = in01 + meta.linear2.l32_1;
        meta.linear2.l32_2 = in02 + meta.linear2.l32_2;
        meta.linear2.l32_3 = in03 + meta.linear2.l32_3;
        //meta.linear2.l32_4 = in04 + meta.linear2.l32_4;
    }
    
    action set_h2_8(bit<32> in01, bit<32> in02, bit<32> in03, bit<32> in04) {
        hdr.output.linear1.l32_1 = in01 + hdr.output.linear1.l32_1;
        hdr.output.linear1.l32_2 = in02 + hdr.output.linear1.l32_2;
        hdr.output.linear1.l32_3 = in03 + hdr.output.linear1.l32_3;
        //hdr.output.linear1.l32_4 = in04 + hdr.output.linear1.l32_4;
    }
    

    table h2_1 {
        key = {
            meta.linear3.l32_1: ternary;
        }
        actions = {
            set_h2_1;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_2 {
        key = {
            meta.linear3.l32_2: ternary;
        }
        actions = {
            set_h2_2;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_3 {
        key = {
            meta.linear3.l32_3: ternary;
        }
        actions = {
            set_h2_3;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_4 {
        key = {
            meta.linear3.l32_4: ternary;
        }
        actions = {
            set_h2_4;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_5 {
        key = {
            meta.linear3.l32_5: ternary;
        }
        actions = {
            set_h2_5;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_6 {
        key = {
            meta.linear3.l32_6: ternary;
        }
        actions = {
            set_h2_6;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_7 {
        key = {
            meta.linear3.l32_7: ternary;
        }
        actions = {
            set_h2_7;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    

    table h2_8 {
        key = {
            meta.linear3.l32_8: ternary;
        }
        actions = {
            set_h2_8;
            noaction;
        }
        size = 2048;
        default_action = noaction();
         
    }
    
// ************************    

    apply
    {
// *****cal h1 as linear1 linear2
        ip_len_total_len.apply();
        protocol_tos.apply();
        offset_max_byte.apply();
        min_byte_max_ipd.apply();
        min_ipd_ipd.apply();
        hdr.output.setValid();
        

        meta.linear1.l32_1 = meta.linear1.l32_1 + meta.linear3.l32_1;
        meta.linear1.l32_2 = meta.linear1.l32_2 + meta.linear3.l32_2;
        meta.linear1.l32_3 = meta.linear1.l32_3 + meta.linear3.l32_3;
        meta.linear1.l32_4 = meta.linear1.l32_4 + meta.linear3.l32_4;
        meta.linear1.l32_5 = meta.linear1.l32_5 + meta.linear3.l32_5;
        meta.linear1.l32_6 = meta.linear1.l32_6 + meta.linear3.l32_6;
        meta.linear1.l32_7 = meta.linear1.l32_7 + meta.linear3.l32_7;
        meta.linear1.l32_8 = meta.linear1.l32_8 + meta.linear3.l32_8;
        meta.linear2.l32_1 = meta.linear2.l32_1 + meta.linear4.l32_1;
        meta.linear2.l32_2 = meta.linear2.l32_2 + meta.linear4.l32_2;
        meta.linear2.l32_3 = meta.linear2.l32_3 + meta.linear4.l32_3;
        meta.linear2.l32_4 = meta.linear2.l32_4 + meta.linear4.l32_4;
        meta.linear2.l32_5 = meta.linear2.l32_5 + meta.linear4.l32_5;
        meta.linear2.l32_6 = meta.linear2.l32_6 + meta.linear4.l32_6;
        meta.linear2.l32_7 = meta.linear2.l32_7 + meta.linear4.l32_7;
        meta.linear2.l32_8 = meta.linear2.l32_8 + meta.linear4.l32_8;

// *****cal h2 as linear3
        h1_1.apply();
        h1_2.apply();
        h1_3.apply();
        h1_4.apply();
        h1_5.apply();
        h1_6.apply();
        h1_7.apply();
        h1_8.apply();
        h1_9.apply();
        h1_10.apply();
        h1_11.apply();
        h1_12.apply();
        h1_13.apply();
        h1_14.apply();
        h1_15.apply();
        h1_16.apply();

        h1_2_4.apply();
        h1_5_7.apply();
        h1_8_10.apply();
        h1_11_13.apply();
        h1_14_16.apply();

        meta.linear3.l32_1 = meta.linear3.l32_1 + meta.linear4.l32_1;
        meta.linear3.l32_2 = meta.linear3.l32_2 + meta.linear4.l32_2;
        meta.linear3.l32_3 = meta.linear3.l32_3 + meta.linear4.l32_3;
        meta.linear3.l32_4 = meta.linear3.l32_4 + meta.linear4.l32_4;
        meta.linear3.l32_5 = meta.linear3.l32_5 + meta.linear4.l32_5;
        meta.linear3.l32_6 = meta.linear3.l32_6 + meta.linear4.l32_6;
        meta.linear3.l32_7 = meta.linear3.l32_7 + meta.linear4.l32_7;
        meta.linear3.l32_8 = meta.linear3.l32_8 + meta.linear4.l32_8;

// *****cal output as linear3
        h2_1.apply();
        h2_2.apply();
        h2_3.apply();
        h2_4.apply();
        h2_5.apply();
        h2_6.apply();
        h2_7.apply();
        h2_8.apply();

        hdr.output.linear1.l32_1 = hdr.output.linear1.l32_1 + meta.linear2.l32_1;
        hdr.output.linear1.l32_2 = hdr.output.linear1.l32_2 + meta.linear2.l32_2;
        hdr.output.linear1.l32_3 = hdr.output.linear1.l32_3 + meta.linear2.l32_3;
        //hdr.output.linear1.l32_4 = hdr.output.linear1.l32_4 + meta.linear2.l32_4;



        standard_metadata.egress_spec = 1;


        hdr.ethernet.dst_addr = 0; //for filter
    }
}


control MyEgress(inout header_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {
    apply { }
}


V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main; 