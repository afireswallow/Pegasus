#!/bin/sh
cd p4_CICIOT2022

mkdir bmv2logs

mkdir build

pkill -f simple_switch_grpc

p4c-bm2-ss --target bmv2 --arch v1model --p4runtime-files build/mlp_CICIOT.p4info.txt -o build/mlp_CICIOT.json basic.p4 

echo "Starting the switch..."

sudo simple_switch_grpc -i 0@veth0 -i 1@veth1 --log-console --no-p4  -- --grpc-server-addr 127.0.0.1:50051 build/mlp_CICIOT.json > bmv2logs/run_switch.log 2>&1
