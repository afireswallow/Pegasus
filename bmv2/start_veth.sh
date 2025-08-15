#!/bin/sh
echo "Cleaning..."
sudo ip link delete veth1 2>/dev/null || true
sudo ip link delete veth0 2>/dev/null || true
sudo ip link delete veth1-peer 2>/dev/null || true
sudo ip link delete veth0-peer 2>/dev/null || true


echo "create veth..."
sudo ip link add veth1 type veth peer name veth1-peer
sudo ip link add veth0 type veth peer name veth0-peer

echo "start ip link..."
sudo ip link set veth1 up
sudo ip link set veth0 up
sudo ip link set veth1-peer up
sudo ip link set veth0-peer up

sudo ip addr add 10.0.1.1/24 dev veth1-peer 2>/dev/null || true
sudo ip addr add 10.0.2.1/24 dev veth0-peer 2>/dev/null || true

echo "Veth you can use:"
ip link show | grep veth 