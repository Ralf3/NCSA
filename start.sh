#!/usr/bin/env bash

ip addr add 192.168.1.100 dev enp3s6
ip route add 192.168.1.0/24 via 192.168.1.100 dev enp3s6
ip link set enp3s6 up
