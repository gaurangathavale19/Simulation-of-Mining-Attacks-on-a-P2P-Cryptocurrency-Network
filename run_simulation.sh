#!/bin/bash

for attack_type in selfish stubborn
do
    echo Attack Type: $attack_type
    for hashing_power in 0.1 0.2 0.3 0.4 0.5 0.6
    do
        echo Attacker Hashing Power: $hashing_power
        for zeta in 25 50 75
        do
            echo Zeta: $zeta
            python simulator.py --n_peers 100 --slow_nodes 50 --lowCPU_nodes 50 --txn_mean_time 60 --blk_mean_time 30 --termination_time 3600 --zeta $zeta --adversary_hashing_power $hashing_power --attack_type $attack_type
        done
    done
done