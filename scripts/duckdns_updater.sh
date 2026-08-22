#!/bin/bash
DOMAIN="algomlb,cellstock,stocktrials,downbeat"
TOKEN="5dfdff5d-a1ad-41d8-accd-0ab6ac5b93e7"
echo url="https://www.duckdns.org/update?domains=$DOMAIN&token=$TOKEN&ip=" | curl -k -o /home/opc/AlgoMLB/logs/duckdns.log -K -
