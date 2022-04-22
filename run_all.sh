#!/bin/bash

sudo tcpdump -i br-6ca6cc61f6c7 port 8080 -q > output.txt &

docker stop server &> /dev/null
docker rm server &> /dev/null
docker run --name server --net dockernet --ip 172.19.0.10 -v ${PWD}:/app server &

docker stop client0 &> /dev/null
docker rm client0 &> /dev/null
docker run --name client0  --net dockernet --ip 172.19.0.2 -v ${PWD}:/app client 0 &

docker stop client1 &> /dev/null
docker rm client1 &> /dev/null
docker run --name client1  --net dockernet --ip 172.19.0.3 -v ${PWD}:/app client 1 &

docker stop client2 &> /dev/null
docker rm client2 &> /dev/null
docker run --name client2  --net dockernet --ip 172.19.0.4 -v ${PWD}:/app client 2 &

docker stop client3 &> /dev/null
docker rm client3 &> /dev/null
docker run --name client3  --net dockernet --ip 172.19.0.5 -v ${PWD}:/app client 3 &

docker stop client4 &> /dev/null
docker rm client4 &> /dev/null
docker run --name client4  --net dockernet --ip 172.19.0.6 -v ${PWD}:/app client 4

LASTDIR=$(ls -t | grep report | head -n 1)
cp output.txt $LASTDIR