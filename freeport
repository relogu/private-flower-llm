#!/bin/sh

netcat -l localhost 0 &

lsof -i \
| grep $! \
| awk '{print $9}' \
| cut -d':' -f2;

kill $!;
