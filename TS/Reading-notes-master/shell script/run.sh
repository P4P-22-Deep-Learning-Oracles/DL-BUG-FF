#!/bin/bash
while true
do
    procnum=`ps -ef|grep ServiceManager-0.0.1-SNAPSHOT.jar|grep -v grep|wc -l`
    if [ $procnum -eq 0 ]; then
        cagent_tools alarm "Eureka_Server_2_Stop"
        nohup java -jar ServiceManager-0.0.1-SNAPSHOT.jar &
    fi
    sleep 60
done