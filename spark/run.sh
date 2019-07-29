#!/bin/sh
/run.sh &
spark-submit --jars='/app/*.jar' '/app/data.py'
