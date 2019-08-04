FROM bitnami/spark:2.4.3

USER 0

# Install python dependencies
RUN apt-get update && apt-get install wget

# Fetch spark-tensorflow-connector
RUN wget -O /opt/bitnami/spark/jars/spark-tf-connector.jar \
https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.14.0/spark-tensorflow-connector_2.11-1.14.0.jar

COPY spark-defaults.conf /opt/bitnami/spark/conf/
COPY target/scala-2.11/trader_2.11-0.1.jar /app/trader.jar

USER 1001

# Copy runner to start spark + spark-submit job
VOLUME [ "/data/src", "/data/dest" ]
