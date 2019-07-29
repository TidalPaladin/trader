FROM bitnami/spark:2.4.3

USER 0

# Install python dependencies
RUN apt-get update && apt-get install wget
RUN pip install numpy ipython pyspark==2.4.3

# Fetch spark-tensorflow-connector
RUN wget -O /opt/bitnami/spark/jars/spark-tf-connector.jar \
https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.14.0/spark-tensorflow-connector_2.11-1.14.0.jar

COPY spark-defaults.conf /opt/bitnami/spark/conf/

USER 1001

# Copy runner to start spark + spark-submit job

COPY python/ /app
VOLUME [ "/artifacts", "/data/src", "/data/dest" ]
