FROM tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter as base

# Spark dependencies
ENV "APACHE_SPARK_VERSION"="2.4.3" "HADOOP_VERSION"="2.7"

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y wget openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

# Install spark
RUN cd /tmp && \
    wget -q http://mirrors.ukfast.co.uk/sites/ftp.apache.org/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    echo "E8B7F9E1DEC868282CADCAD81599038A22F48FB597D44AF1B13FCC76B7DACD2A1CAF431F95E394E1227066087E3CE6C2137C4ABAF60C60076B78F959074FF2AD *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
    tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local --owner root --group root --no-same-owner && \
    rm spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

# Mesos dependencies
# Install from the Xenial Mesosphere repository since there does not (yet)
# exist a Bionic repository and the dependencies seem to be compatible for now.
COPY mesos.key /tmp/
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y gnupg && \
    apt-key add /tmp/mesos.key && \
    echo "deb http://repos.mesosphere.io/ubuntu xenial main" > /etc/apt/sources.list.d/mesosphere.list && \
    apt-get -y update && \
    apt-get --no-install-recommends -y install libcurl3 mesos=1.2\* && \
    apt-get purge --auto-remove -y gnupg && \
    rm -rf /var/lib/apt/lists/*

# Spark and Mesos config
ENV SPARK_HOME /usr/local/spark
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip
ENV MESOS_NATIVE_LIBRARY /usr/local/lib/libmesos.so
ENV SPARK_OPTS --driver-java-options=-Xms4096M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info --driver-memory 6g

# Specify local spark operation, override as applicable
ENV \
"NAMENODE"="local[*]" \
"DEPLOY_MODE"="client" \
"SPARK_TF"="spark-tensorflow-connector_2.11-1.10.0.jar"

COPY [ "trader", "data.py", "train.py", "train.sh", "tfrecords.sh", "${SPARK_TF}", "/app/" ]

# Data source dir and TFRecord output dir
# Execution artifacts (logs/checkpoints)
ENV \
"SRC_DIR"="/mnt/data/src" \
"DEST_DIR"="/mnt/data/dest" \
"ARTIFACTS_DIR"="/mnt/artifacts"

VOLUME ["${ARTIFACTS_DIR}", "${SRC_DIR}", "${DEST_DIR}"]

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

ENTRYPOINT [ "/bin/sh" ]
