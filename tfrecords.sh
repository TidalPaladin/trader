${SPARK_HOME}/bin/spark-submit \
	--master ${NAMENODE} \
	--deploy-mode ${DEPLOY_MODE} \
	--jars /app/${SPARK_TF} \
	/app/data.py
