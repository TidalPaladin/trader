${SPARK_HOME}/bin/spark-submit \
	--master local[4] \
	--deploy-mode ${DEPLOY_MODE} \
	--jars /app/${SPARK_TF} \
	--conf "spark.cores.max=6" \
	--conf "spark.executor.cores=2" \
	--driver-memory 8G \
	/app/data.py
