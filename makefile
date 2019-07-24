.PHONY: build build-example doc test clean dataset build-dataset

IMG_NAME='trader'
LIB_NAME='trader'

DATA_SRC="/mnt/iscsi/amex-nyse-nasdaq-stock-histories/full_history"
DATA_DEST='/mnt/iscsi/tfrecords2'
ARTIFACTS_DIR='/mnt/iscsi/artifacts'

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -wholename '*/.pytest_cache' -exec rm -rf {} +

build:
	docker build --tag=${IMG_NAME} --target=base .

train:
	docker run -it \
		--runtime=nvidia \
		-p 0.0.0.0:6006:6006 \
		-v /home/tidal/Documents/trader:/app\
		-v ${DATA_SRC}:/mnt/data/src \
		-v ${DATA_DEST}:/mnt/data/dest \
		-v ${ARTIFACTS_DIR}:/mnt/artifacts \
		${IMG_NAME} /app/train.sh

records:
	docker run -it \
		-v /home/tidal/Documents/trader:/app\
		-v ${DATA_SRC}:/mnt/data/src \
		-v ${DATA_DEST}:/mnt/data/dest \
		${IMG_NAME} '/app/tfrecords.sh'
