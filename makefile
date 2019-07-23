.PHONY: build build-example doc test clean dataset build-dataset

IMG_NAME='trader'
LIB_NAME='trader'

DATA_SRC="/mnt/iscsi/amex-nyse-nasdaq-stock-histories/full_history"
DATA_DEST='/mnt/iscsi/tfrecords'

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -wholename '*/.pytest_cache' -exec rm -rf {} +

build:
	docker build --tag=${IMG_NAME} --target=base .

train:
	docker run -it \
		--runtime=nvidia \
		-v /home/tidal/Documents/trader:/app\
		-v ${DATA_SRC}:/app/data/src \
		-v ${DATA_DEST}:/app/data/dest \
		${IMG_NAME} -c "python /app/train.py"

records:
	docker run -it \
		-v /home/tidal/Documents/trader:/app\
		-v ${DATA_SRC}:/app/data/src \
		-v ${DATA_DEST}:/app/data/dest \
		${IMG_NAME} '/app/tfrecords.sh'
