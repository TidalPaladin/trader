import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer, UnaryTransformer}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, VectorUDT, DenseVector}
import org.apache.spark.sql.expressions.Window
import scopt.OParser
import org.apache.spark.sql.types._
import org.apache.log4j.{Level, Logger}

object Trader {

  @transient lazy val log = Logger.getLogger(getClass.getName)
  log.info("Writing TFRecords")
  val spark = SparkSession.builder.appName("trader").getOrCreate()

  val sc = spark.sparkContext
  import spark.implicits._

  /* Define UDF for Vector to Array conversion */
  val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
  val vec_to_array = udf(toArr)

  val schema = StructType(
    StructField("date", DateType) ::
    StructField("volume", IntegerType) ::
    StructField("open", FloatType) ::
    StructField("close", FloatType) ::
    StructField("high", FloatType) ::
    StructField("low", FloatType) ::
    StructField("adjclose", FloatType) :: Nil)


  /* Column to encode day of year with a sin function */
  val positionalEncoding: DataFrame => DataFrame = {
    _.withColumn("position", sin(lit(3.14) * dayofyear('date) / lit(365)))
  }

  /* Use ratio of adjclose / close to rescale other price metrics */
  val rescalePrices: DataFrame => DataFrame = {
    val targets = Seq("high", "low", "open")
    val ratio = 'adjclose / 'close

    /* Create new columns for each in targets, drop old cols and rename new cols */
    _.select($"*" +: targets.map(c => (col(c) * ratio).alias(s"adj$c")): _*)
     .drop("close" +: targets: _*)
     .select($"*"+: $"adjclose".alias("close") +: targets.map(c => col(s"adj$c").alias(c)): _*)
  }

  /* Calculate percent change in future close price by user supplied window */
  val getPercentChange: DataFrame => DataFrame = {
    val window = Window
      .partitionBy('symbol)
      .orderBy("date")
      .rowsBetween(Window.currentRow, Window.currentRow + 5)

    val change_col = (avg($"close").over(window) - $"close") / $"close" * lit(100)

    _.withColumn("change", change_col)
  }


  def main(args: Array[String]) {
    Logger.getRootLogger.setLevel(Level.WARN)
    log.info("Started trader")

    OParser.parse(TraderArgs.parser1, args, Config()) match {
      case Some(config) => run(config)
      case _ => log.error("Stopping")
    }
  }

  def run(config: Config) {

    val filterDates: DataFrame => DataFrame = _.filter(year($"date") > config.date)

		/* Filter by absolute value of percent change if set in CLI flag */
    val filterByChange: DataFrame => DataFrame = config.max_change match {
      case Some(x) => {(df) => df.filter(abs($"change") <= x)}
      case _ => {(df) => df}
    }


    val feature_cols = Array("high", "low", "open", "close", "volume", "position")

    /* Vectorizer collects features into a vector */
    val vectorizer = new VectorAssembler()
      .setInputCols(feature_cols)
      .setOutputCol("raw_features")

    /* Per-feature standardization / scale */
    val norm = config.norm match {
      case Some(x: MaxAbsScaler) => x.setInputCol("raw_features").setOutputCol("features")
      case Some(x: StandardScaler) => x.setInputCol("raw_features").setOutputCol("features").setWithMean(true).setWithStd(true)
			case _ => None
    }

    /* Choose a label pipeline stage based on CLI flags */
    val labeler = config.quantize match {
				case Some(x) => new QuantileDiscretizer().setNumBuckets(x).setInputCol("change").setOutputCol("label")
				case _ => config.bucketize match {
					case Some(x) => new Bucketizer().setSplits(x).setInputCol("change").setOutputCol("label")
					case _ => None
			}
		}

    val stages = Array(vectorizer, norm, labeler).map{
			case Some(x: PipelineStage) => x
			case x: PipelineStage => x
		}
    val pipeline = new Pipeline().setStages(stages)

    /* Read raw CSV, zip with hash value indicating file source */
    val raw_df = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv(config.in)
      .withColumn("symbol", hash(input_file_name()))


    /* Define a preprocessing pipeline and apply to raw df */
		val results_df = filterDates
      .andThen(getPercentChange)
      .andThen(filterByChange)
      .andThen(positionalEncoding)
			.apply(raw_df)
			.cache()

    /* Run Spark pipeline for feature extraction */
		val df = pipeline
			.fit(results_df)
      .transform(results_df)
      .cache()

		results_df.unpersist()


    /* Generate an output DataFrame and show results */
    val display_df = df.select("symbol", "date", "features", "label", "change")

    display_df.show()
    display_df.printSchema

    df.select("label", "change" +: feature_cols: _*).describe().show()

    display_df
      .groupBy('label)
      .agg(count("label").as("count"), avg("change").as("avg_change"))
      .orderBy(desc("label"))
      .show()


		/* Performs rollup of features over a historical window to a nested array */
		val getFeatureMatrix: DataFrame => DataFrame = {

			/* Window function to collect historical prices */
			val past_window = Window
				.partitionBy("symbol")
				.orderBy(desc("date"))
				.rowsBetween(Window.currentRow - config.past + 1, Window.currentRow)

			val collected_col = collect_list(vec_to_array($"features")).over(past_window)

			_.withColumn("dense_features", collected_col)
			.drop("features")
			.withColumnRenamed("dense_features", "features")
			.filter(size($"features") === config.past)
		}


    /* If requested, write TFRecords */
    config.out match {
      case Some(path) => {
        log.info("Writing TFRecords")


        /* Collect features over historical window to a matrix */
        val dense_df = getFeatureMatrix
					.apply(df.select($"symbol", $"date", $"features", $"label", $"change"))
          .drop( "date", "symbol")
					.withColumn("cast", $"change".cast(FloatType))
					.drop($"change")
					.withColumnRenamed("cast", "change")
					.withColumn("cast", $"label".cast(IntegerType))
					.drop($"label")
					.withColumnRenamed("cast", "label")
          .cache()

        dense_df.printSchema

        dense_df
          .repartition(config.shards)
          .write
          .format("tfrecords")
          .mode("overwrite")
          .option("recordType", "SequenceExample")
          .save(path + "/tfrecords")
      }
      case _ => Unit
    }
  }

}
