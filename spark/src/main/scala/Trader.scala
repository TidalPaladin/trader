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

  val feature_cols = Array("high", "low", "open", "close", "volume", "position")

  /* Schema of source dataset */
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
     .drop("adjclose" +: targets.map(c => s"adj$c"): _*)
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

  /* Discard dates older than a threshold year */
  val filterByDate = (thresh: Int, df: DataFrame) => {
    df.filter(year($"date") > thresh)
  }

  /* Filter where abs(percent change) > thresh */
  val filterByChange = (thresh: Double, df: DataFrame) => {
    df.filter(abs($"change") <= thresh)
  }

  /**
   Filter where groupBy(symbol).avg(close) < some cutoff.
   Here use a value slightly higher than 1.0 for penny stocks
  */
  val filterPennyStocks = (df: DataFrame) => {
    df.groupBy('symbol)
      .agg(avg('close).as("avg_close"))
      .filter('avg_close > 0.8)
      .withColumnRenamed("symbol", "symbol2")
      .join(df, $"symbol" === $"symbol2")
      .drop("symbol2", "avg_close")
  }

  /* Performs rollup of features over a historical window to a nested array */
  val getFeatureMatrix = (past: Int, stride: Int, df: DataFrame) => {

    /* Window function to collect historical prices */
    val past_window = Window
      .partitionBy("symbol")
      .orderBy(desc("date"))
      .rowsBetween(Window.currentRow - past + 1, Window.currentRow)

    val collected_col = collect_list(vec_to_array($"features")).over(past_window)

    df.withColumn("dense_features", collected_col)
      .drop("features")
      .withColumnRenamed("dense_features", "features")
      .filter(size($"features") === past)
      .filter((dayofyear('date) % stride) === 0)
  }


  /* Recast multiple DataFrame columns given a map of column names to types*/
  val recastColumns = (m: Map[String, DataType], df: DataFrame) => {
     val oldKeys = m.keySet.toSeq
     val tempKeys = oldKeys.map(c => col(c).cast(m(c)).alias(s"cast$c"))
     val newKeys = oldKeys.map(c => col(s"cast$c").alias(c))

      df.select($"*" +: tempKeys : _*)
        .drop(oldKeys: _*)
        .select($"*" +: newKeys: _*)
        .drop(oldKeys.map(c => s"cast$c"): _*)
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

    /* Writes to CSV given subdir and DataFrame based on config.out path */
    def writeCsv: (String, DataFrame) => Unit = config.out match {
      case Some(x) => (subdir, df) => {
        df.repartition(1)
          .write.mode("overwrite").option("header", "true")
          .csv(x + "/" + subdir)
      }
      case _ => (subdir, df) => Unit
    }

    /* Vectorizer collects features into a vector */
    val vectorizer = new VectorAssembler().setInputCols(feature_cols).setOutputCol("raw_features")

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
			case Some(x: PipelineStage) => Some(x)
			case x: PipelineStage => Some(x)
      case None => None
		}.flatten
    val pipeline = new Pipeline().setStages(stages)

    /* Read raw CSV, zip with hash value indicating file source */
    val raw_df = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv(config.in)
      .withColumn("symbol", hash(input_file_name()))

    val results_df = ((df: DataFrame) => filterByDate(config.date, df))
      .andThen(filterPennyStocks)
      .andThen(getPercentChange)
      .andThen(config.max_change match {
        case Some(x) => (df: DataFrame) => filterByChange(x, df)
        case _ => (df: DataFrame) => df
      })
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

    println("Processed DataFrame before feature matrix rollup:")
    display_df.show()
    display_df.printSchema

    println("Processed DataFrame label stats")
    val raw_stats_df = df.select("label", "change" +: feature_cols: _*).describe()
    raw_stats_df.show()
    writeCsv("stats", raw_stats_df)

    println("Average percent change within a label:")
    display_df
      .groupBy('label)
      .agg(count("label").as("count"), avg("change").as("avg_change"))
      .orderBy(desc("label"))
      .show()

   /* Collect features over historical window to a matrix */
   val recast = Map("change" -> FloatType, "label" -> IntegerType)

   /* Perform feature matrix rollup */
   val dense_df = ((df: DataFrame) => getFeatureMatrix(config.past, config.stride, df))
      .andThen((df: DataFrame) => recastColumns(recast, df))
      .apply(df.select($"symbol", $"date", $"features", $"label", $"change"))
      .cache()

    println("Dense DataFrame: Dates should be strided here")
    dense_df.show()

    /* If requested, write TFRecords */
    config.out match {
      case Some(path) => {
        log.info("Writing TFRecords")

        val write_df = dense_df.drop("date", "symbol").repartition(config.shards)
        println("Write DataFrame schema:")
        write_df.printSchema

        println("Write DataFrame metrics:")
        val metric_df = write_df
          .groupBy('label)
          .agg(count("label"), avg("change"), max("change"), min("change"))
          .orderBy(desc("label"))
        metric_df.show()

        writeCsv("metrics", metric_df)

        write_df.write
          .format("tfrecords")
          .mode("overwrite")
          .option("recordType", "SequenceExample")
          .save(path + "/tfrecords")
      }
      case _ => Unit
    }
  }
}
