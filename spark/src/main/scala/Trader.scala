import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import scopt.OParser
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.linalg.{Vector, VectorUDT, DenseVector}
import org.apache.spark.sql.expressions.Window

import org.apache.spark.sql.types._
import org.apache.log4j.{Level, Logger}

case class Config(
    out: String = "",
    date: Int = 2010,
    quantize: Int = 10,
    bucketize: Seq[Int] = Seq(),
    in: String = "",
    tfrecord: Boolean = false,
    past: Int = 128,
    shards: Int = 100
)

object Trader {

  @transient lazy val log = Logger.getLogger(getClass.getName)
  val spark = SparkSession
    .builder
    .appName("trader")
    .getOrCreate()

  val sc = spark.sparkContext
  import spark.implicits._

  val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
  val vec_to_array = udf(toArr)

  def run(config: Config) {

    val schema = StructType(
      StructField("date", DateType) ::
      StructField("volume", IntegerType) ::
      StructField("open", FloatType) ::
      StructField("close", FloatType) ::
      StructField("high", FloatType) ::
      StructField("low", FloatType) ::
      StructField("adjclose", FloatType) :: Nil)

    /* Read raw CSV, zip with hash value indicating file source */
    val raw_df = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv(config.in)
      .withColumn("symbol", hash(input_file_name()))

    /* Column to encode day of year with a sin function */
    val positional_encoding = sin(lit(3.14) * dayofyear($"date") / lit(365))

    /* Use ratio of adjclose / close to rescale other price metrics */
    val rescale_ratio = $"adjclose" / $"close"

    /* Filter by date cutoff, rescale prices, create positional encoding column */
    val df = raw_df
      .filter(year($"date") > config.date)
      .withColumn("position", positional_encoding)
      .withColumn("adjhigh", rescale_ratio * $"high")
      .withColumn("adjlow", rescale_ratio * $"low")
      .withColumn("adjopen",rescale_ratio * $"open")
      .drop("high", "low", "open", "close")
      .withColumnRenamed("adjclose", "close")
      .withColumnRenamed("adjopen", "open")
      .withColumnRenamed("adjhigh", "high")
      .withColumnRenamed("adjlow", "low")


    /* Select future columns */
    val future_df = df.select(
      $"symbol".as("f_symbol"),
      $"date".as("f_date"),
      $"close".as("f_close")
    )

    /* Calculate percent change from present to future */
    val change_df = df
      .join(future_df, $"symbol" === $"f_symbol")
      .filter(date_add($"date", 1) === $"f_date")
      .withColumn("change", ($"f_close" - $"close") / $"close" * lit(100))
      .drop("f_symbol", "f_date", "f_close")

    val feature_cols = Array("high", "low", "open", "close", "volume", "position")

    /* Vectorizer collects features into a vector */
    val vectorizer = new VectorAssembler()
      .setInputCols(feature_cols)
      .setOutputCol("raw_features")

    /* Per-feature standardization / scale */
    val standardizer = new MaxAbsScaler()
      .setInputCol("raw_features")
      .setOutputCol("features")

    /* Generate labels based on percent change quantile */
    val labeler = new QuantileDiscretizer()
      .setInputCol("change")
      .setOutputCol("label")
      .setNumBuckets(config.quantize)

    val stages = Array(vectorizer, standardizer, labeler)
    val pipeline = new Pipeline().setStages(stages)

    /* Run pipeline and clean up output types */
    val raw_result = pipeline
      .fit(change_df)
      .transform(change_df)
      .withColumn("cast", $"change".cast(FloatType))
      .drop($"change")
      .withColumnRenamed("cast", "change")
      .withColumn("cast", $"label".cast(IntegerType))
      .drop($"label")
      .withColumnRenamed("cast", "label")
      .cache()

    /* Generate an output DataFrame and show results */
    val display_df = raw_result.select("symbol", "date", "features", "label", "change")
    display_df.show()
    display_df.printSchema
    display_df.drop($"features").describe().show()
    display_df.groupBy("label").count().show()


    /* If requested, write TFRecords */
    if(config.tfrecord) {
      log.info("Writing TFRecords")

      /* Window function to collect historical prices */
      val past_window = Window
        .partitionBy("symbol")
        .orderBy(desc("date"))
        .rowsBetween(Window.currentRow - config.past + 1, Window.currentRow)

      /* Collect features over historical window to a matrix */
      val dense_df = raw_df
        .select($"symbol", $"date", $"features", $"label", $"change")
        .withColumn("dense_features", collect_list(vec_to_array($"features")).over(past_window))
        .drop("features", "date", "symbol")
        .withColumnRenamed("dense_features", "features")
        .filter(size($"features") === config.past)
        .cache()

      dense_df.printSchema

      dense_df
        .repartition(config.shards)
        .write
        .format("tfrecords")
        .mode("overwrite")
        .option("recordType", "SequenceExample")
        .save(config.out + "/tfrecords")
    }
  }

  def main(args: Array[String]) {
    Logger.getRootLogger.setLevel(Level.WARN)
    log.info("Started trader")

    val builder = OParser.builder[Config]
    val parser1 = {
      import builder._
      OParser.sequence(
        programName("trader"),
        head("trader", "1.0"),

        opt[String]('o', "out")
          .valueName("<path>")
          .action((x, c) => c.copy(out = x))
          .text("output file path"),

        opt[Int]('d', "date")
          .valueName("<year>")
          .validate(x =>
            if (x > 0) success
            else failure("Value <year> must be >0")
          )
          .action((x, c) => c.copy(date = x))
          .text("limit to records newer than <date>"),

        opt[Int]('p', "past")
          .valueName("<int>")
          .validate(x =>
            if (x > 0) success
            else failure("Value <int> must be >0")
          )
          .action((x, c) => c.copy(date = x))
          .text("aggregate past <int> records into an example"),

        opt[Int]('s', "shards")
          .valueName("<int>")
          .validate(x =>
            if (x > 0) success
            else failure("Value <int> must be >0")
          )
          .action((x, c) => c.copy(date = x))
          .text("split the TFRecord set into <int> shards"),

        opt[Int]("quantize")
          .valueName("<int>")
          .action((x, c) => c.copy(quantize = x))
          .text("run QuantileDiscretizer to <int> buckets"),

        opt[Seq[Int]]("bucketize")
          .valueName("<float, float, float...>")
          .action((x, c) => c.copy(bucketize = x))
          .text("run Bucketizer with the given buckets"),

        opt[Unit]("tfrecord")
          .action((_, c) => c.copy(tfrecord = true))
          .text("write final dataset to TFRecord files"),

        help("help").text("prints this usage text"),

        arg[String]("<file>")
          .required()
          .valueName("<shell_glob>")
          .action((x, c) => c.copy(in = x))
          .text("input file shell glob pattern")
      )
    }
    OParser.parse(parser1, args, Config()) match {
      case Some(config) => run(config)
      case _ => log.error("Stopping")
    }
  }
}
