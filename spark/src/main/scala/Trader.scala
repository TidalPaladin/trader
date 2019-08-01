import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import scopt.OParser

import org.apache.spark.sql.types._
import org.apache.log4j.{Level, Logger}

case class Config(
    out: String = "",
    date: Int = 2010,
    quantize: Int = 10,
    bucketize: Seq[Int] = Seq(),
    in: String = "",
    tfrecord: Boolean = false,
    csv: Boolean = false
)

object Trader {

  @transient lazy val log = Logger.getLogger(getClass.getName)
  val spark = SparkSession.builder.appName("trader").getOrCreate()
  val sc = spark.sparkContext
  import spark.implicits._

  def run(config: Config) {

    val raw_df = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(config.in)
      .withColumn("symbol", hash(input_file_name()))

    val df = raw_df
      .filter(year($"date") > config.date)
      .withColumn("position", sin(lit(3.14) * dayofyear($"date") / lit(365)))
      .withColumn("adjhigh", $"adjclose" / $"close" * $"high")
      .withColumn("adjlow", $"adjclose" / $"close" * $"low")
      .withColumn("adjopen", $"adjclose" / $"close" * $"open")
      .drop("high", "low", "open", "close")
      .withColumnRenamed("adjclose", "close")
      .withColumnRenamed("adjopen", "open")
      .withColumnRenamed("adjhigh", "high")
      .withColumnRenamed("adjlow", "low")


    val future_df = df.select(
      $"symbol".as("f_symbol"),
      $"date".as("f_date"),
      $"close".as("f_close")
    )

    val change_df = df
      .join(future_df, $"symbol" === $"f_symbol")
      .filter(date_add($"date", 1) === $"f_date")
      .withColumn("change", ($"f_close" - $"close") / $"close" * lit(100))
      .drop("f_symbol", "f_date", "f_close")

    val feature_cols = Array("high", "low", "open", "close", "volume", "position")

    val vectorizer = new VectorAssembler()
      .setInputCols(feature_cols)
      .setOutputCol("raw_features")

    val standardizer = new MaxAbsScaler()
      .setInputCol("raw_features")
      .setOutputCol("features")

    val labeler = new QuantileDiscretizer()
      .setInputCol("change")
      .setOutputCol("label")
      .setNumBuckets(config.quantize)

    val stages = Array(vectorizer, standardizer, labeler)
    val pipeline = new Pipeline().setStages(stages)

    val result_df = pipeline
      .fit(change_df)
      .transform(change_df)
      .repartition($"symbol")
      .sortWithinPartitions(desc($"date"))
      .cache()

    val display_df = result_df.select("symbol", "date", "features", "label", "change")

    display_df.show()
    display_df.printSchema

    display_df.drop($"features").describe().show()
    display_df.groupBy("label").count().show()


    if(config.tfrecord) {
      log.info("Writing TFRecords")

    }

    if(config.csv) {
      log.info("Writing CSV files")

      display_df
    }

  }

  def main(args: Array[String]) {
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

        opt[Unit]("csv")
          .action((_, c) => c.copy(csv = true))
          .text("write final dataset to CSV files"),

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

