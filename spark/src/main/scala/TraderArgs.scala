/**
 Handles argument parsing for Trader main() method
*/

import scopt.OParser
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature._

case class Config(
    in: String = "",
    out: Option[String] = None,
    date: Int = 2010,
    past: Int = 128,
    future: Int = 1,
    quantize: Option[Int] = None,
    bucketize: Option[Array[Double]] = None,
    shards: Int = 100,
    max_change: Option[Double] = None,
    norm: Option[PipelineStage] = None,
    penny: Boolean = false,
    stride: Int = 5
)

object TraderArgs {

  val builder = OParser.builder[Config]
  val parser1 = {
    import builder._
    OParser.sequence(
      programName("trader"),
      head("trader", "1.0"),

      opt[String]('o', "out")
        .valueName("<path>")
        .action((x, c) => c.copy(out = x match {
          case x: String => Some(x)
          case _ => None
        }))
        .text("output file path"),

      opt[String]('n', "norm")
        .valueName("std, maxabs")
        .validate(x =>
          x match {
            case "std" | "maxabs" => success
            case _ => failure("norm must be one of std,maxabs")
          }
        )
        .action((x, c) => c.copy(norm = x match {
          case "std" => Some(new StandardScaler())
          case "maxabs" => Some(new MaxAbsScaler())
          case _ => None
        }))
        .text("Normalization strategy, StandardScaler or MaxAbsScaler"),

      opt[Int]('d', "date")
        .valueName("<year>")
        .validate(x =>
          if (x > 0) success
          else failure("Value <year> must be >0")
        )
        .action((x, c) => c.copy(date = x))
        .text("limit to records newer than <year>"),

      opt[Int]('p', "past")
        .valueName("<int>")
        .validate(x =>
          if (x > 0) success
          else failure("Value <int> must be >0")
        )
        .action((x, c) => c.copy(past = x))
        .text("aggregate past <int> records into an example"),

      opt[Int]("stride")
        .valueName("<int>")
        .validate(x =>
            if(x > 0) success
            else failure("Value <int> must be >0")
        )
        .action((x, c) => c.copy(stride = x))
        .text("stride training example window by <int> days"),

      opt[Int]('f', "future")
        .valueName("<int>")
        .validate(x =>
          if (x > 0) success
          else failure("Value <int> must be >0")
        )
        .action((x, c) => c.copy(future = x))
        .text("calculate percent change over <int> following days. if >1, use averaging"),

      opt[Double]("max-change")
        .valueName("<float>")
        .validate(x =>
          if (x > 0) success
          else failure("Value <float> must be >0")
        )
        .action((x, c) => c.copy(max_change = Some(x)))
        .text("drop examples with absolute percent change > <float>"),

      opt[Int]('s', "shards")
        .valueName("<int>")
        .validate(x =>
          if (x > 0) success
          else failure("Value <int> must be >0")
        )
        .action((x, c) => c.copy(shards = x))
        .text("split the TFRecord set into <int> shards"),

      opt[Int]("quantize")
        .valueName("<int>")
        .validate(x =>
            if(x > 1) success
            else failure("Value <int> must be >1")
        )
        .action((x, c) => c.copy(quantize = Some(x)))
        .text("run QuantileDiscretizer to <int> buckets"),

      opt[Seq[Double]]("bucketize")
        .valueName("<float>,[float,...]")
        .action((x, c) => c.copy(bucketize = Some(
					Array(Double.NegativeInfinity +: x :+ Double.PositiveInfinity : _*)
				)))
        .text("run Bucketizer with the given buckets"),

      opt[Unit]("penny-stocks")
        .text("if set, allow penny stocks. default false"),

      help("help").text("prints this usage text"),

      note("Note: TFRecords will be written if output file path is specified" + sys.props("line.separator")),
      note("Writing many TFRecords has a high RAM requirement." + sys.props("line.separator")),

      arg[String]("<shell_glob>")
        .required()
        .valueName("<shell_glob>")
        .action((x, c) => c.copy(in = x))
        .text("input file shell glob pattern")
    )
  }
}
