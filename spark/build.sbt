name := "trader"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.3"

assemblyMergeStrategy in assembly := {
	case "META-INF/services/org.apache.spark.sql.sources.DataSourceRegister" => MergeStrategy.concat
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

mainClass in assembly := Some("Trader")
fullClasspath in assembly := (fullClasspath in Runtime).value

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
	"org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
	"org.tensorflow" %% "spark-tensorflow-connector" % "1.14.0",
	"com.github.scopt" % "scopt_2.11" % "4.0.0-RC2"
)
