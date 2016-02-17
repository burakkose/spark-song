import evaluation.MeanAveragePrecision
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}


object RecommendationSystem extends App {

  val conf = new SparkConf()
    .setAppName("movie")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")

  val parseRating: (String) => (String, Int) = (line: String) => {
    line.split("\t") match {
      case Array(userID, _, _, _, _, song) => (userID.split("_")(1) + "\t" + song, 1)
    }
  }

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._

  val dataDF = sc.textFile("/home/whisper/Desktop/dataset/lastfm-dataset-1K/real.tsv")
    .map(parseRating).reduceByKey(_ + _).map {
    case (userAndMusic: String, count: Int) => {
      val Array(userID, music) = userAndMusic.split("\t")
      (userID toInt, music, count toDouble)
    }
  }.toDF("userID", "music", "count").cache

  val music2index = new StringIndexer()
    .setInputCol("music")
    .setOutputCol("musicID")

  val ratingsRDD = music2index.fit(dataDF).transform(dataDF).select("userID", "musicID", "count").map {
    case Row(user: Int, music: Double, count: Double) => Rating(user, music.toInt, count)
  }.cache

  val binarizedRatings = ratingsRDD.map(r => Rating(r.user, r.product,
    if (r.rating > 0) 1.0 else 0.0)).cache()

  def scaledRating(r: Rating): Rating = {
    val scaledRating = math.max(math.min(r.rating, 1.0), 0.0)
    Rating(r.user, r.product, scaledRating)
  }

  /*
  val (trainingRDD, testingRDD) = ratingsRDD.randomSplit(Array(0.7, 0.3)) match {
    case Array(train, test) => (train.cache, test.groupBy(_.user).cache)
  }
  */

  //println(s"Training : ${trainingRDD.count} ratings, test : ${testingRDD.count} users")
  val userMovies = binarizedRatings.groupBy(_.user)
  /*
  val (bestModel, bestTestMPA, bestRank, bestLambda, bestNumIter, bestAlpha) = {
    for (rank <- Array(8, 12);
         lambda <- Array(0.01, 10);
         numIter <- Array(10, 20);
         alpha <- Array(0.01)
    ) yield {
      val model = ALS.trainImplicit(ratingsRDD, rank, numIter, lambda, alpha)
      val testMPA = MeanAveragePrecision(model, userMovies).meanAveragePrecision(500)
      println(s"MPA(TEST)= $testMPA rank= $rank, lambda=$lambda, numIter= $numIter, alpha=$alpha")
      (model, testMPA, rank, lambda, numIter, alpha)
    }
  }.minBy(_._2)

  println(s" BEST MPA(TEST)= $bestTestMPA rank= $bestRank, lambda=$bestLambda, numIter= $bestNumIter, alpha=$bestAlpha")
*/
  val model = ALS.trainImplicit(ratingsRDD, 15, 10, 0.01, -1, 0.01)
  val testMPA = MeanAveragePrecision(model, userMovies).meanAveragePrecision(500)
  println(s"MPA(TEST)= $testMPA")
  sc.stop
}