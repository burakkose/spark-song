package evaluation

import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

class MeanAveragePrecision(model: MatrixFactorizationModel,
                           calculateRDD: RDD[(Int, Iterable[Rating])]) {

  def meanAveragePrecision(truncated: Int): Double = {
    val predictionsForUsers = model.recommendProductsForUsers(truncated)
    val tmp: RDD[(Int, Array[Rating])] = predictionsForUsers.map(tup => (tup._1, tup._2.map(r => Rating(r.user, r.product, if (r.rating > 1) 1 else if (r.rating < 0) 0 else r.rating))))
    val relevantDocuments = calculateRDD.join(tmp).map {
      case (user, (actual, predictions)) =>
        (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
    }
    new RankingMetrics(relevantDocuments).meanAveragePrecision
  }

}

object MeanAveragePrecision {
  def apply(model: MatrixFactorizationModel,
            calculateRDD: RDD[(Int, Iterable[Rating])]): MeanAveragePrecision = {
    new MeanAveragePrecision(model, calculateRDD)
  }

}
