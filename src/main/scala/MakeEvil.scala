import dzq.utils.ArgumentParse

import java.io.{File, PrintWriter}
import org.apache.hadoop.hive.metastore.api.Schema
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
//import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object MakeEvil {
  def getPreDate(daysAgo: Int): String = {
    import java.util.Calendar
    val dateFmt = new java.text.SimpleDateFormat("yyyyMMdd")
    val curTime = Calendar.getInstance()
    curTime.add(Calendar.DAY_OF_MONTH, daysAgo)
    dateFmt.format(curTime.getTime)
  }

  def evilList(uinEvilList: String) = {
    val evilListSize = 205
    val newList = new Array[Int](evilListSize)
    uinEvilList.split(";")
      .map(kvStr => kvStr.split(":"))
      .foreach{kvAry =>
        if (kvAry.length == 2) {if (kvAry(1).length < 6) newList(kvAry(0).toInt -1) = kvAry(1).toInt}
      }
    newList
  }

  def trainData(spark:SparkSession, dateStr:String) = {
    import spark.implicits._
    import spark.sql

    val sqlStr = s"select uin_, source_, valuelist, timestamp_ from log_spaminfo where ds='$dateStr' and keytype_=1 and source_ in (0,10,20,21)"
    val parsedData = sql(sqlStr).rdd
      .map(row => (row.getLong(0), (row.getLong(1), row.getString(2), row.getLong(3))))
      .reduceByKey((preV, curV) => if (curV._3 > preV._3) curV else preV)
      .map{case (k, v) =>
        val et = if (v._1 > 0) 1 else 0
        val el = evilList(v._2).map(_.toDouble)
        (k, LabeledPoint(et.toDouble, Vectors.dense(el)), v)
      }
    parsedData
  }

  def main(args: Array[String]) {
    val cfg = new ArgumentParse(args)

    val sDataDate = cfg.getString("DataDate")

    val spark = SparkSession.builder().appName("svm_test")
      .config("spark.sql.warehouse.dir", new File("spark-warehouse").getAbsolutePath)
      .enableHiveSupport()
      .getOrCreate()

    val appCfg = spark.sparkContext.broadcast(cfg)

    //val data = trainData(sc, getPreDate(-1))
    val data = trainData(spark, sDataDate)
    val parsedData = data.filter(row => row._3._3<appCfg.value.getInt("BeginTime") || row._3._3>appCfg.value.getInt("EndTime")).map(row => row._2)
    val numIterations = 4
    val model = SVMWithSGD.train(parsedData, numIterations)
    //model.clearThreshold

    val predictData = data.filter(row => row._3._3>=appCfg.value.getInt("BeginTime") && row._3._3<=appCfg.value.getInt("EndTime"))
    val labelAndPreds = predictData.map{point =>
      val prediction = model.predict(point._2.features)
      (point._1, point._2.label, prediction, point._3)
    }
    val predictErr = labelAndPreds.filter(r => r._2 != r._3)
    val tp = labelAndPreds.filter(r => r._2 >  0 && r._3 >  0).count
    val fp = labelAndPreds.filter(r => r._2 <= 0 && r._3 >  0).count
    val fn = labelAndPreds.filter(r => r._2 >  0 && r._3 <= 0).count

    val saveFile = "./result.log"
    val outPut = new PrintWriter(saveFile)

    outPut.println("training data:" + parsedData.count)
    outPut.println("prediction data:" + predictData.count)
    outPut.println("prediction err:" + predictErr.count)
    //outPut.println("predic ERROR:" + predictErr.count.toDouble / predictData.count)
    outPut.println("precision : %.4f(%d, %d)".format(tp.toDouble / (tp + fp), tp, fp))
    outPut.println("recall : %.4f(%d)".format(tp.toDouble / (tp + fn), fn))
    //val predictErrData = predictErr.collect()
    //predictErrData.foreach(row => outPut.println(Array(row._1.toString, row._2.toString, row._3.toString, row._4.toString).mkString(",")))
    outPut.close()

    spark.stop
  } // end main
}