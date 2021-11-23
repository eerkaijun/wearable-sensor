package com.specknet.pdiotapp.live

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.specknet.pdiotapp.R
import com.specknet.pdiotapp.utils.Constants
import com.specknet.pdiotapp.utils.RESpeckLiveData
import com.specknet.pdiotapp.utils.ThingyLiveData
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.collections.ArrayList
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.acos
import kotlin.math.PI
import kotlin.math.abs


class LiveDataActivity : AppCompatActivity() {

    var lastMagnitude = 0.0f
    var stepCount = 0

    var inputValue = Array(1) {
        Array(2) {
            Array(1) {
                Array(25) {
                    FloatArray(6)
                }
            }
        }
    }
    var outputValue = Array(1) {
        FloatArray(5)
    }
    var bufferCount = 0

    // tflite interpreter to make real-time prediction
    lateinit var tflite: Interpreter

    //textviews
    lateinit var respeckTextView: TextView
    lateinit var stepCountView: TextView

    lateinit var imageView: ImageView
    // global graph variables
    lateinit var dataSet_res_accel_x: LineDataSet
    lateinit var dataSet_res_accel_y: LineDataSet
    lateinit var dataSet_res_accel_z: LineDataSet

    lateinit var dataSet_thingy_accel_x: LineDataSet
    lateinit var dataSet_thingy_accel_y: LineDataSet
    lateinit var dataSet_thingy_accel_z: LineDataSet

    var time = 0f
    lateinit var allRespeckData: LineData

    lateinit var allThingyData: LineData

    lateinit var respeckChart: LineChart
    lateinit var thingyChart: LineChart

    // global broadcast receiver so we can unregister it
    lateinit var respeckLiveUpdateReceiver: BroadcastReceiver
    lateinit var thingyLiveUpdateReceiver: BroadcastReceiver
    lateinit var looperRespeck: Looper
    lateinit var looperThingy: Looper

    val filterTestRespeck = IntentFilter(Constants.ACTION_RESPECK_LIVE_BROADCAST)
    val filterTestThingy = IntentFilter(Constants.ACTION_THINGY_BROADCAST)

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        // val assets: AssetManager = this.getAssets()
        val fileDescriptor = this.assets.openFd("model_conv_lstm.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun stepCounterWalking(x: Float, y: Float, z: Float){
        val magnitude = sqrt((x*x + y*y + z*z))
        val delta = lastMagnitude - magnitude
        lastMagnitude = magnitude

        if(delta > 1.00002)  stepCount++
    }

    fun stepCounterRunning(x: Float, y: Float, z: Float){
        val magnitude = sqrt(x*x + y*y + z*z)
        val delta = lastMagnitude - magnitude
        lastMagnitude = magnitude

        if(delta > 10) stepCount++
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //to update the live data layout
        setContentView(R.layout.activity_live_data)
        respeckTextView = findViewById<TextView>(R.id.recognisedActivity)
        stepCountView = findViewById<TextView>(R.id.stepCounter)
        imageView = findViewById<ImageView>(R.id.activityIcon)

        setupCharts()

        tflite = Interpreter(loadModelFile())
        Log.i("READ MODEL ", "SUCCESSFUL")

        // set up the broadcast receiver
        respeckLiveUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {

                Log.i("thread", "I am running on thread = " + Thread.currentThread().name)

                val action = intent.action

                if (action == Constants.ACTION_RESPECK_LIVE_BROADCAST) {

                    val liveData =
                        intent.getSerializableExtra(Constants.RESPECK_LIVE_DATA) as RESpeckLiveData
                    Log.d("Live", "onReceive: liveData = " + liveData)

                    // get all relevant intent contents
                    val x = liveData.accelX
                    val y = liveData.accelY
                    val z = liveData.accelZ

                    // Build a buffer with intervals of 2 seconds (25Hz)
                    if (bufferCount >= 25) {
                        // do model prediction
                        tflite.run(inputValue, outputValue)
                        Log.i("Predicted live data", outputValue.contentDeepToString())
                        val maxIdx = outputValue[0].indices.maxBy { outputValue[0][it] } ?: -1

                        when(maxIdx) {
                            0 -> {
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Falling"
                                    imageView.setBackgroundResource(R.drawable.falling_icon)
                                })
                            }
                            1 -> {
                                var text = " "
                                var sumGyroY = 0.0
                                var countGyroY = 0

                                var sumX = 0.0
                                var sumY = 0.0
                                var sumZ = 0.0
                                for (i in 0..1) {
                                    for (j in 0..24) {
                                        sumX += inputValue[0][i][0][j][0]
                                        sumY += inputValue[0][i][0][j][1]
                                        sumZ += inputValue[0][i][0][j][2]
                                        if (inputValue[0][i][0][j][4] > -10 && inputValue[0][i][0][j][4] < 10){
                                            sumGyroY += inputValue[0][i][0][j][4]
                                            countGyroY += 1
                                        }
                                    }
                                }
                                val meanX = sumX / 50
                                val meanY = sumY / 50
                                val meanZ = sumZ / 50
                                val meanGyroY = sumGyroY / countGyroY

                                var stdX = 0.0
                                var stdZ = 0.0
                                for (i in 0..1) {
                                    for (j in 0..24) {
                                        stdX += (inputValue[0][i][0][j][0] - meanX).pow(2)
                                        stdZ += (inputValue[0][i][0][j][2] - meanZ).pow(2)
                                    }
                                }
                                stdX = sqrt(stdX / 50)
                                stdZ = sqrt(stdZ / 50)

                                val cosThetaZ = meanZ / (sqrt(meanX.pow(2) + meanY.pow(2) + meanZ.pow(2)))
                                val thetaZ = acos(cosThetaZ) * 180 / PI

                                if (stdX + stdZ > 0.04) {
                                    text = "You are currently: Doing Desk Work"
                                } else {
                                    if (thetaZ > 85.0 && thetaZ < 95.0) {
                                        if (meanGyroY <= 0.85) {
                                            text = "You are currently: Sitting"
                                        } else {
                                            text = "You are currently: Standing"
                                        }
                                    } else if (thetaZ > 0.0 && thetaZ < 85.0) {
                                        text = "You are currently: Sitting bent backward"
                                    } else if (thetaZ > 95.0 && thetaZ < 180.0) {
                                        text = "You are currently: Sitting bent forward"
                                    } else {
                                        text = "You are currently: Doing Desk Work"
                                    }
                                }
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = text
                                    imageView.setBackgroundResource(R.drawable.sitting_icon)
                                })
                            }
                            2 -> {
                                var text = " "
                                var sumX = 0.0
                                var sumY = 0.0
                                var sumZ = 0.0
                                for (i in 0..1) {
                                    for (j in 0..24) {
                                        sumX += inputValue[0][i][0][j][0]
                                        sumY += inputValue[0][i][0][j][1]
                                        sumZ += inputValue[0][i][0][j][2]
                                    }
                                }
                                val meanX = sumX / 50
                                val meanY = sumY / 50
                                val meanZ = sumZ / 50
                                val cosThetaZ = meanZ / (sqrt(meanX.pow(2) + meanY.pow(2) + meanZ.pow(2)))
                                val thetaZ = acos(cosThetaZ) * 180 / PI

                                if (thetaZ in 0.0..45.0) {
                                    text = "You are currently: Lying Down on Back"
                                } else if (thetaZ > 45 && thetaZ <= 90) {
                                    text = "You are currently: Lying Down on Right"
                                } else if (thetaZ > 90 && thetaZ <= 135) {
                                    text = "You are currently: Lying Down on Left"
                                } else if (thetaZ > 135 && thetaZ <= 180) {
                                    text = "You are currently: Lying Down on Stomach"
                                } else {
                                    text = "You are currently: Lying Down on Back"
                                }
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = text
                                    imageView.setBackgroundResource(R.drawable.sitting_icon)
                                })

                                //respeckTextView.text = "You are currently: Lying down"
                            }
                            3 -> {
                                var text = " "
                                var maximum = 0.0
                                var partialSums = DoubleArray(8)
                                for (i in 0..1) {
                                    for (j in 0..3) {
                                        var partialTotal = 0.0
                                        var end = 5*(j+2) -1
                                        for (k in 5*j..end) {
                                            partialTotal += inputValue[0][i][0][k][4]
                                        }
                                        partialSums[i] = partialTotal
                                        if (abs(partialTotal) > maximum) maximum = partialTotal
                                    }
                                }
                                var total = 0.0
                                for (i in 0..7) {
                                    partialSums[i] = partialSums[i] / maximum
                                    total += partialSums[i]
                                }
                                if (abs(total / 8) >= 0.33) {
                                    if (total / 8 < 0) {
                                        text = "You are currently: Descending Stairs"
                                    } else {
                                        text = "You are currently: Climbing Stairs"
                                    }
                                } else {
                                    text = "You are currently: Walking"
                                }

                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = text
                                    imageView.setBackgroundResource(R.drawable.sitting_icon)
                                })



                                //respeckTextView.text = "You are currently: Walking"
                                //mainPageTextView.text = "Recognised activity: Walking"
                                /*stepCounterWalking(x,y,z)
                                var currentCount  = stepCountView.text.toString().toInt() + stepCount
                                stepCountView.text = currentCount.toString()*/

                            }
                            4 -> {
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Running"
                                    imageView.setBackgroundResource(R.drawable.running_icon)
                                })
                                Log.i("DEBUG", stepCount.toString())
                            }
                        }


                        // only reset half of the buffer to make a one second sliding window
                        val temp = inputValue[0][1][0]
                        inputValue[0][0][0] = temp
                        //inputValue[0][0][0].drop(25)
                        //Log.i("Buffer after resetting", inputValue.contentDeepToString());
                        //Log.i("Length of buffer after resetting", inputValue.size.toString());
                        /*
                        inputValue = Array(1) {
                            Array(50) {
                                FloatArray(6)
                            }
                        }*/
                        bufferCount = 0
                    }

                    // shape (1, 2, 1, 25, 6)
                    inputValue[0][1][0][bufferCount][0] = x
                    inputValue[0][1][0][bufferCount][1] = y
                    inputValue[0][1][0][bufferCount][2] = z
                    inputValue[0][1][0][bufferCount][3] = liveData.gyro.x
                    inputValue[0][1][0][bufferCount][4] = liveData.gyro.y
                    inputValue[0][1][0][bufferCount][5] = liveData.gyro.z

                    bufferCount += 1
                    Log.i("Current buffer content", inputValue.contentDeepToString());

                    time += 1
                    updateGraph("respeck", x, y, z)

                }

            }
        }

        // register receiver on another thread
        val handlerThreadRespeck = HandlerThread("bgThreadRespeckLive")
        handlerThreadRespeck.start()
        looperRespeck = handlerThreadRespeck.looper
        val handlerRespeck = Handler(looperRespeck)
        this.registerReceiver(respeckLiveUpdateReceiver, filterTestRespeck, null, handlerRespeck)

        // set up the broadcast receiver
        thingyLiveUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {

                Log.i("thread", "I am running on thread = " + Thread.currentThread().name)

                val action = intent.action

                if (action == Constants.ACTION_THINGY_BROADCAST) {
                    //thingyStatus.text = "Connected"
                    //thingyStatus.setTextColor(Color.parseColor("#008000"))

                    val liveData =
                        intent.getSerializableExtra(Constants.THINGY_LIVE_DATA) as ThingyLiveData
                    Log.d("Live", "onReceive: liveData = " + liveData)

                    // get all relevant intent contents
                    val x = liveData.accelX
                    val y = liveData.accelY
                    val z = liveData.accelZ

                    time += 1
                    updateGraph("thingy", x, y, z)

                }
                /*else{
                    thingyStatus.text = "Disconnected"
                    thingyStatus.setTextColor(Color.parseColor("#ff0000"))

                }*/
            }
        }

        // register receiver on another thread
        val handlerThreadThingy = HandlerThread("bgThreadThingyLive")
        handlerThreadThingy.start()
        looperThingy = handlerThreadThingy.looper
        val handlerThingy = Handler(looperThingy)
        this.registerReceiver(thingyLiveUpdateReceiver, filterTestThingy, null, handlerThingy)

    }

    fun calculateSD(numArray: FloatArray): Double {
        var sum = 0.0
        var standardDeviation = 0.0

        for (num in numArray) {
            sum += num
        }

        val mean = sum / 10

        for (num in numArray) {
            standardDeviation += (num-mean).pow(2.0)
        }

        return sqrt(standardDeviation / 10)
    }


    fun setupCharts() {
        respeckChart = findViewById(R.id.respeck_chart)
        thingyChart = findViewById(R.id.thingy_chart)

        // Respeck

        time = 0f
        val entries_res_accel_x = ArrayList<Entry>()
        val entries_res_accel_y = ArrayList<Entry>()
        val entries_res_accel_z = ArrayList<Entry>()

        dataSet_res_accel_x = LineDataSet(entries_res_accel_x, "Accel X")
        dataSet_res_accel_y = LineDataSet(entries_res_accel_y, "Accel Y")
        dataSet_res_accel_z = LineDataSet(entries_res_accel_z, "Accel Z")

        dataSet_res_accel_x.setDrawCircles(false)
        dataSet_res_accel_y.setDrawCircles(false)
        dataSet_res_accel_z.setDrawCircles(false)

        dataSet_res_accel_x.setColor(
            ContextCompat.getColor(
                this,
                R.color.red
            )
        )
        dataSet_res_accel_y.setColor(
            ContextCompat.getColor(
                this,
                R.color.green
            )
        )
        dataSet_res_accel_z.setColor(
            ContextCompat.getColor(
                this,
                R.color.blue
            )
        )

        val dataSetsRes = ArrayList<ILineDataSet>()
        dataSetsRes.add(dataSet_res_accel_x)
        dataSetsRes.add(dataSet_res_accel_y)
        dataSetsRes.add(dataSet_res_accel_z)

        allRespeckData = LineData(dataSetsRes)
        respeckChart.data = allRespeckData
        respeckChart.invalidate()

        // Thingy

        time = 0f
        val entries_thingy_accel_x = ArrayList<Entry>()
        val entries_thingy_accel_y = ArrayList<Entry>()
        val entries_thingy_accel_z = ArrayList<Entry>()

        dataSet_thingy_accel_x = LineDataSet(entries_thingy_accel_x, "Accel X")
        dataSet_thingy_accel_y = LineDataSet(entries_thingy_accel_y, "Accel Y")
        dataSet_thingy_accel_z = LineDataSet(entries_thingy_accel_z, "Accel Z")

        dataSet_thingy_accel_x.setDrawCircles(false)
        dataSet_thingy_accel_y.setDrawCircles(false)
        dataSet_thingy_accel_z.setDrawCircles(false)

        dataSet_thingy_accel_x.setColor(
            ContextCompat.getColor(
                this,
                R.color.red
            )
        )
        dataSet_thingy_accel_y.setColor(
            ContextCompat.getColor(
                this,
                R.color.green
            )
        )
        dataSet_thingy_accel_z.setColor(
            ContextCompat.getColor(
                this,
                R.color.blue
            )
        )

        val dataSetsThingy = ArrayList<ILineDataSet>()
        dataSetsThingy.add(dataSet_thingy_accel_x)
        dataSetsThingy.add(dataSet_thingy_accel_y)
        dataSetsThingy.add(dataSet_thingy_accel_z)

        allThingyData = LineData(dataSetsThingy)
        thingyChart.data = allThingyData
        thingyChart.invalidate()
    }

    fun updateGraph(graph: String, x: Float, y: Float, z: Float) {
        // take the first element from the queue
        // and update the graph with it
        if (graph == "respeck") {
            dataSet_res_accel_x.addEntry(Entry(time, x))
            dataSet_res_accel_y.addEntry(Entry(time, y))
            dataSet_res_accel_z.addEntry(Entry(time, z))

            runOnUiThread {
                allRespeckData.notifyDataChanged()
                respeckChart.notifyDataSetChanged()
                respeckChart.invalidate()
                respeckChart.setVisibleXRangeMaximum(150f)
                respeckChart.moveViewToX(respeckChart.lowestVisibleX + 40)
            }
        } else if (graph == "thingy") {
            dataSet_thingy_accel_x.addEntry(Entry(time, x))
            dataSet_thingy_accel_y.addEntry(Entry(time, y))
            dataSet_thingy_accel_z.addEntry(Entry(time, z))

            runOnUiThread {
                allThingyData.notifyDataChanged()
                thingyChart.notifyDataSetChanged()
                thingyChart.invalidate()
                thingyChart.setVisibleXRangeMaximum(150f)
                thingyChart.moveViewToX(thingyChart.lowestVisibleX + 40)
            }
        }


    }


    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(respeckLiveUpdateReceiver)
        unregisterReceiver(thingyLiveUpdateReceiver)
        looperRespeck.quit()
        looperThingy.quit()
    }
}
