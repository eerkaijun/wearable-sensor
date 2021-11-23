package com.specknet.pdiotapp.live

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
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
import kotlin.math.sqrt


class LiveDataActivity : AppCompatActivity() {

    var lastMagnitude = 0.0f
    var stepCount = 0

    var inputValue = Array(1) {
        Array(50) {
            FloatArray(6)
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
        val fileDescriptor = this.assets.openFd("model_cnn.tflite")
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

        if(delta > 0.957)  stepCount++
    }

    fun stepCounterRunning(x: Float, y: Float, z: Float){
        val magnitude = sqrt(x*x + y*y + z*z)
        val delta = lastMagnitude - magnitude
        lastMagnitude = magnitude

        if(delta > 1.2) stepCount++
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
                    if (bufferCount >= 50) {
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
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Sitting/Standing"
                                    imageView.setBackgroundResource(R.drawable.sitting_icon)
                                })
                            }
                            2 -> {
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Lying down"
                                    imageView.setBackgroundResource(R.drawable.lyingdown_icon)
                                })
                            }
                            3 -> {
                                stepCounterWalking(x,y,z)
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Walking"
                                    imageView.setBackgroundResource(R.drawable.walking_icon)
                                    stepCountView.text = stepCount.toString()
                                })
                                Log.i("DEBUG", stepCount.toString())
                            }
                            4 -> {
                                stepCounterRunning(x,y,z)
                                this@LiveDataActivity.runOnUiThread(java.lang.Runnable {
                                    respeckTextView.text = "Running"
                                    imageView.setBackgroundResource(R.drawable.running_icon)
                                    stepCountView.text = stepCount.toString()
                                })
                                Log.i("DEBUG", stepCount.toString())
                            }
                        }


                        // only reset half of the buffer to make a one second sliding window
                        inputValue[0].drop(25)
                        //Log.i("Buffer after resetting", inputValue.contentDeepToString());
                        //Log.i("Length of buffer after resetting", inputValue.size.toString());
                        /*
                        inputValue = Array(1) {
                            Array(50) {
                                FloatArray(6)
                            }
                        }*/
                        bufferCount = 25
                    }
                    inputValue[0][bufferCount][0] = x
                    inputValue[0][bufferCount][1] = y
                    inputValue[0][bufferCount][2] = z
                    inputValue[0][bufferCount][3] = liveData.gyro.x
                    inputValue[0][bufferCount][4] = liveData.gyro.y
                    inputValue[0][bufferCount][5] = liveData.gyro.z

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
