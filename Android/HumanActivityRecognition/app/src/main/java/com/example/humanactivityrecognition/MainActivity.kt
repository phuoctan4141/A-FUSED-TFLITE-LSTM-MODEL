package com.example.humanactivityrecognition

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.recyclerview.widget.RecyclerView
import com.example.humanactivityrecognition.ml.ModelLstmMetadata
import com.example.humanactivityrecognition.ui.RecognitionAdapter
import com.example.humanactivityrecognition.viewmodel.Recognition
import com.example.humanactivityrecognition.viewmodel.RecognitionListViewModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var sensorAccelerometer: Sensor
    private lateinit var sensorGyroscope: Sensor
    private lateinit var sensorLinearAcceleration: Sensor

    private lateinit var results: MutableList<Category>

    // Views attachment
    private val resultRecyclerView by lazy {
        findViewById<RecyclerView>(R.id.recognitionResults) // Display the result of analysis
    }

    // Contains the recognition result. Since  it is a viewModel, it will survive screen rotations
    private val recogViewModel: RecognitionListViewModel by viewModels()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        sensorAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        sensorManager.registerListener(this, sensorAccelerometer, SensorManager.SENSOR_DELAY_FASTEST)

        sensorGyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        sensorManager.registerListener(this, sensorGyroscope, SensorManager.SENSOR_DELAY_FASTEST)

        sensorLinearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        sensorManager.registerListener(this, sensorLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST)

        ax = ArrayList(); ay = ArrayList(); az = ArrayList()
        lx = ArrayList(); ly = ArrayList(); lz = ArrayList()
        gx = ArrayList(); gy = ArrayList(); gz = ArrayList()
        ma = ArrayList(); ml = ArrayList(); mg = ArrayList()

        // Initialising the resultRecyclerView and its linked viewAdaptor
        val viewAdapter = RecognitionAdapter(this)
        resultRecyclerView.adapter = viewAdapter

        // Disable recycler view animation to reduce flickering, otherwise items can move, fade in
        // and out as the list change
        resultRecyclerView.itemAnimator = null

        // Attach an observer on the LiveData field of recognitionList
        // This will notify the recycler view to update every time when a new list is set on the
        // LiveData field of recognitionList.
        recogViewModel.recognitionList.observe(this, viewAdapter::submitList)

    }


    override fun onSensorChanged(p0: SensorEvent?) {
        p0?.let { getSensor(it) }
    }

    private fun getSensor(it: SensorEvent) {
        val event = it.sensor

        when (event.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                ax!!.add(it.values[0])
                ay!!.add(it.values[1])
                az!!.add(it.values[2])
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                lx!!.add(it.values[0])
                ly!!.add(it.values[1])
                lz!!.add(it.values[2])
            }
            Sensor.TYPE_GYROSCOPE -> {
                gx!!.add(it.values[0])
                gy!!.add(it.values[1])
                gz!!.add(it.values[2])
            }
        }

        predictActivity()
    }

    override fun onAccuracyChanged(p0: Sensor?, p1: Int) {}

    private fun predictActivity() {
        val data: MutableList<Float> = java.util.ArrayList()

        if (ax!!.size >= TIME_STAMP && ay!!.size >= TIME_STAMP && az!!.size >= TIME_STAMP
            && lx!!.size >= TIME_STAMP && ly!!.size >= TIME_STAMP && lz!!.size >= TIME_STAMP
            && gx!!.size >= TIME_STAMP && gy!!.size >= TIME_STAMP && gz!!.size >= TIME_STAMP
        ) {
            var maValue: Double
            var mgValue: Double
            var mlValue: Double

            for (i in 0 until TIME_STAMP) {
                maValue = sqrt(
                    ax!![i].toDouble().pow(2.0) + ay!![i].toDouble().pow(2.0) + az!![i].toDouble()
                        .pow(2.0)
                )
                mlValue = sqrt(
                    lx!![i].toDouble().pow(2.0) + ly!![i].toDouble().pow(2.0) + lz!![i].toDouble()
                        .pow(2.0)
                )
                mgValue = sqrt(
                    gx!![i].toDouble().pow(2.0) + gy!![i].toDouble().pow(2.0) + gz!![i].toDouble()
                        .pow(2.0)
                )
                ma!!.add(maValue.toFloat())
                ml!!.add(mlValue.toFloat())
                mg!!.add(mgValue.toFloat())
            }

            data.addAll(ax!!.subList(0, TIME_STAMP))
            data.addAll(ay!!.subList(0, TIME_STAMP))
            data.addAll(az!!.subList(0, TIME_STAMP))

            data.addAll(lx!!.subList(0, TIME_STAMP))
            data.addAll(ly!!.subList(0, TIME_STAMP))
            data.addAll(lz!!.subList(0, TIME_STAMP))

            data.addAll(gx!!.subList(0, TIME_STAMP))
            data.addAll(gy!!.subList(0, TIME_STAMP))
            data.addAll(gz!!.subList(0, TIME_STAMP))

            data.addAll(ma!!.subList(0, TIME_STAMP))
            data.addAll(ml!!.subList(0, TIME_STAMP))
            data.addAll(mg!!.subList(0, TIME_STAMP))

            results = predictProbabilities(toFloatArray(data))

            this.analyze(this.results)

            Log.d("results", results.toString())

            data.clear()
            ax!!.clear(); ay!!.clear(); az!!.clear()
            lx!!.clear(); ly!!.clear(); lz!!.clear()
            gx!!.clear(); gy!!.clear(); gz!!.clear()
            ma!!.clear(); ml!!.clear(); mg!!.clear()
        }
    }

    private fun analyze(data: MutableList<Category>) {
        val items = mutableListOf<Recognition>()

        // Converting the top probability items into a list of recognitions and update them
        with(recogViewModel) {
            for (output in data) {
                items.add(Recognition(output.label, output.score))
            }

            updateData(items)
        }
    }

    private fun predictProbabilities(data: FloatArray): MutableList<Category> {
        val model = ModelLstmMetadata.newInstance(this)

        // Creates inputs for reference.
        val inputSensor = TensorBuffer.createFixedSize(intArrayOf(1, 100, 12), DataType.FLOAT32)
        inputSensor.loadArray(data)

        // Runs model inference and gets result.
        val outputs = model.process(inputSensor)

        // Releases model resources if no longer used.
        // model.close()

        return outputs.probabilityAsCategoryList
    }

    private fun toFloatArray(data: List<Float>): FloatArray {
        var i = 0
        val array = FloatArray(data.size)
        for (f in data) {
            array[i++] = f
        }
        return array
    }

    override fun onResume() {
        super.onResume()
        sensorManager.registerListener(this,
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            SensorManager.SENSOR_DELAY_FASTEST)

        sensorManager.registerListener(this,
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
            SensorManager.SENSOR_DELAY_FASTEST)

        sensorManager.registerListener(this,
            sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
            SensorManager.SENSOR_DELAY_FASTEST)
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    companion object {
        private const val TIME_STAMP = 100

        private var ax: MutableList<Float>? = null
        private var ay: MutableList<Float>? = null
        private var az: MutableList<Float>? = null

        private var gx: MutableList<Float>? = null
        private var gy: MutableList<Float>? = null
        private var gz: MutableList<Float>? = null

        private var lx: MutableList<Float>? = null
        private var ly: MutableList<Float>? = null
        private var lz: MutableList<Float>? = null

        private var ma: MutableList<Float>? = null
        private var ml: MutableList<Float>? = null
        private var mg: MutableList<Float>? = null
    }

//end class
}