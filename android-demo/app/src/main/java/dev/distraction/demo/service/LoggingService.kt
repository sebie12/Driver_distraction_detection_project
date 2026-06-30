package dev.distraction.demo.service

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import dev.distraction.demo.common.TimeUtils
import dev.distraction.demo.device.DeviceStateProvider
import dev.distraction.demo.feedback.AlertFeedbackHelper
import dev.distraction.demo.ml.api.InferenceResult
import dev.distraction.demo.ml.core.InferenceCoordinator
import dev.distraction.demo.ml.core.RawEventMapper
import dev.distraction.demo.ml.runtime.InferenceHistoryStore
import dev.distraction.demo.ml.runtime.InferenceStateStore
import dev.distraction.demo.ml.selection.SharedPrefsModelSelectionRepository
import java.util.concurrent.atomic.AtomicBoolean

class LoggingService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var fused: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback

    private class FilterState(
        var x1: Float = 0.0f,
        var x2: Float = 0.0f,
        var y1: Float = 0.0f,
        var y2: Float = 0.0f,
        var initialized: Boolean = false
    )
    private val filterStates = mutableMapOf<Int, Array<FilterState>>()

    private fun denoiseSample(input: Float, state: FilterState): Float {
        if (!state.initialized) {
            state.x1 = input
            state.x2 = input
            state.y1 = input
            state.y2 = input
            state.initialized = true
        }

        val b0 = 0.2747268510356349f
        val b1 = 0.5494537020712698f
        val b2 = 0.2747268510356349f
        val a1 = -0.07362384638497856f
        val a2 = 0.17253125052751803f

        val output = (b0 * input) + (b1 * state.x1) + (b2 * state.x2) - (a1 * state.y1) - (a2 * state.y2)

        state.x2 = state.x1
        state.x1 = input
        state.y2 = state.y1
        state.y1 = output

        return output
    }

    // Only the three sensors whose data feeds the model (GYRO, LINACC, GAMEROT)
    private var gyro: Sensor? = null
    private var linAcc: Sensor? = null
    private var gameRot: Sensor? = null

    private val running = AtomicBoolean(false)
    private var baseEpochNs: Long = 0L

    private lateinit var deviceStateProvider: DeviceStateProvider
    private lateinit var inferenceCoordinator: InferenceCoordinator
    private lateinit var inferenceStateStore: InferenceStateStore

    private var currentTripId: String? = null
    private var lastPositiveDetectionMillis: Long = 0L
    private var windowCount: Int = 0
    private var isDataValidForInference = true
    private var lastDataInvalidReason: String? = null

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate() called")

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        fused = LocationServices.getFusedLocationProviderClient(this)

        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        linAcc = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gameRot = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

        deviceStateProvider = DeviceStateProvider(this)

        inferenceCoordinator = InferenceCoordinator(
            context = this,
            modelSelectionRepository = SharedPrefsModelSelectionRepository(this)
        )
        inferenceStateStore = InferenceStateStore(this)

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                if (!running.get()) return

                try {
                    val epochNs = baseEpochNs + loc.elapsedRealtimeNanos
                    val epochMs = TimeUtils.toEpochMillis(epochNs)
                    val deviceState = deviceStateProvider.getState()

                    dispatchRawEvent(
                        RawEventMapper.location(
                            timestampMillis = epochMs,
                            latitude = loc.latitude,
                            longitude = loc.longitude,
                            speedMps = if (loc.hasSpeed()) loc.speed else null,
                            bearingDeg = if (loc.hasBearing()) loc.bearing else null,
                            accuracyM = if (loc.hasAccuracy()) loc.accuracy else null,
                            altitudeM = if (loc.hasAltitude()) loc.altitude else null
                        )
                    )

                    dispatchRawEvent(
                        RawEventMapper.deviceState(
                            timestampMillis = epochMs,
                            screenInteractive = deviceState.screenInteractive,
                            deviceLocked = deviceState.deviceLocked,
                            audioActive = deviceState.audioActive
                        )
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Erro em onLocationResult()", e)
                    stopLogging()
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand(action=${intent?.action})")

        when (intent?.action) {
            ACTION_START -> startLogging()
            ACTION_STOP -> stopLogging()
            ACTION_EVENT -> handleManualEvent(intent)
        }
        return START_STICKY
    }

    private fun handleManualEvent(intent: Intent) {
        if (!running.get()) return
        val type = intent.getStringExtra(EXTRA_EVENT_TYPE) ?: return

        try {
            val epochNs = baseEpochNs + SystemClock.elapsedRealtimeNanos()
            dispatchRawEvent(
                RawEventMapper.marker(
                    timestampMillis = TimeUtils.toEpochMillis(epochNs),
                    name = type
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "Erro em handleManualEvent(type=$type)", e)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startLogging() {
        Log.d(TAG, "startLogging() called")

        if (!running.compareAndSet(false, true)) {
            Log.w(TAG, "startLogging() ignored because service is already running")
            return
        }

        try {
            baseEpochNs = TimeUtils.computeBaseEpochNs()
            lastPositiveDetectionMillis = 0L

            val notif = buildNotification("Collecting data…")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
            } else {
                startForeground(NOTIF_ID, notif)
            }

            currentTripId = "session_${System.currentTimeMillis()}"
            filterStates.clear()

            inferenceCoordinator.initialize()
            inferenceStateStore.clearLastError()
            Log.d(TAG, "Inference initialized with model=${inferenceCoordinator.getActiveModelId()}")

            val windowDuration = inferenceCoordinator.getActiveMetadata()?.windowDurationMillis ?: 10_000L
            inferenceStateStore.saveWindowTiming(System.currentTimeMillis(), windowDuration)

            inferenceCoordinator.onTripStarted(currentTripId, System.currentTimeMillis())

            val nowEpochNs = baseEpochNs + SystemClock.elapsedRealtimeNanos()
            val initialDeviceState = deviceStateProvider.getState()

            dispatchRawEvent(
                RawEventMapper.deviceState(
                    timestampMillis = TimeUtils.toEpochMillis(nowEpochNs),
                    screenInteractive = initialDeviceState.screenInteractive,
                    deviceLocked = initialDeviceState.deviceLocked,
                    audioActive = initialDeviceState.audioActive
                )
            )

            val samplingPeriodUs = 60_000
            gyro?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            linAcc?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            gameRot?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }

            val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
                .setMinUpdateIntervalMillis(500L)
                .setWaitForAccurateLocation(false)
                .build()

            try {
                fused.requestLocationUpdates(req, locationCallback, mainLooper)
                Log.d(TAG, "requestLocationUpdates() registered")
            } catch (e: Exception) {
                Log.e(TAG, "Erro ao pedir updates de localização", e)
                stopLogging()
                return
            }

            setRunning(true)
            Log.d(TAG, "startLogging() completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Erro fatal em startLogging()", e)
            inferenceStateStore.saveLastError(e.message ?: e.toString())
            stopLogging()
        }
    }

    private fun stopLogging() {
        Log.w(TAG, "stopLogging() called")

        if (!running.compareAndSet(true, false)) {
            setRunning(false)
            stopSelf()
            return
        }

        try {
            inferenceCoordinator.onTripEnded(currentTripId, System.currentTimeMillis())
        } catch (e: Exception) {
            Log.e(TAG, "Erro em onTripEnded()", e)
        }

        try {
            inferenceCoordinator.reset()
        } catch (e: Exception) {
            Log.e(TAG, "Erro em inferenceCoordinator.reset()", e)
        }

        try {
            sensorManager.unregisterListener(this)
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao unregisterListener()", e)
        }

        try {
            fused.removeLocationUpdates(locationCallback)
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao removeLocationUpdates()", e)
        }

        currentTripId = null
        setRunning(false)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        Log.w(TAG, "onDestroy() called")
        setRunning(false)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        val epochNs = baseEpochNs + event.timestamp

        try {
            val denoisedValues = FloatArray(event.values.size)
            val states = filterStates.getOrPut(event.sensor.type) {
                Array(event.values.size) { FilterState() }
            }

            for (i in event.values.indices) {
                denoisedValues[i] = denoiseSample(event.values[i], states[i])
            }

            dispatchRawEvent(
                RawEventMapper.sensor(
                    timestampMillis = TimeUtils.toEpochMillis(epochNs),
                    sensorType = sensorTypeToName(event.sensor.type),
                    values = denoisedValues
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "Erro em onSensorChanged(sensorType=${event.sensor.type})", e)
            stopLogging()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun checkDataValidity(event: dev.distraction.demo.ml.types.RawEvent) {
        if (event is dev.distraction.demo.ml.types.RawEvent.DeviceStateEvent) {
            val deviceMissing = event.deviceLocked == null || event.audioActive == null
            if (deviceMissing) {
                if (isDataValidForInference || lastDataInvalidReason != "Device features not correct") {
                    isDataValidForInference = false
                    lastDataInvalidReason = "Device features not correct"
                    Handler(Looper.getMainLooper()).post {
                        Toast.makeText(this, "Device features missing or incorrect, inference paused.", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                if (!isDataValidForInference && lastDataInvalidReason == "Device features not correct") {
                    isDataValidForInference = true
                    lastDataInvalidReason = null
                    Handler(Looper.getMainLooper()).post {
                        Toast.makeText(this, "Device features restored, inference resumed.", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun dispatchRawEvent(event: dev.distraction.demo.ml.types.RawEvent) {
        checkDataValidity(event)
        if (!isDataValidForInference) return

        try {
            var result = inferenceCoordinator.onRawEvent(event) ?: return

            if (!deviceStateProvider.getState().screenInteractive) {
                result = result.copy(score = 0f, predictedClass = 0)
            }

            handleInferenceResult(result)
        } catch (e: Exception) {
            Log.e(TAG, "Erro em dispatchRawEvent(event=${event.javaClass.simpleName})", e)
            inferenceStateStore.saveLastError(e.message ?: e.toString())
            stopLogging()
        }
    }

    private fun handleInferenceResult(result: InferenceResult) {
        windowCount++

        Log.d(TAG, "Inference result [#$windowCount] model=${result.modelId} score=${result.score} class=${result.predictedClass}")

        inferenceStateStore.saveLastInference(
            modelId = result.modelId,
            score = result.score,
            predictedClass = result.predictedClass,
            windowEndMillis = result.windowEndMillis
        )

        val newWindowStart = (result.windowStartMillis + result.windowEndMillis) / 2
        inferenceStateStore.saveWindowTiming(newWindowStart, result.windowEndMillis - result.windowStartMillis)

        InferenceHistoryStore.add(
            InferenceHistoryStore.HistoryItem(
                timestampMillis = result.windowEndMillis,
                score = result.score,
                predictedClass = result.predictedClass,
                modelId = result.modelId
            )
        )

        if (result.predictedClass == 1) {
            maybeWriteAutomaticDetectionEvent(result)
        }
    }

    private fun maybeWriteAutomaticDetectionEvent(result: InferenceResult) {
        val nowMs = result.windowEndMillis
        if (nowMs - lastPositiveDetectionMillis < DETECTION_COOLDOWN_MS) return
        lastPositiveDetectionMillis = nowMs
        AlertFeedbackHelper.notifyDistractionDetected(this)
    }

    private fun sensorTypeToName(sensorType: Int): String = when (sensorType) {
        Sensor.TYPE_GYROSCOPE -> "GYRO"
        Sensor.TYPE_LINEAR_ACCELERATION -> "LINACC"
        Sensor.TYPE_GAME_ROTATION_VECTOR -> "GAMEROT"
        else -> "SENSOR_$sensorType"
    }

    private fun setRunning(running: Boolean) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_RUNNING, running).apply()
    }

    private fun buildNotification(text: String): Notification {
        val channelId = "logging_channel"
        val mgr = getSystemService(NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            mgr.createNotificationChannel(
                NotificationChannel(channelId, "Logging", NotificationManager.IMPORTANCE_LOW)
            )
        }

        val stopPending = PendingIntent.getService(
            this, 1,
            Intent(this, LoggingService::class.java).apply { action = ACTION_STOP },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Distraction Detection")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPending)
            .setOngoing(true)
            .build()
    }

    object EventType {
        const val DISTRACTION = "DISTRACTION"
        const val USER_CONFIRMED_DISTRACTION = "USER_CONFIRMED_DISTRACTION"
        const val MODEL_DISTRACTION_DETECTED = "MODEL_DISTRACTION_DETECTED"
    }

    companion object {
        private const val TAG = "LoggingService"

        const val NOTIF_ID = 1001

        const val ACTION_START = "dev.distraction.demo.START"
        const val ACTION_STOP = "dev.distraction.demo.STOP"
        const val ACTION_EVENT = "dev.distraction.demo.EVENT"
        const val ACTION_FEEDBACK = "dev.distraction.demo.ACTION_FEEDBACK"
        const val ACTION_DISMISS_FEEDBACK = "dev.distraction.demo.ACTION_DISMISS_FEEDBACK"
        const val ACTION_SAVE_MODEL = "dev.distraction.demo.ACTION_SAVE_MODEL"
        const val EXTRA_WAS_DISTRACTED = "dev.distraction.demo.EXTRA_WAS_DISTRACTED"

        const val EXTRA_EVENT_TYPE = "event_type"
        const val PREFS = "logging_service_prefs"
        const val KEY_RUNNING = "running"

        private const val DETECTION_COOLDOWN_MS = 10_000L
    }
}
