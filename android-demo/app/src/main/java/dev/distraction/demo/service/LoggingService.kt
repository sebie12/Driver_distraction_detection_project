package dev.distraction.demo.service

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import dev.distraction.demo.activity.ActivityCsvWriter
import dev.distraction.demo.activity.ActivityRecognitionHandler
import dev.distraction.demo.common.TimeUtils
import dev.distraction.demo.device.DeviceState
import dev.distraction.demo.device.DeviceStateCsvWriter
import dev.distraction.demo.device.DeviceStateProvider
import dev.distraction.demo.events.EventCsvWriter
import dev.distraction.demo.location.LocationCsvWriter
import dev.distraction.demo.ml.api.InferenceResult
import dev.distraction.demo.ml.core.InferenceCoordinator
import dev.distraction.demo.ml.core.RawEventMapper
import dev.distraction.demo.ml.debug.DebugModeRepository
import dev.distraction.demo.ml.selection.ModelCategoryRepository
import dev.distraction.demo.ml.api.ModelCategory
import dev.distraction.demo.ml.debug.DebugWindowSummaryCsvWriter
import dev.distraction.demo.ml.logging.InferenceCsvWriter
import dev.distraction.demo.ml.runtime.InferenceStateStore
import dev.distraction.demo.ml.selection.SharedPrefsModelSelectionRepository
import dev.distraction.demo.ml.debug.ModelInputDumpManager
import dev.distraction.demo.sensors.SensorCsvWriter
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.reflect.KMutableProperty0
import dev.distraction.demo.feedback.AlertFeedbackHelper

class LoggingService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var fused: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback

    private lateinit var debugModeRepository: DebugModeRepository
    private lateinit var categoryRepository: ModelCategoryRepository
    private lateinit var debugWindowSummaryCsvWriter: DebugWindowSummaryCsvWriter
    private var debugWindowSummaryEnabled: Boolean = false
    private var denoisedModelsEnabled: Boolean = false

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

    private var accel: Sensor? = null
    private var gyro: Sensor? = null
    private var rot: Sensor? = null
    private var linAcc: Sensor? = null
    private var grav: Sensor? = null
    private var mag: Sensor? = null
    private var gameRot: Sensor? = null

    private var sensorsWriter: BufferedWriter? = null
    private var locationWriter: BufferedWriter? = null
    private var eventsWriter: BufferedWriter? = null
    private var activityWriter: BufferedWriter? = null
    private var deviceStateWriter: BufferedWriter? = null

    private var sessionDir: File? = null
    private val running = AtomicBoolean(false)
    private var baseEpochNs: Long = 0L

    private lateinit var sensorCsvWriter: SensorCsvWriter
    private lateinit var eventCsvWriter: EventCsvWriter
    private lateinit var locationCsvWriter: LocationCsvWriter
    private lateinit var activityCsvWriter: ActivityCsvWriter
    private lateinit var activityHandler: ActivityRecognitionHandler
    private lateinit var deviceStateCsvWriter: DeviceStateCsvWriter
    private lateinit var deviceStateProvider: DeviceStateProvider

    private lateinit var inferenceCoordinator: InferenceCoordinator
    private lateinit var inferenceCsvWriter: InferenceCsvWriter
    private lateinit var inferenceStateStore: InferenceStateStore

    private var currentTripId: String? = null

    private var lastDeviceStateHash: String? = null
    private var lastDeviceStateWriteElapsedNs: Long = 0L

    private var lastPositiveDetectionMillis: Long = 0L
    private var pendingTrainingInputs: dev.distraction.demo.ml.api.ModelInputs? = null
    private var windowCount: Int = 0
    private var isDataValidForInference = true
    private var lastDataInvalidReason: String? = null

    private val activityReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent == null || activityWriter == null) return

            val nowElapsedNs = SystemClock.elapsedRealtimeNanos()
            val epochS = TimeUtils.formatEpochSeconds(baseEpochNs + nowElapsedNs)

            try {
                activityCsvWriter.writeInfo(
                    epochS,
                    "RECEIVED_${intent.action ?: "NO_ACTION"}",
                    nowElapsedNs
                )

                when (intent.action) {
                    ActivityRecognitionHandler.ACTION_ACTIVITY_UPDATE -> {
                        activityHandler.handleActivityUpdateIntent(intent)

                        val activityType = intent.getStringExtra("activity_type")
                        val confidence = if (intent.hasExtra("confidence")) {
                            intent.getIntExtra("confidence", -1).takeIf { it >= 0 }
                        } else {
                            null
                        }

                        if (activityType != null) {
                            dispatchRawEvent(
                                RawEventMapper.activity(
                                    timestampMillis = TimeUtils.toEpochMillis(baseEpochNs + nowElapsedNs),
                                    activityType = activityType,
                                    confidence = confidence
                                )
                            )
                        }
                    }

                    ActivityRecognitionHandler.ACTION_ACTIVITY_TRANSITION -> {
                        activityHandler.handleActivityTransitionIntent(intent)

                        val transitionType = intent.getStringExtra("transition_type")
                        val activityType = intent.getStringExtra("activity_type")
                        val name = buildString {
                            append("ACTIVITY_TRANSITION")
                            if (!activityType.isNullOrBlank()) {
                                append("_")
                                append(activityType)
                            }
                            if (!transitionType.isNullOrBlank()) {
                                append("_")
                                append(transitionType)
                            }
                        }

                        dispatchRawEvent(
                            RawEventMapper.marker(
                                timestampMillis = TimeUtils.toEpochMillis(baseEpochNs + nowElapsedNs),
                                name = name
                            )
                        )
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Erro no activityReceiver(action=${intent.action})", e)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate() called")

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        fused = LocationServices.getFusedLocationProviderClient(this)

        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rot = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        linAcc = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        grav = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        mag = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        gameRot = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

        sensorCsvWriter = SensorCsvWriter { sensorsWriter }
        eventCsvWriter = EventCsvWriter { eventsWriter }
        locationCsvWriter = LocationCsvWriter { locationWriter }
        activityCsvWriter = ActivityCsvWriter { activityWriter }
        activityHandler = ActivityRecognitionHandler(this, activityCsvWriter) { baseEpochNs }

        deviceStateCsvWriter = DeviceStateCsvWriter { deviceStateWriter }
        deviceStateProvider = DeviceStateProvider(this)

        debugModeRepository = DebugModeRepository(this)
        categoryRepository = ModelCategoryRepository(this)
        debugWindowSummaryCsvWriter = DebugWindowSummaryCsvWriter()

        inferenceCoordinator = InferenceCoordinator(
            context = this,
            modelSelectionRepository = SharedPrefsModelSelectionRepository(this),
            categoryRepository = categoryRepository
        )
        inferenceCsvWriter = InferenceCsvWriter(this)
        inferenceStateStore = InferenceStateStore(this)

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                if (locationWriter == null || deviceStateWriter == null) return

                try {
                    val epochNs = baseEpochNs + loc.elapsedRealtimeNanos
                    val epochS = TimeUtils.formatEpochSeconds(epochNs)
                    val epochMs = TimeUtils.toEpochMillis(epochNs)
                    val deviceState = deviceStateProvider.getState()

                    locationCsvWriter.writeLocation(epochS, loc)
                    maybeWriteDeviceState(epochS, deviceState)

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
                            audioActive = deviceState.audioActive,
                            audioOutput = deviceState.audioOutput,
                            batteryPct = deviceState.batteryPct,
                            charging = deviceState.isCharging
                        )
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Erro em onLocationResult()", e)
                    stopLogging()
                }
            }
        }

        val filter = IntentFilter().apply {
            addAction(ActivityRecognitionHandler.ACTION_ACTIVITY_UPDATE)
            addAction(ActivityRecognitionHandler.ACTION_ACTIVITY_TRANSITION)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(activityReceiver, filter, RECEIVER_EXPORTED)
        } else {
            registerReceiver(activityReceiver, filter)
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
            val nowElapsedNs = SystemClock.elapsedRealtimeNanos()
            val epochNs = baseEpochNs + nowElapsedNs
            val epochS = TimeUtils.formatEpochSeconds(epochNs)

            eventCsvWriter.writeEvent(epochS, type)

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
            lastDeviceStateHash = null
            lastDeviceStateWriteElapsedNs = 0L
            lastPositiveDetectionMillis = 0L

            val notif = buildNotification("A recolher dados…")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
            } else {
                startForeground(NOTIF_ID, notif)
            }

            val baseDir = File(getExternalFilesDir(null), "logs")
            baseDir.mkdirs()

            sessionDir = File(baseDir, "session_${System.currentTimeMillis()}").apply { mkdirs() }
            currentTripId = sessionDir?.name

            sensorsWriter = BufferedWriter(FileWriter(File(sessionDir!!, "sensors.csv"), true))
            locationWriter = BufferedWriter(FileWriter(File(sessionDir!!, "location.csv"), true))
            eventsWriter = BufferedWriter(FileWriter(File(sessionDir!!, "events.csv"), true))
            activityWriter = BufferedWriter(FileWriter(File(sessionDir!!, "activity.csv"), true))
            deviceStateWriter = BufferedWriter(FileWriter(File(sessionDir!!, "device_state.csv"), true))

            sensorCsvWriter.writeHeader()
            locationCsvWriter.writeHeader()
            eventCsvWriter.writeHeader()
            activityCsvWriter.writeHeader()
            deviceStateCsvWriter.writeHeader()

            inferenceCsvWriter.open(sessionDir!!)

            debugWindowSummaryEnabled = debugModeRepository.isDebugWindowSummaryEnabled()
            if (debugWindowSummaryEnabled) {
                debugWindowSummaryCsvWriter.open(sessionDir!!)
            }

            denoisedModelsEnabled = categoryRepository.getSelectedCategory() == ModelCategory.DENOISED
            filterStates.clear()

            inferenceCoordinator.initialize()
            inferenceStateStore.clearLastError()
            Log.d(TAG, "Inference initialized with model=${inferenceCoordinator.getActiveModelId()}")

            ModelInputDumpManager.configure(
                enabled = debugWindowSummaryEnabled,
                sessionFolder = sessionDir,
                modelId = inferenceCoordinator.getActiveModelId()
            )

            inferenceCoordinator.onTripStarted(currentTripId, System.currentTimeMillis())



            flushAll()

            val nowElapsedNs = SystemClock.elapsedRealtimeNanos()
            val nowEpochNs = baseEpochNs + nowElapsedNs
            val nowEpoch = TimeUtils.formatEpochSeconds(nowEpochNs)
            val initialDeviceState = deviceStateProvider.getState()

            deviceStateCsvWriter.writeState(nowEpoch, initialDeviceState)
            lastDeviceStateHash = buildDeviceStateKey(initialDeviceState)
            lastDeviceStateWriteElapsedNs = nowElapsedNs

            dispatchRawEvent(
                RawEventMapper.deviceState(
                    timestampMillis = TimeUtils.toEpochMillis(nowEpochNs),
                    screenInteractive = initialDeviceState.screenInteractive,
                    deviceLocked = initialDeviceState.deviceLocked,
                    audioActive = initialDeviceState.audioActive,
                    audioOutput = initialDeviceState.audioOutput,
                    batteryPct = initialDeviceState.batteryPct,
                    charging = initialDeviceState.isCharging
                )
            )

            val samplingPeriodUs = 60_000
            accel?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            gyro?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            rot?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            linAcc?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            grav?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
            mag?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
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
                val errEpoch = TimeUtils.formatEpochSeconds(
                    baseEpochNs + SystemClock.elapsedRealtimeNanos()
                )
                locationCsvWriter.writeError(errEpoch)
                stopLogging()
                return
            }

            activityHandler.start()
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

        try {
            activityHandler.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao parar activityHandler", e)
        }

        closeWriter(::sensorsWriter)
        closeWriter(::locationWriter)
        closeWriter(::eventsWriter)
        closeWriter(::activityWriter)
        closeWriter(::deviceStateWriter)

        try {
            debugWindowSummaryCsvWriter.close()
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao fechar debugWindowSummaryCsvWriter", e)
        }

        try {
            inferenceCsvWriter.close()
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao fechar inferenceCsvWriter", e)
        }

        ModelInputDumpManager.clear()

        sessionDir = null
        currentTripId = null
        lastDeviceStateHash = null
        lastDeviceStateWriteElapsedNs = 0L

        setRunning(false)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        Log.w(TAG, "onDestroy() called")

        setRunning(false)
        try {
            unregisterReceiver(activityReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao unregisterReceiver()", e)
        }

        try {
            inferenceCsvWriter.close()
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao fechar inferenceCsvWriter em onDestroy()", e)
        }

        try {
            debugWindowSummaryCsvWriter.close()
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao fechar debugWindowSummaryCsvWriter em onDestroy()", e)
        }

        ModelInputDumpManager.clear()

        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        val epochNs = baseEpochNs + event.timestamp
        val epochS = TimeUtils.formatEpochSeconds(epochNs)

        try {
            // Write raw data to CSV for logging, but denoise for inference pipeline
            sensorCsvWriter.writeSensor(epochS, event)

            // Denoise the values before sending to the inference pipeline
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

    private fun maybeWriteDeviceState(epochS: String, deviceState: DeviceState) {
        val nowElapsedNs = SystemClock.elapsedRealtimeNanos()
        val stateKey = buildDeviceStateKey(deviceState)

        val shouldWrite =
            stateKey != lastDeviceStateHash ||
                    nowElapsedNs - lastDeviceStateWriteElapsedNs >= DEVICE_STATE_WRITE_INTERVAL_NS

        if (shouldWrite) {
            deviceStateCsvWriter.writeState(epochS, deviceState)
            lastDeviceStateHash = stateKey
            lastDeviceStateWriteElapsedNs = nowElapsedNs
        }
    }

    private fun buildDeviceStateKey(state: DeviceState): String {
        return listOf(
            state.batteryPct,
            state.isCharging,
            state.plugType,
            state.screenInteractive,
            state.deviceLocked,
            state.powerSaveMode,
            state.networkType,
            state.networkMetered,
            state.restrictBackgroundStatus,
            state.audioActive,
            state.audioOutput
        ).joinToString("|")
    }

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
        if (!isDataValidForInference) {
            return
        }

        try {
            var result = inferenceCoordinator.onRawEvent(event) ?: return
            
            if (!deviceStateProvider.getState().screenInteractive) {
                result = result.copy(
                    score = 0f,
                    predictedClass = 0
                )
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

        Log.d(
            TAG,
            "Inference result [#$windowCount] model=${result.modelId} score=${result.score} class=${result.predictedClass}"
        )

        inferenceCsvWriter.write(
            windowStartMillis = result.windowStartMillis,
            windowEndMillis = result.windowEndMillis,
            modelId = result.modelId,
            modelVersion = result.modelVersion,
            score = result.score,
            predictedClass = result.predictedClass,
            threshold = result.threshold,
            sensorSamplesCount = result.sensorSamplesCount,
            locationSamplesCount = result.locationSamplesCount,
            deviceSamplesCount = result.deviceSamplesCount,
            extras = result.extras
        )

        inferenceStateStore.saveLastInference(
            modelId = result.modelId,
            score = result.score,
            predictedClass = result.predictedClass,
            windowEndMillis = result.windowEndMillis
        )

        val autoEventEmitted = if (result.predictedClass == 1) {
            maybeWriteAutomaticDetectionEvent(result)
        } else {
            false
        }

        if (debugWindowSummaryEnabled) {
            debugWindowSummaryCsvWriter.write(
                windowStartMillis = result.windowStartMillis,
                windowEndMillis = result.windowEndMillis,
                modelId = result.modelId,
                modelVersion = result.modelVersion,
                score = result.score,
                predictedClass = result.predictedClass,
                threshold = result.threshold,
                sensorSamplesCount = result.sensorSamplesCount,
                locationSamplesCount = result.locationSamplesCount,
                deviceSamplesCount = result.deviceSamplesCount,
                autoEventEmitted = autoEventEmitted
            )
        }
    }

    private fun maybeWriteAutomaticDetectionEvent(result: InferenceResult): Boolean {
        val nowMs = result.windowEndMillis
        if (nowMs - lastPositiveDetectionMillis < DETECTION_COOLDOWN_MS) {
            return false
        }

        lastPositiveDetectionMillis = nowMs

        val epochS = String.format(Locale.US, "%.3f", nowMs / 1000.0)
        eventCsvWriter.writeEvent(epochS, EventType.MODEL_DISTRACTION_DETECTED)

        AlertFeedbackHelper.notifyDistractionDetected(this)

        return true
    }

    private fun sensorTypeToName(sensorType: Int): String {
        return when (sensorType) {
            Sensor.TYPE_ACCELEROMETER -> "ACC"
            Sensor.TYPE_GYROSCOPE -> "GYRO"
            Sensor.TYPE_ROTATION_VECTOR -> "ROT"
            Sensor.TYPE_LINEAR_ACCELERATION -> "LINACC"
            Sensor.TYPE_GRAVITY -> "GRAV"
            Sensor.TYPE_MAGNETIC_FIELD -> "MAG"
            Sensor.TYPE_GAME_ROTATION_VECTOR -> "GAMEROT"
            else -> "SENSOR_$sensorType"
        }
    }

    private fun flushAll() {
        try {
            sensorsWriter?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush sensorsWriter", e)
        }
        try {
            locationWriter?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush locationWriter", e)
        }
        try {
            eventsWriter?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush eventsWriter", e)
        }
        try {
            activityWriter?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush activityWriter", e)
        }
        try {
            deviceStateWriter?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush deviceStateWriter", e)
        }
    }

    private fun closeWriter(ref: KMutableProperty0<BufferedWriter?>) {
        try {
            ref.get()?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a flush writer ${ref.name}", e)
        }
        try {
            ref.get()?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Erro a close writer ${ref.name}", e)
        }
        ref.set(null)
    }

    private fun setRunning(running: Boolean) {
        getSharedPreferences(PREFS, MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_RUNNING, running)
            .apply()
    }

    private fun buildNotification(text: String): Notification {
        val channelId = "logging_channel"
        val mgr = getSystemService(NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(channelId, "Logging", NotificationManager.IMPORTANCE_LOW)
            mgr.createNotificationChannel(ch)
        }

        val stopIntent = Intent(this, LoggingService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPending = PendingIntent.getService(
            this,
            1,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Sensor dump")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .addAction(android.R.drawable.ic_media_pause, "Parar", stopPending)
            .setOngoing(true)
            .build()
    }

    object EventType {
        const val HARD_BRAKE = "HARD_BRAKE"
        const val HARD_ACCEL = "HARD_ACCEL"
        const val HARD_TURN = "HARD_TURN"
        const val DISTRACTION = "DISTRACTION"
        const val USER_CONFIRMED_DISTRACTION = "USER_CONFIRMED_DISTRACTION"
        const val MODEL_DISTRACTION_DETECTED = "MODEL_DISTRACTION_DETECTED"
    }

    companion object {

        private const val TAG = "LoggingService"

        const val NOTIF_ID = 1001

        const val ACTION_FEEDBACK = "dev.distraction.demo.ACTION_FEEDBACK"
        const val ACTION_DISMISS_FEEDBACK = "dev.distraction.demo.ACTION_DISMISS_FEEDBACK"
        const val ACTION_SAVE_MODEL = "dev.distraction.demo.ACTION_SAVE_MODEL"
        const val EXTRA_WAS_DISTRACTED = "dev.distraction.demo.EXTRA_WAS_DISTRACTED"
        const val ACTION_START = "dev.distraction.demo.START"
        const val ACTION_STOP = "dev.distraction.demo.STOP"
        const val ACTION_EVENT = "dev.distraction.demo.EVENT"

        const val EXTRA_EVENT_TYPE = "event_type"
        const val PREFS = "logging_service_prefs"
        const val KEY_RUNNING = "running"

        private const val DEVICE_STATE_WRITE_INTERVAL_NS = 10_000_000_000L
        private const val DETECTION_COOLDOWN_MS = 10_000L
    }
}