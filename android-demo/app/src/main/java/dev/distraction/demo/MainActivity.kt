package dev.distraction.demo

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import dev.distraction.demo.ml.core.ModelRegistry
import dev.distraction.demo.ml.runtime.InferenceStateStore
import dev.distraction.demo.ml.selection.SharedPrefsModelSelectionRepository
import dev.distraction.demo.service.LoggingService
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var txtSelectedModel: TextView
    private lateinit var txtLastInference: TextView
    private lateinit var progressSection: View
    private lateinit var progressBar: ProgressBar
    private lateinit var btnToggle: Button
    private lateinit var btnHistory: Button

    private lateinit var modelSelectionRepository: SharedPrefsModelSelectionRepository
    private lateinit var inferenceStateStore: InferenceStateStore
    private var defaultTextColor: android.content.res.ColorStateList? = null

    private val uiHandler = Handler(Looper.getMainLooper())

    private val refreshInferenceRunnable = object : Runnable {
        override fun run() {
            refreshLastInferenceUi()
            updateUi(isLoggingRunning())
            uiHandler.postDelayed(this, 1000L)
        }
    }

    private val progressRunnable = object : Runnable {
        var second = false
        override fun run() {
            if (isLoggingRunning()) {
                val windowStart = inferenceStateStore.getWindowStartMillis()
                val windowDuration = inferenceStateStore.getWindowDurationMillis()

                if (windowDuration > 0 && windowStart > 0) {
                    val duration = if (second) windowDuration * 0.5 else windowDuration.toDouble()

                    val actualStart = if (second) windowStart + (windowDuration * 0.5).toLong() else windowStart

                    val elapsed = System.currentTimeMillis() - actualStart
                    val progress = (elapsed.toDouble() / duration * 100).toInt().coerceIn(0, 100)

                    progressBar.progress = progress
                    second = true
                }
            } else {
                progressBar.progress = 0
            }
            uiHandler.postDelayed(this, 200L)
        }
    }

    private var pendingStartAfterPermissions = false

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { _ ->
            handlePermissionFlowAfterRequest()
        }

    private val appSettingsLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            handlePermissionFlowAfterRequest()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ModelRegistry.init(this)

        modelSelectionRepository = SharedPrefsModelSelectionRepository(this)
        inferenceStateStore = InferenceStateStore(this)

        txtSelectedModel = findViewById(R.id.txtSelectedModel)
        txtLastInference = findViewById(R.id.txtLastInference)
        progressSection = findViewById(R.id.progressSection)
        progressBar = findViewById(R.id.progressBar)
        btnToggle = findViewById(R.id.btnToggle)
        btnHistory = findViewById(R.id.btnHistory)

        defaultTextColor = txtLastInference.textColors

        ensureDefaultModelSelected()
        refreshSelectedModelUi()
        refreshLastInferenceUi()

        btnToggle.setOnClickListener {
            val running = isLoggingRunning()
            if (running) {
                pendingStartAfterPermissions = false
                stopServiceLogging()
                updateUi(false)
            } else {
                pendingStartAfterPermissions = true
                ensurePermissionsAndStart()
            }
        }

        btnHistory.setOnClickListener {
            startActivity(Intent(this, InferenceHistoryActivity::class.java))
        }

        updateUi(isLoggingRunning())

        if (!hasAllRequiredPermissions() && !isLoggingRunning()) {
            showMissingPermissionsDialog(getMissingPermissions())
        }
    }

    override fun onPause() {
        uiHandler.removeCallbacks(refreshInferenceRunnable)
        uiHandler.removeCallbacks(progressRunnable)
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        refreshSelectedModelUi()
        refreshLastInferenceUi()
        updateUi(isLoggingRunning())

        uiHandler.post(refreshInferenceRunnable)
        uiHandler.post(progressRunnable)

        if (pendingStartAfterPermissions && !isLoggingRunning()) {
            if (hasAllRequiredPermissions()) {
                startServiceLogging()
                updateUi(true)
                pendingStartAfterPermissions = false
            }
        }
    }

    private fun ensureDefaultModelSelected() {
        val selected = modelSelectionRepository.getSelectedModelId()
        if (selected.isNullOrBlank() || ModelRegistry.getById(selected) == null) {
            ModelRegistry.getDefault()?.id?.let { modelSelectionRepository.setSelectedModelId(it) }
        }
    }

    private fun refreshSelectedModelUi() {
        val selectedId = modelSelectionRepository.getSelectedModelId()
        val model = ModelRegistry.getById(selectedId) ?: ModelRegistry.getDefault()
        txtSelectedModel.text = model?.displayName ?: "No model available"
    }

    private fun refreshLastInferenceUi() {
        val error = inferenceStateStore.getLastError()
        if (error != null) {
            txtLastInference.text = "Error: $error"
            txtLastInference.setTextColor(getColor(android.R.color.holo_red_dark))
            return
        }
        txtLastInference.setTextColor(defaultTextColor)

        val last = inferenceStateStore.getLastInference()
        if (last == null) {
            txtLastInference.text = "Last inference: no data"
            return
        }

        val time = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(last.windowEndMillis))
        txtLastInference.text =
            "Last inference: score=${String.format(Locale.US, "%.3f", last.score)} | class=${last.predictedClass} | $time"
    }

    private fun updateUi(running: Boolean) {
        btnToggle.text = if (running) "Stop" else "Start"
        progressSection.visibility = if (running) View.VISIBLE else View.INVISIBLE
    }

    // --- Permissions ---

    private fun getMissingPermissions(): List<String> {
        val required = mutableListOf(Manifest.permission.ACCESS_FINE_LOCATION)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            required += Manifest.permission.POST_NOTIFICATIONS
        }

        return required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
    }

    private fun hasAllRequiredPermissions(): Boolean = getMissingPermissions().isEmpty()

    private fun ensurePermissionsAndStart() {
        val missing = getMissingPermissions()
        if (missing.isNotEmpty()) {
            showMissingPermissionsDialog(missing)
            return
        }
        startServiceLogging()
        updateUi(true)
        pendingStartAfterPermissions = false
    }

    private fun handlePermissionFlowAfterRequest() {
        if (hasAllRequiredPermissions()) {
            if (pendingStartAfterPermissions && !isLoggingRunning()) {
                startServiceLogging()
                updateUi(true)
            } else {
                updateUi(isLoggingRunning())
            }
            pendingStartAfterPermissions = false
            return
        }

        updateUi(isLoggingRunning())

        if (pendingStartAfterPermissions) {
            val missing = getMissingPermissions()
            if (missing.isNotEmpty()) showMissingPermissionsDialog(missing)
        }
    }

    private fun showMissingPermissionsDialog(missing: List<String>) {
        if (missing.isEmpty()) return

        val message = buildString {
            append("To start data collection, please grant:\n\n")
            missing.forEach { append("• ${permissionToLabel(it)}\n") }
        }

        AlertDialog.Builder(this)
            .setTitle("Required permissions")
            .setMessage(message)
            .setCancelable(false)
            .setPositiveButton("Grant") { _, _ ->
                permissionLauncher.launch(getMissingPermissions().toTypedArray())
            }
            .setNegativeButton("Cancel") { _, _ ->
                pendingStartAfterPermissions = false
            }
            .setNeutralButton("Settings") { _, _ ->
                appSettingsLauncher.launch(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.fromParts("package", packageName, null))
                )
            }
            .show()
    }

    private fun permissionToLabel(permission: String): String = when (permission) {
        Manifest.permission.ACCESS_FINE_LOCATION -> "Precise location"
        Manifest.permission.POST_NOTIFICATIONS -> "Notifications"
        else -> permission
    }

    // --- Service control ---

    private fun startServiceLogging() {
        ContextCompat.startForegroundService(
            this,
            Intent(this, LoggingService::class.java).apply { action = LoggingService.ACTION_START }
        )
    }

    private fun stopServiceLogging() {
        startService(
            Intent(this, LoggingService::class.java).apply { action = LoggingService.ACTION_STOP }
        )
    }

    private fun isLoggingRunning(): Boolean =
        getSharedPreferences(LoggingService.PREFS, Context.MODE_PRIVATE)
            .getBoolean(LoggingService.KEY_RUNNING, false)
}
