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
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import android.widget.RadioGroup
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import dev.distraction.demo.ml.core.ModelRegistry
import dev.distraction.demo.ml.debug.DebugModeRepository
import dev.distraction.demo.ml.runtime.InferenceStateStore
import dev.distraction.demo.ml.selection.SharedPrefsModelSelectionRepository
import dev.distraction.demo.ml.ui.ModelOptionUi
import dev.distraction.demo.ml.ui.ModelSelectorDialog
import dev.distraction.demo.service.LoggingService
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var txtSelectedModel: TextView
    private lateinit var txtLastInference: TextView
    private lateinit var btnSelectModel: Button
    private lateinit var btnInspectModel: Button
    private lateinit var cbDebugWindowSummary: CheckBox

    private lateinit var btnToggle: Button
    private lateinit var btnExport: Button

    private lateinit var btnDistract: Button

    private lateinit var modelSelectionRepository: SharedPrefsModelSelectionRepository
    private lateinit var inferenceStateStore: InferenceStateStore
    private lateinit var debugModeRepository: DebugModeRepository
    private var defaultTextColor: android.content.res.ColorStateList? = null

    private val uiHandler = Handler(Looper.getMainLooper())

    private val refreshInferenceRunnable = object : Runnable {
        override fun run() {
            refreshLastInferenceUi()
            updateUi(isLoggingRunning())
            uiHandler.postDelayed(this, 1000L)
        }
    }

    private var pendingStartAfterPermissions = false
    private var suppressInitialPermissionDialog = false

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
        debugModeRepository = DebugModeRepository(this)

        txtSelectedModel = findViewById(R.id.txtSelectedModel)
        txtLastInference = findViewById(R.id.txtLastInference)
        btnSelectModel = findViewById(R.id.btnSelectModel)
        btnInspectModel = findViewById(R.id.btnInspectModel)
        cbDebugWindowSummary = findViewById(R.id.cbDebugWindowSummary)

        btnToggle = findViewById(R.id.btnToggle)
        btnExport = findViewById(R.id.btnExport)

        btnDistract = findViewById(R.id.btnDistract)

        defaultTextColor = txtLastInference.textColors

        ensureDefaultModelSelected()
        refreshSelectedModelUi()
        refreshLastInferenceUi()
        refreshDebugUi()

        btnSelectModel.setOnClickListener {
            openModelSelector()
        }

        btnInspectModel.setOnClickListener {
            inspectSelectedModel()
        }

        cbDebugWindowSummary.setOnCheckedChangeListener { _, checked ->
            debugModeRepository.setDebugWindowSummaryEnabled(checked)
        }

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

        btnDistract.setOnClickListener { sendManualEventWithHaptic(LoggingService.EventType.DISTRACTION) }

        btnExport.setOnClickListener {
            exportModelWeights()
        }

        updateUi(isLoggingRunning())

        if (!hasAllRequiredPermissions() && !isLoggingRunning()) {
            suppressInitialPermissionDialog = true
            showMissingPermissionsDialog(getMissingPermissionsForStart())
        }
    }

    override fun onPause() {
        uiHandler.removeCallbacks(refreshInferenceRunnable)
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        refreshSelectedModelUi()
        refreshLastInferenceUi()
        refreshDebugUi()
        updateUi(isLoggingRunning())

        uiHandler.post(refreshInferenceRunnable)

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
            val defaultId = ModelRegistry.getDefault()?.id
            if (defaultId != null) {
                modelSelectionRepository.setSelectedModelId(defaultId)
            }
        }
    }

    private fun refreshSelectedModelUi() {
        val selectedId = modelSelectionRepository.getSelectedModelId()
        val selectedModel = ModelRegistry.getById(selectedId) ?: ModelRegistry.getDefault()

        if (selectedModel != null) {
            txtSelectedModel.text =
                "Modelo ativo: ${selectedModel.displayName} (${selectedModel.version})"
        } else {
            txtSelectedModel.text = "Modelo ativo: nenhum"
        }
    }

    private fun refreshLastInferenceUi() {
        val error = inferenceStateStore.getLastError()
        if (error != null) {
            txtLastInference.text = "Erro: $error"
            txtLastInference.setTextColor(getColor(android.R.color.holo_red_dark))
            return
        }
        txtLastInference.setTextColor(defaultTextColor)

        val last = inferenceStateStore.getLastInference()
        if (last == null) {
            txtLastInference.text = "Última inferência: sem dados"
            return
        }

        val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val time = sdf.format(Date(last.windowEndMillis))

        txtLastInference.text =
            "Última inferência: score=${String.format(Locale.US, "%.3f", last.score)} | classe=${last.predictedClass} | $time"
    }

    private fun refreshDebugUi() {
        cbDebugWindowSummary.setOnCheckedChangeListener(null)
        cbDebugWindowSummary.isChecked = debugModeRepository.isDebugWindowSummaryEnabled()
        cbDebugWindowSummary.setOnCheckedChangeListener { _, checked ->
            debugModeRepository.setDebugWindowSummaryEnabled(checked)
        }
    }

    private fun openModelSelector() {
        val selectedId = modelSelectionRepository.getSelectedModelId()

        val allModels = ModelRegistry.getAll()
        if (allModels.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Nenhum modelo")
                .setMessage("Nenhum modelo disponível.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val options = allModels.map { model ->
            ModelOptionUi(
                id = model.id,
                title = model.displayName,
                subtitle = "Versão ${model.version}",
                isSelected = model.id == selectedId
            )
        }

        ModelSelectorDialog.show(this, options) { selected ->
            modelSelectionRepository.setSelectedModelId(selected.id)
            refreshSelectedModelUi()
        }
    }

    private fun inspectSelectedModel() {
        val selectedId = modelSelectionRepository.getSelectedModelId()
        val selectedModel = ModelRegistry.getById(selectedId) ?: ModelRegistry.getDefault()
        val metadataPath = selectedModel?.metadataAssetPath

        if (selectedModel == null || metadataPath.isNullOrBlank()) {
            AlertDialog.Builder(this)
                .setTitle("Inspeção do modelo")
                .setMessage("Não foi possível resolver o metadata do modelo selecionado.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        try {
            val text = assets.open(metadataPath).bufferedReader().use { it.readText() }

            Log.d("ModelInspector", text)

            AlertDialog.Builder(this)
                .setTitle("Inspeção do modelo")
                .setMessage(text)
                .setPositiveButton("OK", null)
                .show()
        } catch (e: Exception) {
            Log.e("ModelInspector", "Erro ao inspecionar modelo", e)

            AlertDialog.Builder(this)
                .setTitle("Erro ao inspecionar")
                .setMessage(e.stackTraceToString())
                .setPositiveButton("OK", null)
                .show()
        }
    }

    private fun updateUi(running: Boolean) {
        btnToggle.text = if (running) "Terminar" else "Iniciar"

        btnDistract.isEnabled = running

        btnExport.isEnabled = true
        btnSelectModel.isEnabled = !running
        btnInspectModel.isEnabled = !running
        cbDebugWindowSummary.isEnabled = !running
    }

    private fun ensurePermissionsAndStart() {
        val directMissing = getMissingDirectRequestPermissions()

        if (directMissing.isNotEmpty()) {
            showMissingPermissionsDialog(directMissing)
            return
        }

        if (needsBackgroundLocationPermission()) {
            showBackgroundLocationSettingsDialog()
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
            val directMissing = getMissingDirectRequestPermissions()
            if (directMissing.isNotEmpty()) {
                showMissingPermissionsDialog(directMissing)
            } else if (needsBackgroundLocationPermission()) {
                showBackgroundLocationSettingsDialog()
            }
        }
    }

    private fun getMissingDirectRequestPermissions(): List<String> {
        val required = mutableListOf<String>()

        required += Manifest.permission.ACCESS_FINE_LOCATION

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            required += Manifest.permission.ACTIVITY_RECOGNITION
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            required += Manifest.permission.POST_NOTIFICATIONS
        }

        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            required += Manifest.permission.ACCESS_BACKGROUND_LOCATION
        }

        return required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
    }

    private fun getMissingPermissionsForStart(): List<String> {
        val required = mutableListOf<String>()

        required += Manifest.permission.ACCESS_FINE_LOCATION

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            required += Manifest.permission.ACTIVITY_RECOGNITION
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            required += Manifest.permission.POST_NOTIFICATIONS
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            required += Manifest.permission.ACCESS_BACKGROUND_LOCATION
        }

        return required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
    }

    private fun hasAllRequiredPermissions(): Boolean {
        return getMissingPermissionsForStart().isEmpty()
    }

    private fun needsBackgroundLocationPermission(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_BACKGROUND_LOCATION
                ) != PackageManager.PERMISSION_GRANTED
    }

    private fun showMissingPermissionsDialog(missing: List<String>) {
        if (missing.isEmpty()) return

        val labels = missing.map { permissionToLabel(it) }

        val message = buildString {
            append("Para iniciar a recolha de dados, tens de autorizar:\n\n")
            labels.forEach { label ->
                append("• ")
                append(label)
                append('\n')
            }

            if (missing.contains(Manifest.permission.ACCESS_BACKGROUND_LOCATION) &&
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
            ) {
                append("\nNo Android 11 ou superior, a localização em background é ativada nas definições da app.")
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Permissões necessárias")
            .setMessage(message)
            .setCancelable(false)
            .setPositiveButton("Conceder") { _, _ ->
                val directMissing = getMissingDirectRequestPermissions()
                if (directMissing.isNotEmpty()) {
                    permissionLauncher.launch(directMissing.toTypedArray())
                } else if (needsBackgroundLocationPermission()) {
                    showBackgroundLocationSettingsDialog()
                }
            }
            .setNegativeButton("Cancelar") { _, _ ->
                pendingStartAfterPermissions = false
            }
            .setNeutralButton("Definições") { _, _ ->
                openAppSettings()
            }
            .show()
    }

    private fun showBackgroundLocationSettingsDialog() {
        AlertDialog.Builder(this)
            .setTitle("Localização em background")
            .setMessage(
                "Para permitir a recolha de dados mesmo quando a app não está aberta, " +
                        "tens de ativar \"Permitir sempre\" nas definições de localização da app."
            )
            .setCancelable(false)
            .setPositiveButton("Abrir definições") { _, _ ->
                openAppLocationSettings()
            }
            .setNegativeButton("Cancelar") { _, _ ->
                pendingStartAfterPermissions = false
            }
            .show()
    }

    private fun permissionToLabel(permission: String): String {
        return when (permission) {
            Manifest.permission.ACCESS_FINE_LOCATION -> "Localização precisa"
            Manifest.permission.ACCESS_BACKGROUND_LOCATION -> "Localização em background (Permitir sempre)"
            Manifest.permission.ACTIVITY_RECOGNITION -> "Reconhecimento de atividade"
            Manifest.permission.POST_NOTIFICATIONS -> "Notificações"
            else -> permission
        }
    }

    private fun openAppSettings() {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", packageName, null)
        )
        appSettingsLauncher.launch(intent)
    }

    private fun openAppLocationSettings() {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", packageName, null)
        )
        appSettingsLauncher.launch(intent)
    }

    private fun startServiceLogging() {
        val intent = Intent(this, LoggingService::class.java).apply {
            action = LoggingService.ACTION_START
        }
        ContextCompat.startForegroundService(this, intent)
    }

    private fun stopServiceLogging() {
        val intent = Intent(this, LoggingService::class.java).apply {
            action = LoggingService.ACTION_STOP
        }
        startService(intent)
    }

    private fun sendManualEventWithHaptic(type: String) {
        sendManualEvent(type)
        vibrateOnEvent()
    }

    private fun sendManualEvent(type: String) {
        val intent = Intent(this, LoggingService::class.java).apply {
            action = LoggingService.ACTION_EVENT
            putExtra(LoggingService.EXTRA_EVENT_TYPE, type)
        }
        startService(intent)
    }

    private fun vibrateOnEvent() {
        // Haptics.vibrateTick(this)
    }

    private fun isLoggingRunning(): Boolean {
        val prefs = getSharedPreferences(LoggingService.PREFS, Context.MODE_PRIVATE)
        return prefs.getBoolean(LoggingService.KEY_RUNNING, false)
    }

    private fun formatSessionLabel(dir: File): String {
        val ts = dir.name.removePrefix("session_").toLongOrNull()
        if (ts == null) return dir.name

        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        return sdf.format(Date(ts))
    }

    private fun exportModelWeights() {
        val sessions = listSessionDirs()

        if (sessions.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Exportar sessão")
                .setMessage("Não existem sessões para exportar.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val labels = sessions.map { dir ->
            formatSessionLabel(dir)
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Escolhe a sessão")
            .setItems(labels) { _, which ->
                val dir = sessions[which]
                exportSessionDir(dir)
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun listSessionDirs(): List<File> {
        val baseDir = File(getExternalFilesDir(null), "logs")
        if (!baseDir.exists()) return emptyList()

        val dirs = baseDir.listFiles { f ->
            f.isDirectory && f.name.startsWith("session_")
        }?.toList().orEmpty()

        return dirs.sortedByDescending { it.name }
    }

    private fun exportSessionDir(sessionDir: File) {
        val zipName = "${sessionDir.name}.zip"
        val zipFile = File(cacheDir, zipName)

        if (zipFile.exists()) zipFile.delete()

        try {
            zipDirectory(sessionDir, zipFile)

            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                zipFile
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/zip"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            startActivity(Intent.createChooser(shareIntent, "Partilhar sessão"))
        } catch (e: Exception) {
            AlertDialog.Builder(this)
                .setTitle("Erro ao exportar")
                .setMessage(e.message ?: "Erro desconhecido")
                .setPositiveButton("OK", null)
                .show()
        }
    }

    private fun zipDirectory(sourceDir: File, outZip: File) {
        ZipOutputStream(FileOutputStream(outZip)).use { zos ->
            val basePath = sourceDir.absolutePath.trimEnd(File.separatorChar) + File.separator
            addToZipRecursively(sourceDir, basePath, zos)
        }
    }

    private fun addToZipRecursively(file: File, basePath: String, zos: ZipOutputStream) {
        if (file.isDirectory) {
            file.listFiles()?.forEach { child ->
                addToZipRecursively(child, basePath, zos)
            }
        } else {
            val entryName = file.absolutePath.removePrefix(basePath).replace(File.separatorChar, '/')
            val entry = ZipEntry(entryName)
            zos.putNextEntry(entry)

            BufferedInputStream(FileInputStream(file)).use { input ->
                val buffer = ByteArray(8 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    zos.write(buffer, 0, read)
                }
            }

            zos.closeEntry()
        }
    }
}