package dev.distraction.demo.device

import android.app.KeyguardManager
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

data class DeviceState(
    val batteryPct: Int?,
    val isCharging: Boolean,
    val plugType: String,
    val screenInteractive: Boolean,
    val deviceLocked: Boolean,
    val powerSaveMode: Boolean,
    val networkType: String,
    val networkMetered: Boolean,
    val restrictBackgroundStatus: String,
    val audioActive: Boolean,
    val audioOutput: String
)

class DeviceStateProvider(private val context: Context) {

    fun getState(): DeviceState {
        val batteryIntent = context.registerReceiver(
            null,
            android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED)
        )

        val batteryStatus = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val plugged = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1

        val batteryPct =
            if (level >= 0 && scale > 0) ((level * 100f) / scale).toInt() else null

        val isCharging =
            batteryStatus == BatteryManager.BATTERY_STATUS_CHARGING ||
                    batteryStatus == BatteryManager.BATTERY_STATUS_FULL

        val dockPlugValue =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                BatteryManager.BATTERY_PLUGGED_DOCK
            } else {
                -1
            }

        val plugType = when (plugged) {
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_AC -> "AC"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "WIRELESS"
            dockPlugValue -> "DOCK"
            else -> "NONE"
        }

        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val screenInteractive = powerManager.isInteractive
        val powerSaveMode = powerManager.isPowerSaveMode

        val keyguardManager = context.getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager
        val deviceLocked = keyguardManager.isDeviceLocked

        val connectivityManager =
            context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        val activeNetwork = connectivityManager.activeNetwork
        val caps = connectivityManager.getNetworkCapabilities(activeNetwork)

        val networkType = when {
            caps == null -> "NONE"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "WIFI"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "CELLULAR"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ETHERNET"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_BLUETOOTH) -> "BLUETOOTH"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN) -> "VPN"
            else -> "OTHER"
        }

        val networkMetered = connectivityManager.isActiveNetworkMetered

        val restrictBackgroundStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            when (connectivityManager.restrictBackgroundStatus) {
                ConnectivityManager.RESTRICT_BACKGROUND_STATUS_DISABLED -> "DISABLED"
                ConnectivityManager.RESTRICT_BACKGROUND_STATUS_WHITELISTED -> "WHITELISTED"
                ConnectivityManager.RESTRICT_BACKGROUND_STATUS_ENABLED -> "ENABLED"
                else -> "UNKNOWN"
            }
        } else {
            "UNSUPPORTED"
        }

        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val audioActive = audioManager.isMusicActive
        val audioOutput = getAudioOutput(audioManager)

        return DeviceState(
            batteryPct = batteryPct,
            isCharging = isCharging,
            plugType = plugType,
            screenInteractive = screenInteractive,
            deviceLocked = deviceLocked,
            powerSaveMode = powerSaveMode,
            networkType = networkType,
            networkMetered = networkMetered,
            restrictBackgroundStatus = restrictBackgroundStatus,
            audioActive = audioActive,
            audioOutput = audioOutput
        )
    }

    private fun getAudioOutput(audioManager: AudioManager): String {
        val outputs = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)

        return when {
            outputs.any {
                it.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP ||
                        it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO ||
                        it.type == AudioDeviceInfo.TYPE_BLE_HEADSET ||
                        it.type == AudioDeviceInfo.TYPE_BLE_SPEAKER ||
                        it.type == AudioDeviceInfo.TYPE_BLE_BROADCAST
            } -> "BLUETOOTH"

            outputs.any {
                it.type == AudioDeviceInfo.TYPE_WIRED_HEADPHONES ||
                        it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET ||
                        it.type == AudioDeviceInfo.TYPE_USB_HEADSET
            } -> "WIRED"

            outputs.any { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER } -> "SPEAKER"
            outputs.isEmpty() -> "NONE"
            else -> "OTHER"
        }
    }
}