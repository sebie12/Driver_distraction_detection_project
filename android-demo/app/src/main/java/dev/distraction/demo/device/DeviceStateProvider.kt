package dev.distraction.demo.device

import android.app.KeyguardManager
import android.content.Context
import android.media.AudioManager
import android.os.PowerManager

data class DeviceState(
    val screenInteractive: Boolean,
    val deviceLocked: Boolean,
    val audioActive: Boolean
)

class DeviceStateProvider(private val context: Context) {

    fun getState(): DeviceState {
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val keyguardManager = context.getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

        return DeviceState(
            screenInteractive = powerManager.isInteractive,
            deviceLocked = keyguardManager.isDeviceLocked,
            audioActive = audioManager.isMusicActive
        )
    }
}
