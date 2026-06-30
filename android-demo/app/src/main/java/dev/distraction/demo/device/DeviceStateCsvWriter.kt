package dev.distraction.demo.device

import java.io.BufferedWriter

class DeviceStateCsvWriter(
    private val writerProvider: () -> BufferedWriter?
) {
    fun writeHeader() {
        writeLine(
            "epoch_s,type,battery_pct,is_charging,plug_type,screen_interactive," +
                    "device_locked,power_save_mode,network_type,network_metered," +
                    "restrict_background_status,audio_active,audio_output"
        )
    }

    fun writeState(epochS: String, state: DeviceState) {
        writeLine(
            listOf(
                epochS,
                "STATE",
                csvOpt(state.batteryPct),
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
            ).joinToString(",")
        )
    }

    fun writeInfo(epochS: String, info: String) {
        writeLine("$epochS,INFO,,,,,,,,,,,$info")
    }

    fun writeError(epochS: String, error: String) {
        writeLine("$epochS,ERR,,,,,,,,,,,$error")
    }

    private fun writeLine(line: String) {
        writerProvider()?.apply {
            write(line)
            newLine()
        }
    }

    private fun csvOpt(value: Any?): String = value?.toString() ?: ""
}