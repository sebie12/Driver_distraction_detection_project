package dev.distraction.demo.feedback

import android.app.Activity
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import androidx.annotation.RequiresApi
import dev.distraction.demo.service.LoggingService

class DistractionFeedbackActivity : Activity() {

    private var responded = false

    private val dismissReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == LoggingService.ACTION_DISMISS_FEEDBACK) {
                if (!responded) {
                    sendFeedback(false) // Assume no distraction
                }
                finish()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        val filter = IntentFilter(LoggingService.ACTION_DISMISS_FEEDBACK)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(dismissReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(dismissReceiver, filter)
        }

        AlertDialog.Builder(this)
            .setTitle("Distração?")
            .setMessage("Estavas distraído nos últimos 2.5 segundos?")
            .setPositiveButton("Sim") { _, _ ->
                responded = true
                sendFeedback(true)
                finish()
            }
            .setCancelable(false)
            .show()
    }

    override fun onDestroy() {
        unregisterReceiver(dismissReceiver)
        super.onDestroy()
    }
    
    private fun sendFeedback(wasDistracted: Boolean) {
        val intent = Intent(this, LoggingService::class.java).apply {
            action = LoggingService.ACTION_FEEDBACK
            putExtra(LoggingService.EXTRA_WAS_DISTRACTED, wasDistracted)
        }
        startService(intent)
    }
}
