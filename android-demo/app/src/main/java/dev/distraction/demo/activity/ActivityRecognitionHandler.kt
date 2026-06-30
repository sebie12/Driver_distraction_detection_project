package dev.distraction.demo.activity

import android.Manifest
import android.annotation.SuppressLint
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.google.android.gms.location.ActivityRecognition
import com.google.android.gms.location.ActivityRecognitionClient
import com.google.android.gms.location.ActivityRecognitionResult
import com.google.android.gms.location.ActivityTransition
import com.google.android.gms.location.ActivityTransitionEvent
import com.google.android.gms.location.ActivityTransitionRequest
import com.google.android.gms.location.ActivityTransitionResult
import com.google.android.gms.location.DetectedActivity
import dev.distraction.demo.common.TimeUtils

class ActivityRecognitionHandler(
    private val context: Context,
    private val activityCsvWriter: ActivityCsvWriter,
    private val baseEpochNsProvider: () -> Long
) {
    val client: ActivityRecognitionClient = ActivityRecognition.getClient(context)

    val updatesPendingIntent: PendingIntent =
        PendingIntent.getBroadcast(
            context,
            100,
            Intent(ACTION_ACTIVITY_UPDATE).setPackage(context.packageName),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
        )

    val transitionsPendingIntent: PendingIntent =
        PendingIntent.getBroadcast(
            context,
            101,
            Intent(ACTION_ACTIVITY_TRANSITION).setPackage(context.packageName),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
        )

    fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACTIVITY_RECOGNITION
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (!hasPermission()) {
            val elapsedNs = SystemClock.elapsedRealtimeNanos()
            val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
            activityCsvWriter.writeError(epochS, "NO_PERMISSION", elapsedNs)
            return
        }

        client.requestActivityUpdates(5000L, updatesPendingIntent)
            .addOnSuccessListener {
                val elapsedNs = SystemClock.elapsedRealtimeNanos()
                val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
                activityCsvWriter.writeInfo(epochS, "REQUEST_UPDATES_OK", elapsedNs)
            }
            .addOnFailureListener {
                val elapsedNs = SystemClock.elapsedRealtimeNanos()
                val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
                activityCsvWriter.writeError(epochS, "REQUEST_UPDATES_FAILED", elapsedNs)
            }

        val request = ActivityTransitionRequest(buildTransitions())
        client.requestActivityTransitionUpdates(request, transitionsPendingIntent)
            .addOnSuccessListener {
                val elapsedNs = SystemClock.elapsedRealtimeNanos()
                val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
                activityCsvWriter.writeInfo(epochS, "REQUEST_TRANSITIONS_OK", elapsedNs)
            }
            .addOnFailureListener {
                val elapsedNs = SystemClock.elapsedRealtimeNanos()
                val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
                activityCsvWriter.writeError(epochS, "REQUEST_TRANSITIONS_FAILED", elapsedNs)
            }
    }

    fun stop() {
        client.removeActivityUpdates(updatesPendingIntent)
        client.removeActivityTransitionUpdates(transitionsPendingIntent)
    }

    fun handleActivityUpdateIntent(intent: Intent) {
        val result = ActivityRecognitionResult.extractResult(intent)
        if (result == null) {
            val elapsedNs = SystemClock.elapsedRealtimeNanos()
            val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
            activityCsvWriter.writeError(epochS, "UPDATE_INTENT_WITHOUT_RESULT", elapsedNs)
            return
        }

        val probable = result.probableActivities
        if (probable.isNullOrEmpty()) {
            val elapsedNs = SystemClock.elapsedRealtimeNanos()
            val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
            activityCsvWriter.writeError(epochS, "UPDATE_WITHOUT_ACTIVITIES", elapsedNs)
            return
        }

        val best = probable.maxByOrNull { it.confidence } ?: return

        val elapsedNs = SystemClock.elapsedRealtimeNanos()
        val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
        activityCsvWriter.writeCurrent(epochS, activityToString(best.type), best.confidence, elapsedNs)
    }

    fun handleActivityTransitionIntent(intent: Intent) {
        if (!ActivityTransitionResult.hasResult(intent)) return

        val result = ActivityTransitionResult.extractResult(intent) ?: return
        for (event: ActivityTransitionEvent in result.transitionEvents) {
            val elapsedNs = event.elapsedRealTimeNanos
            val epochS = TimeUtils.formatEpochSeconds(baseEpochNsProvider() + elapsedNs)
            activityCsvWriter.writeTransition(
                epochS,
                activityToString(event.activityType),
                transitionToString(event.transitionType),
                elapsedNs
            )
        }
    }

    private fun buildTransitions(): List<ActivityTransition> {
        val activityTypes = listOf(
            DetectedActivity.IN_VEHICLE,
            DetectedActivity.ON_BICYCLE,
            DetectedActivity.ON_FOOT,
            DetectedActivity.RUNNING,
            DetectedActivity.STILL,
            DetectedActivity.WALKING
        )

        val transitions = mutableListOf<ActivityTransition>()

        for (type in activityTypes) {
            transitions += ActivityTransition.Builder()
                .setActivityType(type)
                .setActivityTransition(ActivityTransition.ACTIVITY_TRANSITION_ENTER)
                .build()

            transitions += ActivityTransition.Builder()
                .setActivityType(type)
                .setActivityTransition(ActivityTransition.ACTIVITY_TRANSITION_EXIT)
                .build()
        }

        return transitions
    }

    private fun activityToString(type: Int): String {
        return when (type) {
            DetectedActivity.IN_VEHICLE -> "IN_VEHICLE"
            DetectedActivity.ON_BICYCLE -> "ON_BICYCLE"
            DetectedActivity.ON_FOOT -> "ON_FOOT"
            DetectedActivity.RUNNING -> "RUNNING"
            DetectedActivity.STILL -> "STILL"
            DetectedActivity.WALKING -> "WALKING"
            DetectedActivity.TILTING -> "TILTING"
            DetectedActivity.UNKNOWN -> "UNKNOWN"
            else -> "ACT_$type"
        }
    }

    private fun transitionToString(type: Int): String {
        return when (type) {
            ActivityTransition.ACTIVITY_TRANSITION_ENTER -> "ENTER"
            ActivityTransition.ACTIVITY_TRANSITION_EXIT -> "EXIT"
            else -> "TRANS_$type"
        }
    }

    companion object {
        const val ACTION_ACTIVITY_UPDATE =
            "dev.distraction.demo.ACTIVITY_UPDATE"
        const val ACTION_ACTIVITY_TRANSITION =
            "dev.distraction.demo.ACTIVITY_TRANSITION"
    }
}