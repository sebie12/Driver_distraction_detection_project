package dev.distraction.demo.ml.ui

import android.app.AlertDialog
import android.content.Context

object ModelSelectorDialog {

    fun show(
        context: Context,
        models: List<ModelOptionUi>,
        onSelected: (ModelOptionUi) -> Unit
    ) {
        if (models.isEmpty()) return

        val titles = models.map { option ->
            if (option.subtitle.isBlank()) {
                option.title
            } else {
                "${option.title}\n${option.subtitle}"
            }
        }.toTypedArray()

        val checkedIndex = models.indexOfFirst { it.isSelected }.coerceAtLeast(0)

        AlertDialog.Builder(context)
            .setTitle("Selecionar modelo")
            .setSingleChoiceItems(titles, checkedIndex) { dialog, which ->
                onSelected(models[which])
                dialog.dismiss()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }
}