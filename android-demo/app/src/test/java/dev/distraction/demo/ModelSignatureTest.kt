package dev.distraction.demo

import org.junit.Ignore
import org.junit.Test
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.file.Files

class ModelSignatureTest {
    @Test
    @Ignore("Requires Android environment for TFLite native libs")
    fun checkSignatures() {
        val file = File("src/main/assets/models_signed/tflite_5s.tflite")
        val buffer = java.nio.ByteBuffer.wrap(Files.readAllBytes(file.toPath()))
        val interpreter = Interpreter(buffer)
        
        val signatures = interpreter.signatureKeys
        println("==================== SIGNATURES ====================")
        for (sig in signatures) {
            println("Signature: $sig")
            val inputs = interpreter.getSignatureInputs(sig)
            println("  Inputs: ${inputs.contentToString()}")
            val outputs = interpreter.getSignatureOutputs(sig)
            println("  Outputs: ${outputs.contentToString()}")
        }
        println("====================================================")
    }
}