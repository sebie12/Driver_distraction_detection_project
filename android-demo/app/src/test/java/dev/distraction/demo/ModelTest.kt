package dev.distraction.demo

import org.junit.Ignore
import org.junit.Test
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.file.Files
import java.nio.ByteBuffer

class ModelTest {
    @Test
    @Ignore("Requires Android environment for TFLite native libs")
    fun testInfer() {
        val file = File("src/main/assets/models_signed/tflite_5s.tflite")
        val buffer = ByteBuffer.wrap(Files.readAllBytes(file.toPath()))
        val interpreter = Interpreter(buffer)
        
        val inputMap = mutableMapOf<String, Any>()
        inputMap["x"] = Array(1) { Array(250) { FloatArray(16) } }
        inputMap["x_1"] = Array(1) { Array(250) { FloatArray(5) } }
        inputMap["x_2"] = Array(1) { Array(250) { FloatArray(3) } }
        
        val outputMap = mutableMapOf<String, Any>()
        val outputBuffer = java.nio.FloatBuffer.allocate(1)
        outputMap["output"] = outputBuffer
        
        println("Running infer...")
        try {
            interpreter.runSignature(inputMap, outputMap, "infer")
            println("Infer success! output=${outputBuffer.get(0)}")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}