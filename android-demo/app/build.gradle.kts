import com.android.build.api.artifact.SingleArtifact
import org.gradle.api.GradleException

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "dev.distraction.demo"

    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "dev.distraction.demo"
        minSdk = 28
        targetSdk = 36
        versionCode = 3
        versionName = "1.0.2"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug { }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

/**
 * Nome base para o APK “exportado”.
 * (O APK original continua a ter o nome default do AGP; nós criamos uma cópia com este nome.)
 */
val exportedApkAppName = "SensorsDataCollection"

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.play.services.location)

    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

androidComponents {
    onVariants { variant ->
        val apkFolderProvider = variant.artifacts.get(SingleArtifact.APK)

        val capName = variant.name.replaceFirstChar { it.uppercaseChar() }

        val exportTask = tasks.register("export${capName}Apk") {
            inputs.dir(apkFolderProvider)

            val outDirProvider = layout.buildDirectory.dir("outputs/exported-apk/${variant.name}")
            outputs.dir(outDirProvider)

            doLast {
                val apkFolder = apkFolderProvider.get().asFile
                val outDir = outDirProvider.get().asFile
                outDir.mkdirs()

                val apks = apkFolder.walkTopDown()
                    .filter { it.isFile && it.extension.equals("apk", ignoreCase = true) }
                    .toList()

                if (apks.isEmpty()) {
                    throw GradleException("Não foi encontrado nenhum APK em: ${apkFolder.absolutePath}")
                }

                val vName = android.defaultConfig.versionName ?: "0.0"
                val vCode = android.defaultConfig.versionCode
                val appName = "SensorsDataCollection"

                apks.forEachIndexed { index, apk ->
                    val suffix = if (apks.size == 1) "" else "_part${index + 1}"
                    val newName = "${appName}_v${vName}(${vCode})_${variant.name}$suffix.apk"
                    apk.copyTo(File(outDir, newName), overwrite = true)
                }

                println("APK(s) exportado(s) para: ${outDir.absolutePath}")
            }
        }

        // Tenta ligar o export ao task de packaging mais comum.
        // Não falha se o task não existir.
        listOf(
            "package$capName",   // comum em muitos projetos
            "assemble$capName",  // fallback clássico
            "bundle$capName"     // só por precaução
        ).firstNotNullOfOrNull { taskName ->
            tasks.findByName(taskName)?.also { t ->
                t.finalizedBy(exportTask)
            }
        }
        // Se nenhum existir, tudo bem: corres o export manualmente (ver abaixo).
    }
}