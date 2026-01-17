package com.landfall.hybriddetector

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.provider.OpenableColumns
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.provider.Settings
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_THRESHOLD = 0.20f
    }

    private lateinit var outputView: TextView
    private lateinit var statusView: TextView
    private lateinit var runButton: Button
    private lateinit var selectButton: Button
    private lateinit var modelRunner: ModelRunner
    private lateinit var extractor: FeatureExtractor
    private lateinit var attributionModel: AttributionModel
    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            handleSelectedUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        outputView = findViewById(R.id.outputView)
        statusView = findViewById(R.id.statusView)
        runButton = findViewById(R.id.runButton)
        selectButton = findViewById(R.id.selectButton)

        updateStatus("Status: --", R.color.status_neutral)

        modelRunner = ModelRunner.fromAssets(this)
        extractor = FeatureExtractor.fromAssets(this)
        attributionModel = AttributionModel.fromAssets(this)

        selectButton.setOnClickListener {
            pickFileLauncher.launch(arrayOf("*/*"))
        }

        runButton.setOnClickListener {
            if (hasStoragePermission() || hasLocalBenchList()) {
                runBenchmark()
            } else {
                requestStoragePermission()
            }
        }

        if (!hasStoragePermission()) {
            requestStoragePermission()
        }

        if (intent.getBooleanExtra("auto", false) && (hasStoragePermission() || hasLocalBenchList())) {
            runBenchmark()
        }
    }

    private fun hasStoragePermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_MEDIA_IMAGES
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val intent = android.content.Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:$packageName")
            )
            startActivity(intent)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_MEDIA_IMAGES),
                1001
            )
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                1001
            )
        }
    }

    private fun runBenchmark() {
        val paths = loadPaths()
        if (paths.isEmpty()) {
            outputView.text = "No paths found. Put a list in /sdcard/bench_list.txt or " +
                "/sdcard/Android/data/com.landfall.hybriddetector/files/bench_list.txt, " +
                "or pass a path via intent."
            return
        }

        Log.i("HybridDetector", "Running benchmark on ${paths.size} files")
        var totalExtractNs = 0L
        var totalInferNs = 0L
        val sb = StringBuilder()
        sb.append("Files: ").append(paths.size).append("\n")

        for (path in paths) {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val features = extractor.extract(path)
            val t1 = SystemClock.elapsedRealtimeNanos()
            if (features == null) {
                sb.append("FAIL: ").append(path).append("\n")
                Log.w("HybridDetector", "FAIL: $path")
                continue
            }
            val score = modelRunner.predict(features)
            val t2 = SystemClock.elapsedRealtimeNanos()

            val extractMs = (t1 - t0) / 1_000_000.0
            val inferMs = (t2 - t1) / 1_000_000.0
            totalExtractNs += (t1 - t0)
            totalInferNs += (t2 - t1)

            sb.append(String.format("score=%.4f extract=%.3fms infer=%.3fms %s\n", score, extractMs, inferMs, path))
            Log.i("HybridDetector", "score=%.4f extract=%.3fms infer=%.3fms %s".format(score, extractMs, inferMs, path))
        }

        val avgExtractMs = totalExtractNs / 1_000_000.0 / paths.size
        val avgInferMs = totalInferNs / 1_000_000.0 / paths.size
        sb.append(String.format("\nAverage extract=%.3fms infer=%.3fms\n", avgExtractMs, avgInferMs))
        outputView.text = sb.toString()
        Log.i("HybridDetector", "Average extract=%.3fms infer=%.3fms".format(avgExtractMs, avgInferMs))
    }

    private fun handleSelectedUri(uri: Uri) {
        updateStatus("Status: Analyzing...", R.color.status_neutral)
        outputView.text = "Loading sample..."
        Thread {
            val displayName = queryDisplayName(uri)
            val cached = copyToCache(uri, displayName)
            if (cached == null) {
                runOnUiThread {
                    updateStatus("Status: Error", R.color.status_malicious)
                    outputView.text = "Failed to read selected sample."
                }
                return@Thread
            }

            val t0 = SystemClock.elapsedRealtimeNanos()
            val features = extractor.extract(cached.absolutePath)
            val t1 = SystemClock.elapsedRealtimeNanos()
            if (features == null) {
                runOnUiThread {
                    updateStatus("Status: Error", R.color.status_malicious)
                    outputView.text = "Failed to extract features for sample."
                }
                return@Thread
            }
            val score = modelRunner.predict(features)
            val t2 = SystemClock.elapsedRealtimeNanos()

            val extractMs = (t1 - t0) / 1_000_000.0
            val inferMs = (t2 - t1) / 1_000_000.0
            val isMalicious = score >= MODEL_THRESHOLD
            val label = if (isMalicious) "MALICIOUS" else "BENIGN"

            val tiff = TiffParser.parse(cached.absolutePath)
            val sb = StringBuilder()
            sb.append("Selected: ").append(displayName ?: cached.name).append("\n")
            sb.append("Cache: ").append(cached.absolutePath).append("\n")
            sb.append("Size: ").append(cached.length()).append(" bytes\n")
            sb.append(String.format("Score: %.6f (thr=%.2f) -> %s\n", score, MODEL_THRESHOLD, label))
            sb.append(String.format("Extract: %.3fms  Infer: %.3fms\n", extractMs, inferMs))
            sb.append("\n")
            sb.append("TIFF: ").append(if (tiff.isTiff == 1) "yes" else "no")
                .append("  DNG: ").append(if (tiff.isDng == 1) "yes" else "no").append("\n")
            sb.append("Min dims: ").append(tiff.minWidth).append(" x ").append(tiff.minHeight).append("\n")
            sb.append("IFD max entries: ").append(tiff.ifdEntryMax).append("\n")
            sb.append("SubIFD count: ").append(tiff.subifdCountSum).append("\n")
            sb.append("NewSubfileType unique: ").append(tiff.newSubfileTypesUnique).append("\n")
            sb.append("Opcodes total/unknown/max: ")
                .append(tiff.totalOpcodes).append("/")
                .append(tiff.unknownOpcodes).append("/")
                .append(tiff.maxOpcodeId).append("\n")
            sb.append("Opcode list bytes: ")
                .append(tiff.opcodeList1Bytes).append("/")
                .append(tiff.opcodeList2Bytes).append("/")
                .append(tiff.opcodeList3Bytes).append("\n")
            sb.append("ZIP flags (eocd/local): ")
                .append(tiff.zipEocdNearEnd).append("/")
                .append(tiff.zipLocalInTail).append("\n")
            sb.append("Rule flags: opcode_anom=")
                .append(tiff.flagOpcodeAnomaly)
                .append(" tiny_dims=").append(tiff.flagTinyDimsLowIfd)
                .append(" zip_polyglot=").append(tiff.flagZipPolyglot)
                .append(" dng_jpeg=").append(tiff.flagDngJpegMismatch)
                .append(" any=").append(tiff.flagAny)
                .append("\n")

            val attribution = attributionModel.computeTopContributions(features, 5)
            if (attribution != null) {
                sb.append("\nTop positive contributions:\n")
                for (c in attribution.topPositive) {
                    sb.append(String.format("  +%.6f %s (x=%.4f w=%.4f)\n",
                        c.contribution, c.name, c.value, c.weight))
                }
                sb.append("Top negative contributions:\n")
                for (c in attribution.topNegative) {
                    sb.append(String.format("  %.6f %s (x=%.4f w=%.4f)\n",
                        c.contribution, c.name, c.value, c.weight))
                }
                sb.append(String.format("Attribution score: %.6f (logit=%.4f)\n",
                    attribution.score, attribution.logit))
            }

            runOnUiThread {
                if (isMalicious) {
                    updateStatus("Status: MALICIOUS", R.color.status_malicious)
                } else {
                    updateStatus("Status: BENIGN", R.color.status_benign)
                }
                outputView.text = sb.toString()
            }
        }.start()
    }

    private fun updateStatus(text: String, colorRes: Int) {
        statusView.text = text
        statusView.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun queryDisplayName(uri: Uri): String? {
        val cursor = contentResolver.query(uri, null, null, null, null) ?: return null
        cursor.use {
            val nameIdx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIdx >= 0 && it.moveToFirst()) {
                return it.getString(nameIdx)
            }
        }
        return null
    }

    private fun copyToCache(uri: Uri, displayName: String?): File? {
        val safeName = (displayName ?: "sample.bin").replace(Regex("[^A-Za-z0-9._-]"), "_")
        val outFile = File(cacheDir, "sample_${System.currentTimeMillis()}_$safeName")
        return try {
            contentResolver.openInputStream(uri).use { input ->
                if (input == null) return null
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(8192)
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                    }
                }
            }
            outFile
        } catch (e: Exception) {
            Log.w("HybridDetector", "Failed to cache selected uri: ${e.message}")
            null
        }
    }

    private fun loadPaths(): List<String> {
        val intentPath = intent.getStringExtra("path")
        if (!intentPath.isNullOrBlank()) {
            return listOf(intentPath)
        }

        val localListFile = File(getExternalFilesDir(null), "bench_list.txt")
        if (localListFile.exists()) {
            return readListFile(localListFile)
        }

        val listFile = File("/sdcard/bench_list.txt")
        if (listFile.exists()) {
            return readListFile(listFile)
        }

        return emptyList()
    }

    private fun readListFile(file: File): List<String> {
        return try {
            file.readLines().map { it.trim() }.filter { it.isNotEmpty() }
        } catch (e: Exception) {
            Log.w("HybridDetector", "Failed to read list file ${file.absolutePath}: ${e.message}")
            emptyList()
        }
    }

    private fun hasLocalBenchList(): Boolean {
        val localListFile = File(getExternalFilesDir(null), "bench_list.txt")
        return localListFile.exists()
    }
}
