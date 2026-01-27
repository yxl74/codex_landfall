package com.landfall.hybriddetector

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
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

    private lateinit var summaryView: TextView
    private lateinit var statusView: TextView
    private lateinit var cveResultView: TextView
    private lateinit var hybridResultView: TextView
    private lateinit var anomalyResultView: TextView
    private lateinit var magikaResultView: TextView
    private lateinit var tagseqResultView: TextView
    private lateinit var hybridDetailView: TextView
    private lateinit var anomalyDetailView: TextView
    private lateinit var magikaDetailView: TextView
    private lateinit var tagseqDetailView: TextView
    private lateinit var runButton: Button
    private lateinit var selectButton: Button
    private lateinit var modelRunner: ModelRunner
    private lateinit var extractor: FeatureExtractor
    private lateinit var attributionModel: AttributionModel
    private lateinit var anomalyExtractor: AnomalyFeatureExtractor
    private lateinit var anomalyRunner: AnomalyModelRunner
    private lateinit var anomalyMeta: AnomalyMeta
    private lateinit var magikaRunner: MagikaModelRunner
    private lateinit var tagSeqMeta: TagSeqMeta
    private lateinit var tagSeqRunner: TagSeqModelRunner
    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            handleSelectedUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        summaryView = findViewById(R.id.summaryView)
        statusView = findViewById(R.id.statusView)
        cveResultView = findViewById(R.id.cveResultView)
        hybridResultView = findViewById(R.id.hybridResultView)
        anomalyResultView = findViewById(R.id.anomalyResultView)
        magikaResultView = findViewById(R.id.magikaResultView)
        tagseqResultView = findViewById(R.id.tagseqResultView)
        hybridDetailView = findViewById(R.id.hybridDetailView)
        anomalyDetailView = findViewById(R.id.anomalyDetailView)
        magikaDetailView = findViewById(R.id.magikaDetailView)
        tagseqDetailView = findViewById(R.id.tagseqDetailView)
        runButton = findViewById(R.id.runButton)
        selectButton = findViewById(R.id.selectButton)

        updateStatus("Status: --", R.color.status_neutral)
        cveResultView.visibility = View.GONE
        updateResult(hybridResultView, "Hybrid: --", R.color.status_neutral)
        updateResult(anomalyResultView, "Anomaly AE: --", R.color.status_neutral)
        updateResult(magikaResultView, "Magika: --", R.color.status_neutral)
        updateResult(tagseqResultView, "TagSeq GRU-AE: --", R.color.status_neutral)
        updateResult(hybridDetailView, "Hybrid details...", R.color.status_neutral)
        updateResult(anomalyDetailView, "Anomaly details...", R.color.status_neutral)
        updateResult(magikaDetailView, "Magika details...", R.color.status_neutral)
        updateResult(tagseqDetailView, "TagSeq details...", R.color.status_neutral)

        modelRunner = ModelRunner.fromAssets(this)
        extractor = FeatureExtractor.fromAssets(this)
        attributionModel = AttributionModel.fromAssets(this)
        anomalyMeta = AnomalyMeta.fromAssets(this)
        anomalyExtractor = AnomalyFeatureExtractor.fromMeta(anomalyMeta)
        anomalyRunner = AnomalyModelRunner.fromAssets(this, anomalyMeta.threshold)
        magikaRunner = MagikaModelRunner.fromAssets(this)
        tagSeqMeta = TagSeqMeta.fromAssets(this)
        tagSeqRunner = TagSeqModelRunner.fromAssets(this, tagSeqMeta)

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
            summaryView.text = "No paths found. Put a list in /sdcard/bench_list.txt or " +
                "/sdcard/Android/data/com.landfall.hybriddetector/files/bench_list.txt, " +
                "or pass a path via intent."
            updateResult(hybridDetailView, "Hybrid details...", R.color.status_neutral)
            updateResult(anomalyDetailView, "Anomaly details...", R.color.status_neutral)
            updateResult(magikaDetailView, "Magika details...", R.color.status_neutral)
            updateResult(tagseqDetailView, "TagSeq details...", R.color.status_neutral)
            return
        }

        cveResultView.visibility = View.GONE
        updateResult(hybridResultView, "Hybrid: (batch)", R.color.status_neutral)
        updateResult(anomalyResultView, "Anomaly AE: (batch)", R.color.status_neutral)
        updateResult(magikaResultView, "Magika: (single-file)", R.color.status_neutral)
        updateResult(tagseqResultView, "TagSeq GRU-AE: (batch)", R.color.status_neutral)
        updateResult(hybridDetailView, "Hybrid details...", R.color.status_neutral)
        updateResult(anomalyDetailView, "Anomaly details...", R.color.status_neutral)
        updateResult(magikaDetailView, "Magika details...", R.color.status_neutral)
        updateResult(tagseqDetailView, "TagSeq details...", R.color.status_neutral)
        Log.i("HybridDetector", "Running benchmark on ${paths.size} files")
        var totalExtractNs = 0L
        var totalInferNs = 0L
        var totalAnomalyExtractNs = 0L
        var totalAnomalyInferNs = 0L
        var totalTagSeqExtractNs = 0L
        var totalTagSeqInferNs = 0L
        var totalCveNs = 0L
        var hybridCount = 0
        var anomalyCount = 0
        var tagSeqCount = 0
        var cveHitCount = 0
        val summarySb = StringBuilder()
        summarySb.append("Files: ").append(paths.size).append("\n")
        summarySb.append(String.format(
            "Hybrid thr=%.2f  Anomaly thr=%.3f  TagSeq thr=%.3f\n",
            MODEL_THRESHOLD,
            anomalyMeta.threshold,
            tagSeqMeta.threshold
        ))
        val hybridSb = StringBuilder()
        val anomalySb = StringBuilder()
        val tagSeqSb = StringBuilder()

        for (path in paths) {
            // TIER 1: Parse once, run CVE rules
            val tiffFeatures = TiffParser.parse(path)
            val c0 = SystemClock.elapsedRealtimeNanos()
            val cveResult = CveDetector.evaluate(tiffFeatures)
            val c1 = SystemClock.elapsedRealtimeNanos()
            totalCveNs += (c1 - c0)
            if (cveResult.hasCveHits) cveHitCount++

            val cvePart = if (cveResult.hasCveHits) {
                val ids = cveResult.hits.joinToString(",") { it.cveId }
                "CVE=$ids"
            } else {
                "CVE=none"
            }

            // TIER 2: ML models (run for all files in benchmark mode for comparison)
            val h0 = SystemClock.elapsedRealtimeNanos()
            val features = extractor.extractFromParsed(tiffFeatures, path)
            val h1 = SystemClock.elapsedRealtimeNanos()
            val score = if (features != null) modelRunner.predict(features) else null
            val h2 = SystemClock.elapsedRealtimeNanos()

            val a0 = SystemClock.elapsedRealtimeNanos()
            val anomalyFeatures = anomalyExtractor.extractFromParsed(tiffFeatures, path)
            val a1 = SystemClock.elapsedRealtimeNanos()
            val anomalyScore = if (anomalyFeatures != null) anomalyRunner.predict(anomalyFeatures) else null
            val a2 = SystemClock.elapsedRealtimeNanos()

            val t0 = SystemClock.elapsedRealtimeNanos()
            val tagSeqInput = TagSequenceExtractor.extract(path, tagSeqMeta.maxSeqLen)
            val t1 = SystemClock.elapsedRealtimeNanos()
            val tagSeqScore = if (tagSeqInput != null && tagSeqInput.isDng) {
                tagSeqRunner.predict(tagSeqInput)
            } else {
                null
            }
            val t2 = SystemClock.elapsedRealtimeNanos()

            val hExtractMs = (h1 - h0) / 1_000_000.0
            val hInferMs = (h2 - h1) / 1_000_000.0
            val aExtractMs = (a1 - a0) / 1_000_000.0
            val aInferMs = (a2 - a1) / 1_000_000.0
            val tExtractMs = (t1 - t0) / 1_000_000.0
            val tInferMs = (t2 - t1) / 1_000_000.0

            if (score != null) {
                totalExtractNs += (h1 - h0)
                totalInferNs += (h2 - h1)
                hybridCount += 1
            }
            if (anomalyScore != null) {
                totalAnomalyExtractNs += (a1 - a0)
                totalAnomalyInferNs += (a2 - a1)
                anomalyCount += 1
            }
            if (tagSeqScore != null) {
                totalTagSeqExtractNs += (t1 - t0)
                totalTagSeqInferNs += (t2 - t1)
                tagSeqCount += 1
            }

            val hybridPart = if (score != null) {
                val label = if (score >= MODEL_THRESHOLD) "MAL" else "BEN"
                String.format("hybrid=%.4f(%s) ex=%.2fms in=%.2fms", score, label, hExtractMs, hInferMs)
            } else {
                "hybrid=FAIL"
            }
            val anomalyPart = if (anomalyScore != null) {
                val label = if (anomalyScore >= anomalyMeta.threshold) "MAL" else "BEN"
                String.format("anomaly=%.4f(%s) ex=%.2fms in=%.2fms", anomalyScore, label, aExtractMs, aInferMs)
            } else {
                "anomaly=FAIL"
            }
            val tagSeqPart = if (tagSeqScore != null) {
                val label = if (tagSeqScore >= tagSeqMeta.threshold) "MAL" else "BEN"
                String.format("tagseq=%.4f(%s) ex=%.2fms in=%.2fms", tagSeqScore, label, tExtractMs, tInferMs)
            } else {
                "tagseq=N/A"
            }

            hybridSb.append(hybridPart).append(" | ").append(cvePart).append(" | ").append(path).append("\n")
            anomalySb.append(anomalyPart).append(" | ").append(path).append("\n")
            tagSeqSb.append(tagSeqPart).append(" | ").append(path).append("\n")
            Log.i("HybridDetector", "$cvePart | $hybridPart | $anomalyPart | $tagSeqPart | $path")
        }

        val avgCveMs = if (paths.isNotEmpty()) totalCveNs / 1_000_000.0 / paths.size else 0.0
        summarySb.append(String.format("CVE hits: %d  avg CVE scan: %.3fms\n", cveHitCount, avgCveMs))

        if (hybridCount > 0) {
            val avgExtractMs = totalExtractNs / 1_000_000.0 / hybridCount
            val avgInferMs = totalInferNs / 1_000_000.0 / hybridCount
            hybridSb.append(String.format("\nAvg extract=%.3fms infer=%.3fms\n", avgExtractMs, avgInferMs))
            Log.i("HybridDetector", "Hybrid avg extract=%.3fms infer=%.3fms".format(avgExtractMs, avgInferMs))
        }
        if (anomalyCount > 0) {
            val avgExtractMs = totalAnomalyExtractNs / 1_000_000.0 / anomalyCount
            val avgInferMs = totalAnomalyInferNs / 1_000_000.0 / anomalyCount
            anomalySb.append(String.format("\nAvg extract=%.3fms infer=%.3fms\n", avgExtractMs, avgInferMs))
            Log.i("HybridDetector", "Anomaly avg extract=%.3fms infer=%.3fms".format(avgExtractMs, avgInferMs))
        }
        if (tagSeqCount > 0) {
            val avgExtractMs = totalTagSeqExtractNs / 1_000_000.0 / tagSeqCount
            val avgInferMs = totalTagSeqInferNs / 1_000_000.0 / tagSeqCount
            tagSeqSb.append(String.format("\nAvg extract=%.3fms infer=%.3fms\n", avgExtractMs, avgInferMs))
            Log.i("HybridDetector", "TagSeq avg extract=%.3fms infer=%.3fms".format(avgExtractMs, avgInferMs))
        }
        summaryView.text = summarySb.toString()
        updateResult(hybridDetailView, hybridSb.toString(), R.color.status_neutral)
        updateResult(anomalyDetailView, anomalySb.toString(), R.color.status_neutral)
        updateResult(tagseqDetailView, tagSeqSb.toString(), R.color.status_neutral)
    }

    private fun handleSelectedUri(uri: Uri) {
        updateStatus("Status: Analyzing...", R.color.status_neutral)
        summaryView.text = "Loading sample..."
        cveResultView.visibility = View.GONE
        updateResult(hybridDetailView, "Hybrid details...", R.color.status_neutral)
        updateResult(anomalyDetailView, "Anomaly details...", R.color.status_neutral)
        updateResult(magikaDetailView, "Magika details...", R.color.status_neutral)
        updateResult(tagseqDetailView, "TagSeq details...", R.color.status_neutral)
        Thread {
            val displayName = queryDisplayName(uri)
            val cached = copyToCache(uri, displayName)
            if (cached == null) {
                runOnUiThread {
                    updateStatus("Status: Error", R.color.status_malicious)
                    summaryView.text = "Failed to read selected sample."
                }
                return@Thread
            }

            // TIER 1: Parse once, run CVE rules
            val tiff = TiffParser.parse(cached.absolutePath)
            val cveResult = CveDetector.evaluate(tiff)

            if (cveResult.hasCveHits) {
                // CVE detected — show CVE verdict, skip ML
                val cveIds = cveResult.hits.joinToString(", ") { it.cveId }
                val cveSb = StringBuilder()
                cveSb.append("CVE DETECTED: $cveIds\n")
                for (hit in cveResult.hits) {
                    cveSb.append("  ${hit.cveId}: ${hit.description}\n")
                }
                cveSb.append(String.format("CVE scan: %.3fms\n", cveResult.scanTimeMs))

                val summarySb = StringBuilder()
                summarySb.append("Selected: ").append(displayName ?: cached.name).append("\n")
                summarySb.append("Cache: ").append(cached.absolutePath).append("\n")
                summarySb.append("Size: ").append(cached.length()).append(" bytes\n")
                summarySb.append("Decision: MALICIOUS (CVE rule match)\n\n")
                summarySb.append("TIFF: ").append(if (tiff.isTiff == 1) "yes" else "no")
                    .append("  DNG: ").append(if (tiff.isDng == 1) "yes" else "no").append("\n")
                summarySb.append("Max declared opcode count: ").append(tiff.maxDeclaredOpcodeCount).append("\n")
                summarySb.append("Compression values: ").append(tiff.compressionValues).append("\n")
                summarySb.append("SPP values: ").append(tiff.sppValues).append("\n")
                summarySb.append("SOF3 mismatch: ").append(tiff.sof3ComponentMismatch).append("\n")
                summarySb.append("Tile offsets/bytecounts: ")
                    .append(tiff.tileOffsetsCount).append("/")
                    .append(tiff.tileByteCountsCount).append("\n")
                summarySb.append("Expected tile count: ").append(tiff.expectedTileCount).append("\n")

                runOnUiThread {
                    updateStatus("Status: MALICIOUS", R.color.status_malicious)
                    cveResultView.visibility = View.VISIBLE
                    updateResult(cveResultView, cveSb.toString(), R.color.status_malicious)
                    summaryView.text = summarySb.toString()
                    updateResult(hybridResultView, "Hybrid: skipped (CVE)", R.color.status_neutral)
                    updateResult(anomalyResultView, "Anomaly AE: skipped (CVE)", R.color.status_neutral)
                    updateResult(magikaResultView, "Magika: skipped (CVE)", R.color.status_neutral)
                    updateResult(tagseqResultView, "TagSeq: skipped (CVE)", R.color.status_neutral)
                    updateResult(hybridDetailView, "Skipped — CVE rule triggered", R.color.status_neutral)
                    updateResult(anomalyDetailView, "Skipped — CVE rule triggered", R.color.status_neutral)
                    updateResult(magikaDetailView, "Skipped — CVE rule triggered", R.color.status_neutral)
                    updateResult(tagseqDetailView, "Skipped — CVE rule triggered", R.color.status_neutral)
                }
                return@Thread
            }

            // TIER 2: No CVE hit — run ML models (reuse tiffFeatures to avoid re-parsing)
            val h0 = SystemClock.elapsedRealtimeNanos()
            val features = extractor.extractFromParsed(tiff, cached.absolutePath)
            val h1 = SystemClock.elapsedRealtimeNanos()
            val score = if (features != null) modelRunner.predict(features) else null
            val h2 = SystemClock.elapsedRealtimeNanos()

            val a0 = SystemClock.elapsedRealtimeNanos()
            val anomalyFeatures = anomalyExtractor.extractFromParsed(tiff, cached.absolutePath)
            val a1 = SystemClock.elapsedRealtimeNanos()
            val anomalyScore = if (anomalyFeatures != null) anomalyRunner.predict(anomalyFeatures) else null
            val a2 = SystemClock.elapsedRealtimeNanos()

            val m0 = SystemClock.elapsedRealtimeNanos()
            val magikaResult = magikaRunner.classify(cached.absolutePath)
            val m1 = SystemClock.elapsedRealtimeNanos()

            val t0 = SystemClock.elapsedRealtimeNanos()
            val tagSeqInput = TagSequenceExtractor.extract(cached.absolutePath, tagSeqMeta.maxSeqLen)
            val t1 = SystemClock.elapsedRealtimeNanos()
            val tagSeqScore = if (tagSeqInput != null && tagSeqInput.isDng) {
                tagSeqRunner.predict(tagSeqInput)
            } else {
                null
            }
            val t2 = SystemClock.elapsedRealtimeNanos()

            if (score == null && anomalyScore == null && magikaResult == null && tagSeqScore == null) {
                runOnUiThread {
                    updateStatus("Status: Error", R.color.status_malicious)
                    summaryView.text = "Failed to extract features for sample."
                }
                return@Thread
            }

            val hExtractMs = (h1 - h0) / 1_000_000.0
            val hInferMs = (h2 - h1) / 1_000_000.0
            val aExtractMs = (a1 - a0) / 1_000_000.0
            val aInferMs = (a2 - a1) / 1_000_000.0
            val mInferMs = (m1 - m0) / 1_000_000.0
            val tExtractMs = (t1 - t0) / 1_000_000.0
            val tInferMs = (t2 - t1) / 1_000_000.0
            val hybridMalicious = score != null && score >= MODEL_THRESHOLD
            val anomalyMalicious = anomalyScore != null && anomalyScore >= anomalyMeta.threshold
            val tagSeqMalicious = tagSeqScore != null && tagSeqScore >= tagSeqMeta.threshold
            val isMalicious = hybridMalicious || anomalyMalicious
            val label = if (isMalicious) "MALICIOUS" else "BENIGN"

            val summarySb = StringBuilder()
            summarySb.append("Selected: ").append(displayName ?: cached.name).append("\n")
            summarySb.append("Cache: ").append(cached.absolutePath).append("\n")
            summarySb.append("Size: ").append(cached.length()).append(" bytes\n")
            summarySb.append(String.format("Decision: %s\n", label))
            summarySb.append(String.format("CVE scan: %.3fms (no hits)\n", cveResult.scanTimeMs))
            summarySb.append("\n")
            if (magikaResult != null) {
                summarySb.append(String.format(
                    "Magika: %s (score=%.3f)\n",
                    magikaResult.outputLabel,
                    magikaResult.score
                ))
                summarySb.append("\n")
            }
            if (tagSeqScore != null) {
                summarySb.append(String.format(
                    "TagSeq GRU-AE: %.4f (thr=%.3f)\n",
                    tagSeqScore,
                    tagSeqMeta.threshold
                ))
                summarySb.append("\n")
            }
            summarySb.append("TIFF: ").append(if (tiff.isTiff == 1) "yes" else "no")
                .append("  DNG: ").append(if (tiff.isDng == 1) "yes" else "no").append("\n")
            summarySb.append("Min dims: ").append(tiff.minWidth).append(" x ").append(tiff.minHeight).append("\n")
            summarySb.append("IFD max entries: ").append(tiff.ifdEntryMax).append("\n")
            summarySb.append("SubIFD count: ").append(tiff.subifdCountSum).append("\n")
            summarySb.append("NewSubfileType unique: ").append(tiff.newSubfileTypesUnique).append("\n")
            summarySb.append("Opcodes total/unknown/max: ")
                .append(tiff.totalOpcodes).append("/")
                .append(tiff.unknownOpcodes).append("/")
                .append(tiff.maxOpcodeId).append("\n")
            summarySb.append("Opcode list bytes: ")
                .append(tiff.opcodeList1Bytes).append("/")
                .append(tiff.opcodeList2Bytes).append("/")
                .append(tiff.opcodeList3Bytes).append("\n")
            summarySb.append("ZIP flags (eocd/local): ")
                .append(tiff.zipEocdNearEnd).append("/")
                .append(tiff.zipLocalInTail).append("\n")
            summarySb.append("Rule flags: opcode_anom=")
                .append(tiff.flagOpcodeAnomaly)
                .append(" tiny_dims=").append(tiff.flagTinyDimsLowIfd)
                .append(" zip_polyglot=").append(tiff.flagZipPolyglot)
                .append(" dng_jpeg=").append(tiff.flagDngJpegMismatch)
                .append(" any=").append(tiff.flagAny)
                .append("\n")

            val hybridSb = StringBuilder()
            if (score != null) {
                hybridSb.append(String.format("Score: %.6f\n", score))
                hybridSb.append(String.format("Threshold: %.2f\n", MODEL_THRESHOLD))
                hybridSb.append(String.format("Result: %s\n", if (hybridMalicious) "MALICIOUS" else "BENIGN"))
                hybridSb.append(String.format("Extract: %.3fms\nInfer: %.3fms\n", hExtractMs, hInferMs))
            } else {
                hybridSb.append("Score: FAIL\n")
            }

            val attribution = if (features != null) {
                attributionModel.computeTopContributions(features, 5)
            } else {
                null
            }
            if (attribution != null) {
                hybridSb.append("\nTop positive contributions:\n")
                for (c in attribution.topPositive) {
                    hybridSb.append(String.format("  +%.6f %s (x=%.4f w=%.4f)\n",
                        c.contribution, c.name, c.value, c.weight))
                }
                hybridSb.append("Top negative contributions:\n")
                for (c in attribution.topNegative) {
                    hybridSb.append(String.format("  %.6f %s (x=%.4f w=%.4f)\n",
                        c.contribution, c.name, c.value, c.weight))
                }
                hybridSb.append(String.format("Attribution score: %.6f (logit=%.4f)\n",
                    attribution.score, attribution.logit))
            }

            val anomalySb = StringBuilder()
            if (anomalyScore != null) {
                anomalySb.append(String.format("Score: %.6f\n", anomalyScore))
                anomalySb.append(String.format("Threshold: %.3f\n", anomalyMeta.threshold))
                anomalySb.append(String.format("Result: %s\n", if (anomalyMalicious) "MALICIOUS" else "BENIGN"))
                anomalySb.append(String.format("Extract: %.3fms\nInfer: %.3fms\n", aExtractMs, aInferMs))
            } else {
                anomalySb.append("Score: FAIL\n")
            }

            val tagSeqSb = StringBuilder()
            if (tagSeqScore != null) {
                tagSeqSb.append(String.format("Score: %.6f\n", tagSeqScore))
                tagSeqSb.append(String.format("Threshold: %.3f\n", tagSeqMeta.threshold))
                tagSeqSb.append(String.format("Result: %s\n", if (tagSeqMalicious) "MALICIOUS" else "BENIGN"))
                tagSeqSb.append(String.format("Extract: %.3fms\nInfer: %.3fms\n", tExtractMs, tInferMs))
                if (tagSeqInput != null) {
                    tagSeqSb.append(String.format("Seq length: %d\n", tagSeqInput.length))
                    tagSeqSb.append(String.format("Is DNG: %s\n", if (tagSeqInput.isDng) "yes" else "no"))
                }
            } else {
                tagSeqSb.append("Result: N/A (not DNG or parse failed)\n")
            }

            val magikaSb = StringBuilder()
            if (magikaResult != null) {
                magikaSb.append(String.format("DL label: %s\n", magikaResult.dlLabel))
                magikaSb.append(String.format("Output: %s\n", magikaResult.outputLabel))
                magikaSb.append(String.format("Score: %.6f\n", magikaResult.score))
                magikaSb.append(String.format("Infer: %.3fms\n", mInferMs))
            } else {
                magikaSb.append("Result: FAIL\n")
            }

            runOnUiThread {
                if (isMalicious) {
                    updateStatus("Status: MALICIOUS", R.color.status_malicious)
                } else {
                    updateStatus("Status: BENIGN", R.color.status_benign)
                }
                cveResultView.visibility = View.GONE
                val hybridColor = if (hybridMalicious) R.color.status_malicious else R.color.status_benign
                val anomalyColor = if (anomalyMalicious) R.color.status_malicious else R.color.status_benign
                val hybridText = if (score != null) {
                    String.format("Hybrid (thr=%.2f): %s", MODEL_THRESHOLD, if (hybridMalicious) "MAL" else "BEN")
                } else {
                    "Hybrid: FAIL"
                }
                val anomalyText = if (anomalyScore != null) {
                    String.format("Anomaly AE (thr=%.3f): %s", anomalyMeta.threshold,
                        if (anomalyMalicious) "MAL" else "BEN")
                } else {
                    "Anomaly AE: FAIL"
                }
                val magikaText = if (magikaResult != null) {
                    String.format("Magika: %s", magikaResult.outputLabel)
                } else {
                    "Magika: FAIL"
                }
                val tagSeqText = if (tagSeqScore != null) {
                    String.format("TagSeq GRU-AE (thr=%.3f): %s", tagSeqMeta.threshold,
                        if (tagSeqMalicious) "MAL" else "BEN")
                } else {
                    "TagSeq GRU-AE: N/A"
                }
                updateResult(hybridResultView, hybridText, if (score != null) hybridColor else R.color.status_neutral)
                updateResult(anomalyResultView, anomalyText, if (anomalyScore != null) anomalyColor else R.color.status_neutral)
                updateResult(magikaResultView, magikaText, R.color.status_neutral)
                updateResult(tagseqResultView, tagSeqText, R.color.status_neutral)
                summaryView.text = summarySb.toString()
                updateResult(hybridDetailView, hybridSb.toString(), if (score != null) hybridColor else R.color.status_neutral)
                updateResult(anomalyDetailView, anomalySb.toString(), if (anomalyScore != null) anomalyColor else R.color.status_neutral)
                updateResult(magikaDetailView, magikaSb.toString(), R.color.status_neutral)
                updateResult(tagseqDetailView, tagSeqSb.toString(), R.color.status_neutral)
            }
        }.start()
    }

    private fun updateStatus(text: String, colorRes: Int) {
        statusView.text = text
        statusView.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun updateResult(view: TextView, text: String, colorRes: Int) {
        view.text = text
        view.setTextColor(ContextCompat.getColor(this, colorRes))
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
            file.readLines()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .map { normalizeBenchPath(it) }
        } catch (e: Exception) {
            Log.w("HybridDetector", "Failed to read list file ${file.absolutePath}: ${e.message}")
            emptyList()
        }
    }

    private fun normalizeBenchPath(raw: String): String {
        if (raw.startsWith("/")) {
            return raw
        }
        val base = getExternalFilesDir(null)
        return if (base != null) {
            File(base, raw).absolutePath
        } else {
            raw
        }
    }

    private fun hasLocalBenchList(): Boolean {
        val localListFile = File(getExternalFilesDir(null), "bench_list.txt")
        return localListFile.exists()
    }
}
