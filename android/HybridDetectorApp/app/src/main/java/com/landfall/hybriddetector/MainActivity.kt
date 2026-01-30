package com.landfall.hybriddetector

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.OpenableColumns
import android.provider.Settings
import android.util.Log
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private enum class ChipState {
        NEUTRAL,
        OK,
        WARN,
        BAD,
    }

    private enum class Verdict {
        BENIGN,
        SUSPICIOUS,
        MALICIOUS,
    }

    private lateinit var statusView: TextView
    private lateinit var fileNameView: TextView
    private lateinit var fileMetaView: TextView

    private lateinit var magikaTypeChip: Chip
    private lateinit var magikaScoreChip: Chip
    private lateinit var magikaDetailView: TextView

    private lateinit var routeChip: Chip
    private lateinit var cveChip: Chip
    private lateinit var dngChip: Chip
    private lateinit var engineChip: Chip
    private lateinit var engineScoreView: TextView
    private lateinit var engineDetailView: TextView

    private lateinit var verdictView: TextView
    private lateinit var verdictDetailView: TextView
    private lateinit var summaryView: TextView

    private lateinit var selectButton: MaterialButton
    private lateinit var runButton: MaterialButton
    private lateinit var sampleButton: MaterialButton

    private lateinit var tiffAeMeta: AnomalyMeta
    private lateinit var tiffAeExtractor: AnomalyFeatureExtractor
    private lateinit var tiffAeRunner: AnomalyModelRunner

    private lateinit var tagSeqMeta: TagSeqMeta
    private lateinit var tagSeqRunner: TagSeqModelRunner

    private lateinit var jpegMeta: JpegMeta
    private lateinit var jpegExtractor: JpegFeatureExtractor
    private lateinit var jpegRunner: JpegModelRunner

    private lateinit var magikaRunner: MagikaModelRunner

    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) handleSelectedUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusView = findViewById(R.id.statusView)
        fileNameView = findViewById(R.id.fileNameView)
        fileMetaView = findViewById(R.id.fileMetaView)

        magikaTypeChip = findViewById(R.id.magikaTypeChip)
        magikaScoreChip = findViewById(R.id.magikaScoreChip)
        magikaDetailView = findViewById(R.id.magikaDetailView)

        routeChip = findViewById(R.id.routeChip)
        cveChip = findViewById(R.id.cveChip)
        dngChip = findViewById(R.id.dngChip)
        engineChip = findViewById(R.id.engineChip)
        engineScoreView = findViewById(R.id.engineScoreView)
        engineDetailView = findViewById(R.id.engineDetailView)

        verdictView = findViewById(R.id.verdictView)
        verdictDetailView = findViewById(R.id.verdictDetailView)
        summaryView = findViewById(R.id.summaryView)

        selectButton = findViewById(R.id.selectButton)
        runButton = findViewById(R.id.runButton)
        sampleButton = findViewById(R.id.sampleButton)

        magikaRunner = MagikaModelRunner.fromAssets(this)

        tiffAeMeta = AnomalyMeta.fromAssets(this)
        tiffAeExtractor = AnomalyFeatureExtractor.fromMeta(tiffAeMeta)
        tiffAeRunner = AnomalyModelRunner.fromAssets(this, tiffAeMeta.threshold)

        tagSeqMeta = TagSeqMeta.fromAssets(this)
        tagSeqRunner = TagSeqModelRunner.fromAssets(this, tagSeqMeta)

        jpegMeta = JpegMeta.fromAssets(this)
        jpegExtractor = JpegFeatureExtractor.fromMeta(jpegMeta)
        jpegRunner = JpegModelRunner.fromAssets(this, jpegMeta.threshold)

        resetUi()

        selectButton.setOnClickListener {
            pickFileLauncher.launch(arrayOf("*/*"))
        }

        sampleButton.setOnClickListener {
            if (hasStoragePermission()) {
                showSamplePicker()
            } else {
                requestStoragePermission()
            }
        }

        runButton.setOnClickListener {
            if (hasStoragePermission() || hasLocalBenchList()) {
                runBenchmark()
            } else {
                requestStoragePermission()
            }
        }

        if (intent.getBooleanExtra("auto", false) && (hasStoragePermission() || hasLocalBenchList())) {
            runBenchmark()
        }
    }

    private fun resetUi() {
        updateStatus("Ready", R.color.status_neutral)
        fileNameView.text = "No file selected"
        fileMetaView.text = "—"

        setChip(magikaTypeChip, "type: —", ChipState.NEUTRAL)
        setChip(magikaScoreChip, "score: —", ChipState.NEUTRAL)
        magikaDetailView.text = "—"

        setChip(routeChip, "route: —", ChipState.NEUTRAL)
        setChip(cveChip, "CVE: —", ChipState.NEUTRAL)
        setChip(dngChip, "DNG: —", ChipState.NEUTRAL)
        setChip(engineChip, "engine: —", ChipState.NEUTRAL)

        engineScoreView.text = "—"
        engineDetailView.text = "—"

        verdictView.text = "—"
        verdictView.setTextColor(ContextCompat.getColor(this, R.color.status_neutral))
        verdictDetailView.text = "—"

        summaryView.text = "—"
    }

    private fun setChip(chip: Chip, text: String, state: ChipState) {
        chip.text = text
        when (state) {
            ChipState.NEUTRAL -> {
                chip.setChipBackgroundColorResource(R.color.chip_neutral_bg)
                chip.setTextColor(ContextCompat.getColor(this, R.color.chip_neutral_text))
            }
            ChipState.OK -> {
                chip.setChipBackgroundColorResource(R.color.status_benign)
                chip.setTextColor(ContextCompat.getColor(this, android.R.color.white))
            }
            ChipState.WARN -> {
                chip.setChipBackgroundColorResource(R.color.status_suspicious)
                chip.setTextColor(ContextCompat.getColor(this, android.R.color.white))
            }
            ChipState.BAD -> {
                chip.setChipBackgroundColorResource(R.color.status_malicious)
                chip.setTextColor(ContextCompat.getColor(this, android.R.color.white))
            }
        }
    }

    private fun verdictLabel(verdict: Verdict): String {
        return when (verdict) {
            Verdict.BENIGN -> "BENIGN"
            Verdict.SUSPICIOUS -> "SUSPICIOUS"
            Verdict.MALICIOUS -> "MALICIOUS"
        }
    }

    private fun verdictColor(verdict: Verdict): Int {
        return when (verdict) {
            Verdict.BENIGN -> R.color.status_benign
            Verdict.SUSPICIOUS -> R.color.status_suspicious
            Verdict.MALICIOUS -> R.color.status_malicious
        }
    }

    private fun verdictChipState(verdict: Verdict): ChipState {
        return when (verdict) {
            Verdict.BENIGN -> ChipState.OK
            Verdict.SUSPICIOUS -> ChipState.WARN
            Verdict.MALICIOUS -> ChipState.BAD
        }
    }

    private fun handleSelectedUri(uri: Uri) {
        resetUi()

        val displayName = queryDisplayName(uri)
        val cached = copyToCache(uri, displayName)
        if (cached == null) {
            updateStatus("Failed to read selection", R.color.status_malicious)
            return
        }

        fileNameView.text = displayName ?: cached.name
        fileMetaView.text = "${cached.length()} bytes • cache: ${cached.absolutePath}"

        scanSingleFile(cached.absolutePath, displayName ?: cached.name)
    }

    private fun scanSingleFile(path: String, displayName: String) {
        Thread {
            runOnUiThread {
                updateStatus("Scanning…", R.color.status_neutral)
                setChip(routeChip, "route: —", ChipState.NEUTRAL)
                setChip(cveChip, "CVE: —", ChipState.NEUTRAL)
                setChip(dngChip, "DNG: —", ChipState.NEUTRAL)
                setChip(engineChip, "engine: —", ChipState.NEUTRAL)
                engineScoreView.text = "—"
                engineDetailView.text = "—"
                verdictView.text = "—"
                verdictView.setTextColor(ContextCompat.getColor(this, R.color.status_neutral))
                verdictDetailView.text = "—"
                summaryView.text = "—"
            }

            val m0 = SystemClock.elapsedRealtimeNanos()
            val magika = magikaRunner.classify(path)
            val m1 = SystemClock.elapsedRealtimeNanos()
            val magikaMs = (m1 - m0) / 1_000_000.0

            val magikaType = magika?.outputLabel?.lowercase() ?: "unknown"
            val expectedType = expectedTypeFromName(displayName)

            runOnUiThread {
                setChip(magikaTypeChip, "type: $magikaType", ChipState.NEUTRAL)
                val scoreText = if (magika != null) String.format("score: %.3f", magika.score) else "score: —"
                setChip(magikaScoreChip, scoreText, ChipState.NEUTRAL)
                magikaDetailView.text = if (magika != null) {
                    "dlLabel=${magika.dlLabel} • outputLabel=${magika.outputLabel} • time=${"%.2f".format(magikaMs)}ms"
                } else {
                    "Magika: FAIL"
                }
            }

            if (expectedType != "unknown" && !isTypeCompatible(expectedType, magikaType)) {
                renderTypeMismatch(path, displayName, expectedType, magikaType, magikaMs)
                return@Thread
            }

            when (magikaType) {
                "jpeg" -> scanJpeg(path, displayName)
                "tiff" -> scanTiff(path, displayName)
                else -> renderUnsupported(path, displayName, magikaType, magikaMs)
            }
        }.start()
    }

    private fun expectedTypeFromName(nameOrPath: String?): String {
        if (nameOrPath.isNullOrBlank()) return "unknown"
        val dot = nameOrPath.lastIndexOf('.')
        if (dot == -1 || dot == nameOrPath.length - 1) return "unknown"
        val ext = nameOrPath.substring(dot + 1).lowercase()
        return when (ext) {
            "jpg", "jpeg" -> "jpeg"
            "tif", "tiff" -> "tiff"
            "dng" -> "dng"
            else -> "unknown"
        }
    }

    private fun isTypeCompatible(expected: String, magika: String): Boolean {
        if (expected == "unknown") return true
        if (magika == "unknown") return false
        val allowed = when (expected) {
            "jpeg" -> setOf("jpeg")
            // DNG is TIFF; accept either label.
            "tiff" -> setOf("tiff", "dng")
            "dng" -> setOf("dng", "tiff")
            else -> emptySet()
        }
        return allowed.contains(magika)
    }

    private fun renderTypeMismatch(
        path: String,
        displayName: String,
        expectedType: String,
        magikaType: String,
        magikaMs: Double,
    ) {
        val verdict = Verdict.SUSPICIOUS
        val verdictText = verdictLabel(verdict)
        runOnUiThread {
            setChip(routeChip, "route: mismatch", verdictChipState(verdict))
            setChip(cveChip, "CVE: n/a", ChipState.NEUTRAL)
            setChip(dngChip, "DNG: n/a", ChipState.NEUTRAL)
            setChip(engineChip, "engine: type check", verdictChipState(verdict))

            engineScoreView.text = "Type mismatch: expected=$expectedType • magika=$magikaType"
            engineDetailView.text = String.format("magika_time=%.2fms", magikaMs)

            verdictView.text = verdictText
            verdictView.setTextColor(ContextCompat.getColor(this, verdictColor(verdict)))
            verdictDetailView.text = "Decision: $verdictText (type_mismatch)"

            summaryView.text = buildString {
                append("Selected: ").append(displayName).append("\n")
                append("Path: ").append(path).append("\n")
                append("\nWorkflow:\n")
                append("1) Expected (extension) → ").append(expectedType).append("\n")
                append("2) Magika → ").append(magikaType).append("\n")
                append(String.format("3) Static rule: mismatch (%.2fms)\n\n", magikaMs))
                append("Decision: ").append(verdictText).append("\n")
            }

            updateStatus("Done", verdictColor(verdict))
        }
    }

    private fun showSamplePicker() {
        val base = File("/sdcard/Download/LandFallDetectorSamples")
        if (!base.exists() || !base.isDirectory) {
            updateStatus("Samples not found under /sdcard/Download", R.color.status_malicious)
            summaryView.text = "Missing folder: /sdcard/Download/LandFallDetectorSamples"
            return
        }

        val categories = base.listFiles()
            ?.filter { it.isDirectory }
            ?.sortedBy { it.name }
            ?: emptyList()

        if (categories.isEmpty()) {
            updateStatus("No sample folders found", R.color.status_malicious)
            summaryView.text = "Empty folder: ${base.absolutePath}"
            return
        }

        val labels = categories.map { it.name }.toTypedArray()
        MaterialAlertDialogBuilder(this)
            .setTitle("Choose a sample folder")
            .setItems(labels) { _, which ->
                showSampleFilePicker(categories[which])
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showSampleFilePicker(dir: File) {
        val files = dir.listFiles()
            ?.filter { it.isFile }
            ?.sortedBy { it.name }
            ?: emptyList()

        if (files.isEmpty()) {
            updateStatus("No files in ${dir.name}", R.color.status_malicious)
            summaryView.text = "Empty folder: ${dir.absolutePath}"
            return
        }

        val labels = files.map { it.name }.toTypedArray()
        MaterialAlertDialogBuilder(this)
            .setTitle("Choose a file (${dir.name})")
            .setItems(labels) { _, which ->
                val file = files[which]
                fileNameView.text = file.name
                fileMetaView.text = "${file.length()} bytes • path: ${file.absolutePath}"
                scanSingleFile(file.absolutePath, file.name)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun scanJpeg(path: String, displayName: String) {
        val j0 = SystemClock.elapsedRealtimeNanos()
        val features = jpegExtractor.extract(path)
        val j1 = SystemClock.elapsedRealtimeNanos()
        val score = if (features != null) jpegRunner.predict(features) else null
        val j2 = SystemClock.elapsedRealtimeNanos()

        val extractMs = (j1 - j0) / 1_000_000.0
        val inferMs = (j2 - j1) / 1_000_000.0

        val verdict = when {
            score == null -> Verdict.SUSPICIOUS
            score >= jpegMeta.threshold -> Verdict.SUSPICIOUS
            else -> Verdict.BENIGN
        }
        val verdictText = verdictLabel(verdict)
        val verdictColor = verdictColor(verdict)

        runOnUiThread {
            setChip(routeChip, "route: jpeg", ChipState.NEUTRAL)
            setChip(cveChip, "CVE: n/a", ChipState.NEUTRAL)
            setChip(dngChip, "DNG: n/a", ChipState.NEUTRAL)
            setChip(engineChip, "engine: JPEG AE", verdictChipState(verdict))

            val scoreLine = if (score != null) {
                String.format("JPEG AE score: %.6f (thr=%.6f)", score, jpegMeta.threshold)
            } else {
                "JPEG AE score: FAIL"
            }
            engineScoreView.text = scoreLine
            engineDetailView.text = String.format("extract=%.2fms • infer=%.2fms", extractMs, inferMs)

            verdictView.text = verdictText
            verdictView.setTextColor(ContextCompat.getColor(this, verdictColor))
            verdictDetailView.text = when {
                score == null -> "Decision: $verdictText (jpeg_ae_failed)"
                verdict == Verdict.SUSPICIOUS -> "Decision: $verdictText (jpeg_anomaly)"
                else -> "Decision: $verdictText (jpeg_ae)"
            }

            summaryView.text = buildString {
                append("Selected: ").append(displayName).append("\n")
                append("Path: ").append(path).append("\n")
                append("\nWorkflow:\n")
                append("1) Magika → jpeg\n")
                append("2) JPEG AE\n\n")
                append(scoreLine).append("\n")
                append(String.format("Timing: extract=%.2fms infer=%.2fms\n", extractMs, inferMs))
                append("Decision: ").append(verdictText).append("\n")
            }

            updateStatus("Done", verdictColor)
        }
    }

    private fun scanTiff(path: String, displayName: String) {
        val tiff = TiffParser.parse(path)
        val expectedType = expectedTypeFromName(displayName)

        val c0 = SystemClock.elapsedRealtimeNanos()
        val cve = CveDetector.evaluate(tiff)
        val c1 = SystemClock.elapsedRealtimeNanos()
        val cveMs = (c1 - c0) / 1_000_000.0

        if (cve.hasCveHits) {
            val verdict = if (cve.hasMaliciousHits) Verdict.MALICIOUS else Verdict.SUSPICIOUS
            val verdictText = verdictLabel(verdict)
            val verdictColor = verdictColor(verdict)
            val ids = cve.hits.joinToString(", ") { it.cveId }

            runOnUiThread {
                setChip(routeChip, "route: tiff", ChipState.NEUTRAL)
                setChip(
                    cveChip,
                    if (verdict == Verdict.MALICIOUS) "static: CVE" else "static: WARN",
                    verdictChipState(verdict)
                )
                setChip(dngChip, "DNG: ${if (tiff.isDng == 1) "yes" else "no"}", ChipState.NEUTRAL)
                setChip(engineChip, "engine: static rules", verdictChipState(verdict))
                engineScoreView.text = "Static match: $ids"
                engineDetailView.text = String.format("scan=%.2fms", cveMs)

                verdictView.text = verdictText
                verdictView.setTextColor(ContextCompat.getColor(this, verdictColor))
                verdictDetailView.text = if (verdict == Verdict.MALICIOUS) {
                    "Decision: $verdictText (cve_rules)"
                } else {
                    "Decision: $verdictText (static_rules)"
                }

                summaryView.text = buildString {
                    append("Selected: ").append(displayName).append("\n")
                    append("Path: ").append(path).append("\n")
                    append("\nWorkflow:\n")
                    append("1) Magika → tiff\n")
                    append(String.format("2) Static rules (%.2fms) → HIT\n\n", cveMs))
                    for (hit in cve.hits) {
                        append("- ").append(hit.cveId).append(": ").append(hit.description).append("\n")
                    }
                    append("\nDecision: ").append(verdictText).append("\n")
                }

                updateStatus("Done", verdictColor)
            }
            return
        }

        val mustUseTagSeq = (expectedType == "dng") || (tiff.isDng == 1)
        if (mustUseTagSeq) {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val tagSeqInput = TagSequenceExtractor.extract(path, tagSeqMeta.maxSeqLen)
            val t1 = SystemClock.elapsedRealtimeNanos()
            val tagSeqScore = if (tagSeqInput != null) tagSeqRunner.predict(tagSeqInput) else null
            val t2 = SystemClock.elapsedRealtimeNanos()

            val tExtractMs = (t1 - t0) / 1_000_000.0
            val tInferMs = (t2 - t1) / 1_000_000.0

            if (tagSeqInput == null || !tagSeqInput.isDng || tagSeqScore == null) {
                val reason = when {
                    tagSeqInput == null -> "TagSeq extract failed"
                    !tagSeqInput.isDng -> "Not DNG per TagSeq"
                    else -> "TagSeq inference failed"
                }
                val verdict = Verdict.SUSPICIOUS
                val verdictText = verdictLabel(verdict)
                runOnUiThread {
                    setChip(routeChip, "route: dng", ChipState.NEUTRAL)
                    setChip(cveChip, "CVE: none", ChipState.OK)
                    setChip(dngChip, "DNG: yes", ChipState.NEUTRAL)
                    setChip(engineChip, "engine: TagSeq required", verdictChipState(verdict))

                    engineScoreView.text = "FAIL (DNG requires TagSeq)"
                    engineDetailView.text = buildString {
                        append(reason)
                        append(String.format(" • extract=%.2fms • infer=%.2fms • cve_scan=%.2fms", tExtractMs, tInferMs, cveMs))
                    }

                    verdictView.text = verdictText
                    verdictView.setTextColor(ContextCompat.getColor(this, verdictColor(verdict)))
                    verdictDetailView.text = "Decision: $verdictText (tagseq_required_failed)"

                    summaryView.text = buildString {
                        append("Selected: ").append(displayName).append("\n")
                        append("Path: ").append(path).append("\n")
                        append("\nWorkflow:\n")
                        append("1) Magika → tiff\n")
                        append(String.format("2) CVE rules (%.2fms) → none\n", cveMs))
                        append("3) DNG route → TagSeq GRU-AE (required)\n\n")
                        append("TagSeq: FAIL (").append(reason).append(")\n")
                        append(String.format("Timing: extract=%.2fms infer=%.2fms\n", tExtractMs, tInferMs))
                        append("\nDecision: ").append(verdictText).append("\n")
                    }

                    updateStatus("Done", verdictColor(verdict))
                }
                return
            }

            val verdict = if (tagSeqScore >= tagSeqMeta.threshold) Verdict.SUSPICIOUS else Verdict.BENIGN
            val verdictText = verdictLabel(verdict)
            val verdictColor = verdictColor(verdict)
            val perElementMse = tagSeqScore / tagSeqMeta.featureDim

            runOnUiThread {
                setChip(routeChip, "route: dng", ChipState.NEUTRAL)
                setChip(cveChip, "CVE: none", ChipState.OK)
                setChip(dngChip, "DNG: yes", ChipState.NEUTRAL)
                setChip(engineChip, "engine: TagSeq GRU-AE", verdictChipState(verdict))

                engineScoreView.text = String.format(
                    "TagSeq score: %.6f (thr=%.6f) • mse/elem: %.6f",
                    tagSeqScore,
                    tagSeqMeta.threshold,
                    perElementMse
                )
                engineDetailView.text = String.format(
                    "seq_len=%d • extract=%.2fms • infer=%.2fms • cve_scan=%.2fms",
                    tagSeqInput.length,
                    tExtractMs,
                    tInferMs,
                    cveMs
                )

                verdictView.text = verdictText
                verdictView.setTextColor(ContextCompat.getColor(this, verdictColor))
                verdictDetailView.text = if (verdict == Verdict.SUSPICIOUS) {
                    "Decision: $verdictText (dng_anomaly)"
                } else {
                    "Decision: $verdictText (tagseq)"
                }

                summaryView.text = buildString {
                    append("Selected: ").append(displayName).append("\n")
                    append("Path: ").append(path).append("\n")
                    append("\nWorkflow:\n")
                    append("1) Magika → tiff\n")
                    append(String.format("2) CVE rules (%.2fms) → none\n", cveMs))
                    append("3) DNG detected → TagSeq GRU-AE\n\n")
                    append(String.format("TagSeq score: %.6f (thr=%.6f)\n", tagSeqScore, tagSeqMeta.threshold))
                    append(String.format("TagSeq mse/elem: %.6f\n", perElementMse))
                append(String.format("Timing: extract=%.2fms infer=%.2fms\n", tExtractMs, tInferMs))
                append("Decision: ").append(verdictText).append("\n")
            }

            updateStatus("Done", verdictColor)
        }
            return
        }

        val a0 = SystemClock.elapsedRealtimeNanos()
        val anomalyFeatures = tiffAeExtractor.extractFromParsed(tiff, path)
        val a1 = SystemClock.elapsedRealtimeNanos()
        val anomalyScore = if (anomalyFeatures != null) tiffAeRunner.predict(anomalyFeatures) else null
        val a2 = SystemClock.elapsedRealtimeNanos()

        val aExtractMs = (a1 - a0) / 1_000_000.0
        val aInferMs = (a2 - a1) / 1_000_000.0

        val verdict = when {
            anomalyScore == null -> Verdict.SUSPICIOUS
            anomalyScore >= tiffAeMeta.threshold -> Verdict.SUSPICIOUS
            else -> Verdict.BENIGN
        }
        val verdictText = verdictLabel(verdict)
        val verdictColor = verdictColor(verdict)

        runOnUiThread {
            setChip(routeChip, "route: tiff", ChipState.NEUTRAL)
            setChip(cveChip, "CVE: none", ChipState.OK)
            setChip(dngChip, "DNG: no", ChipState.NEUTRAL)
            setChip(engineChip, "engine: TIFF AE", verdictChipState(verdict))

            val scoreLine = if (anomalyScore != null) {
                String.format("TIFF AE score: %.6f (thr=%.6f)", anomalyScore, tiffAeMeta.threshold)
            } else {
                "TIFF AE score: FAIL"
            }
            engineScoreView.text = scoreLine
            engineDetailView.text = buildString {
                append(String.format("extract=%.2fms • infer=%.2fms • cve_scan=%.2fms", aExtractMs, aInferMs, cveMs))
            }

            verdictView.text = verdictText
            verdictView.setTextColor(ContextCompat.getColor(this, verdictColor))
            verdictDetailView.text = when {
                anomalyScore == null -> "Decision: $verdictText (tiff_ae_failed)"
                verdict == Verdict.SUSPICIOUS -> "Decision: $verdictText (tiff_anomaly)"
                else -> "Decision: $verdictText (tiff_ae)"
            }

            summaryView.text = buildString {
                append("Selected: ").append(displayName).append("\n")
                append("Path: ").append(path).append("\n")
                append("\nWorkflow:\n")
                append("1) Magika → tiff\n")
                append(String.format("2) CVE rules (%.2fms) → none\n", cveMs))
                append("3) Non-DNG TIFF → TIFF AE\n\n")
                append(scoreLine).append("\n")
                append(String.format("Timing: extract=%.2fms infer=%.2fms\n", aExtractMs, aInferMs))
                append("Decision: ").append(verdictText).append("\n")
            }

            updateStatus("Done", verdictColor)
        }
    }

    private fun renderUnsupported(path: String, displayName: String, type: String, magikaMs: Double) {
        val verdict = Verdict.SUSPICIOUS
        val verdictText = verdictLabel(verdict)
        runOnUiThread {
            setChip(routeChip, "route: unsupported", verdictChipState(verdict))
            setChip(cveChip, "CVE: n/a", ChipState.NEUTRAL)
            setChip(dngChip, "DNG: n/a", ChipState.NEUTRAL)
            setChip(engineChip, "engine: —", ChipState.NEUTRAL)
            engineScoreView.text = "Unsupported type: $type"
            engineDetailView.text = String.format("magika_time=%.2fms", magikaMs)
            verdictView.text = verdictText
            verdictView.setTextColor(ContextCompat.getColor(this, verdictColor(verdict)))
            verdictDetailView.text = "Decision: $verdictText (unsupported_type)"

            summaryView.text = buildString {
                append("Selected: ").append(displayName).append("\n")
                append("Path: ").append(path).append("\n")
                append("\nWorkflow:\n")
                append("1) Magika → ").append(type).append("\n")
                append("2) No detector available\n\n")
                append("Decision: ").append(verdictText).append("\n")
            }

            updateStatus("Done", verdictColor(verdict))
        }
    }

    private fun runBenchmark() {
        val paths = loadPaths()
        if (paths.isEmpty()) {
            summaryView.text = "No paths found. Put a list in /sdcard/bench_list.txt or " +
                "/sdcard/Android/data/com.landfall.hybriddetector/files/bench_list.txt."
            return
        }

        resetUi()
        updateStatus("Benchmark running…", R.color.status_neutral)
        verdictView.text = "BENCHMARK"
        verdictDetailView.text = "Running ${paths.size} files…"

        Thread {
            Log.i("HybridDetector", "Running benchmark on ${paths.size} files")

            var totalMagikaNs = 0L
            var totalCveNs = 0L

            var totalTagSeqExtractNs = 0L
            var totalTagSeqInferNs = 0L
            var tagSeqCount = 0

            var totalTiffAeExtractNs = 0L
            var totalTiffAeInferNs = 0L
            var tiffAeCount = 0

            var totalJpegExtractNs = 0L
            var totalJpegInferNs = 0L
            var jpegCount = 0

            var cveHitCount = 0
            var staticWarnCount = 0
            var unsupportedCount = 0
            var typeMismatchCount = 0
            var anomalyCount = 0
            var errorCount = 0

            var verdictMalCount = 0
            var verdictSusCount = 0
            var verdictBenCount = 0

            for (path in paths) {
                val m0 = SystemClock.elapsedRealtimeNanos()
                val magika = magikaRunner.classify(path)
                val m1 = SystemClock.elapsedRealtimeNanos()
                totalMagikaNs += (m1 - m0)
                val type = magika?.outputLabel?.lowercase() ?: "unknown"
                val expected = expectedTypeFromName(path)

                if (expected != "unknown" && !isTypeCompatible(expected, type)) {
                    typeMismatchCount += 1
                    verdictSusCount += 1
                    Log.i("HybridDetector", "type=$type | decision=SUSP(type_mismatch) | expected=$expected | $path")
                    continue
                }

                if (type == "jpeg") {
                    val j0 = SystemClock.elapsedRealtimeNanos()
                    val jpegFeatures = jpegExtractor.extract(path)
                    val j1 = SystemClock.elapsedRealtimeNanos()
                    val jpegScore = if (jpegFeatures != null) jpegRunner.predict(jpegFeatures) else null
                    val j2 = SystemClock.elapsedRealtimeNanos()

                    val jExtractMs = (j1 - j0) / 1_000_000.0
                    val jInferMs = (j2 - j1) / 1_000_000.0

                    if (jpegScore != null) {
                        totalJpegExtractNs += (j1 - j0)
                        totalJpegInferNs += (j2 - j1)
                        jpegCount += 1
                    }

                    val decision = when {
                        jpegScore == null -> {
                            errorCount += 1
                            verdictSusCount += 1
                            "SUSP(jpeg_ae_failed)"
                        }
                        jpegScore >= jpegMeta.threshold -> {
                            anomalyCount += 1
                            verdictSusCount += 1
                            "SUSP(jpeg_anomaly)"
                        }
                        else -> {
                            verdictBenCount += 1
                            "BEN(jpeg_ae)"
                        }
                    }
                    val part = if (jpegScore != null) {
                        String.format("score=%.6f thr=%.6f ex=%.2fms in=%.2fms",
                            jpegScore, jpegMeta.threshold, jExtractMs, jInferMs)
                    } else {
                        "score=FAIL"
                    }
                    Log.i("HybridDetector", "type=jpeg | decision=$decision | $part | $path")
                    continue
                }

                if (type != "tiff") {
                    unsupportedCount += 1
                    verdictSusCount += 1
                    Log.i("HybridDetector", "type=$type | decision=SUSP(unsupported_type) | $path")
                    continue
                }

                val tiff = TiffParser.parse(path)
                val c0 = SystemClock.elapsedRealtimeNanos()
                val cve = CveDetector.evaluate(tiff)
                val c1 = SystemClock.elapsedRealtimeNanos()
                totalCveNs += (c1 - c0)

                if (cve.hasCveHits) {
                    val ids = cve.hits.joinToString(",") { it.cveId }
                    if (cve.hasMaliciousHits) {
                        cveHitCount += 1
                        verdictMalCount += 1
                        Log.i("HybridDetector", "type=tiff | decision=MAL(cve_rules) | ids=$ids | $path")
                    } else {
                        staticWarnCount += 1
                        verdictSusCount += 1
                        Log.i("HybridDetector", "type=tiff | decision=SUSP(static_rules) | ids=$ids | $path")
                    }
                    continue
                }

                val mustUseTagSeq = (expected == "dng") || (tiff.isDng == 1)
                if (mustUseTagSeq) {
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    val tagSeqInput = TagSequenceExtractor.extract(path, tagSeqMeta.maxSeqLen)
                    val t1 = SystemClock.elapsedRealtimeNanos()
                    val tagSeqScore = if (tagSeqInput != null) tagSeqRunner.predict(tagSeqInput) else null
                    val t2 = SystemClock.elapsedRealtimeNanos()

                    val tExtractMs = (t1 - t0) / 1_000_000.0
                    val tInferMs = (t2 - t1) / 1_000_000.0

                    if (tagSeqScore != null) {
                        totalTagSeqExtractNs += (t1 - t0)
                        totalTagSeqInferNs += (t2 - t1)
                        tagSeqCount += 1
                    }

                    if (tagSeqInput == null || !tagSeqInput.isDng || tagSeqScore == null) {
                        errorCount += 1
                        verdictSusCount += 1
                        val reason = when {
                            tagSeqInput == null -> "extract_fail"
                            !tagSeqInput.isDng -> "not_dng"
                            else -> "infer_fail"
                        }
                        Log.i("HybridDetector", "type=tiff | decision=SUSP(tagseq_required_failed) | reason=$reason | ex=%.2fms in=%.2fms | %s"
                            .format(tExtractMs, tInferMs, path))
                        continue
                    }

                    val decision = if (tagSeqScore >= tagSeqMeta.threshold) {
                        anomalyCount += 1
                        verdictSusCount += 1
                        "SUSP(dng_anomaly)"
                    } else {
                        verdictBenCount += 1
                        "BEN(tagseq)"
                    }
                    Log.i("HybridDetector", "type=tiff | decision=$decision | score=%.6f thr=%.6f ex=%.2fms in=%.2fms | %s"
                        .format(tagSeqScore, tagSeqMeta.threshold, tExtractMs, tInferMs, path))
                    continue
                }

                val a0 = SystemClock.elapsedRealtimeNanos()
                val anomalyFeatures = tiffAeExtractor.extractFromParsed(tiff, path)
                val a1 = SystemClock.elapsedRealtimeNanos()
                val anomalyScore = if (anomalyFeatures != null) tiffAeRunner.predict(anomalyFeatures) else null
                val a2 = SystemClock.elapsedRealtimeNanos()

                val aExtractMs = (a1 - a0) / 1_000_000.0
                val aInferMs = (a2 - a1) / 1_000_000.0

                if (anomalyScore != null) {
                    totalTiffAeExtractNs += (a1 - a0)
                    totalTiffAeInferNs += (a2 - a1)
                    tiffAeCount += 1
                }

                val decision = when {
                    anomalyScore == null -> {
                        errorCount += 1
                        verdictSusCount += 1
                        "SUSP(tiff_ae_failed)"
                    }
                    anomalyScore >= tiffAeMeta.threshold -> {
                        anomalyCount += 1
                        verdictSusCount += 1
                        "SUSP(tiff_anomaly)"
                    }
                    else -> {
                        verdictBenCount += 1
                        "BEN(tiff_ae)"
                    }
                }
                Log.i("HybridDetector", "type=tiff | decision=$decision | score=%s thr=%.6f ex=%.2fms in=%.2fms | %s"
                    .format(anomalyScore?.let { "%.6f".format(it) } ?: "FAIL", tiffAeMeta.threshold, aExtractMs, aInferMs, path))
            }

            val avgMagikaMs = totalMagikaNs / 1_000_000.0 / paths.size
            val avgCveMs = totalCveNs / 1_000_000.0 / paths.size
            val avgTagSeqExtractMs = if (tagSeqCount > 0) totalTagSeqExtractNs / 1_000_000.0 / tagSeqCount else 0.0
            val avgTagSeqInferMs = if (tagSeqCount > 0) totalTagSeqInferNs / 1_000_000.0 / tagSeqCount else 0.0
            val avgTiffAeExtractMs = if (tiffAeCount > 0) totalTiffAeExtractNs / 1_000_000.0 / tiffAeCount else 0.0
            val avgTiffAeInferMs = if (tiffAeCount > 0) totalTiffAeInferNs / 1_000_000.0 / tiffAeCount else 0.0
            val avgJpegExtractMs = if (jpegCount > 0) totalJpegExtractNs / 1_000_000.0 / jpegCount else 0.0
            val avgJpegInferMs = if (jpegCount > 0) totalJpegInferNs / 1_000_000.0 / jpegCount else 0.0

            val benchSummary = buildString {
                append("Files: ").append(paths.size).append("\n\n")
                append("Thresholds:\n")
                append(String.format("- TIFF AE: %.6f\n", tiffAeMeta.threshold))
                append(String.format("- TagSeq: %.6f\n", tagSeqMeta.threshold))
                append(String.format("- JPEG AE: %.6f\n\n", jpegMeta.threshold))
                append("Results:\n")
                append(String.format("- BENIGN: %d\n", verdictBenCount))
                append(String.format("- SUSPICIOUS: %d\n", verdictSusCount))
                append(String.format("- MALICIOUS: %d\n\n", verdictMalCount))
                append("Suspicious breakdown:\n")
                append(String.format("- static warnings: %d\n", staticWarnCount))
                append(String.format("- anomalies: %d\n", anomalyCount))
                append(String.format("- type mismatch: %d\n", typeMismatchCount))
                append(String.format("- unsupported type: %d\n", unsupportedCount))
                append(String.format("- errors: %d\n\n", errorCount))
                append("Malicious breakdown:\n")
                append(String.format("- CVE hits: %d\n\n", cveHitCount))
                append("Average timings:\n")
                append(String.format("- Magika: %.3fms/file\n", avgMagikaMs))
                append(String.format("- CVE rules: %.3fms/file\n", avgCveMs))
                append(String.format("- TagSeq: extract=%.3fms infer=%.3fms (n=%d)\n", avgTagSeqExtractMs, avgTagSeqInferMs, tagSeqCount))
                append(String.format("- TIFF AE: extract=%.3fms infer=%.3fms (n=%d)\n", avgTiffAeExtractMs, avgTiffAeInferMs, tiffAeCount))
                append(String.format("- JPEG AE: extract=%.3fms infer=%.3fms (n=%d)\n", avgJpegExtractMs, avgJpegInferMs, jpegCount))
            }

            runOnUiThread {
                verdictView.text = "BENCHMARK COMPLETE"
                verdictView.setTextColor(ContextCompat.getColor(this, R.color.status_neutral))
                verdictDetailView.text = "See logcat (tag=HybridDetector) for per-file decisions"
                summaryView.text = benchSummary
                updateStatus("Benchmark done", R.color.status_neutral)
            }
        }.start()
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
