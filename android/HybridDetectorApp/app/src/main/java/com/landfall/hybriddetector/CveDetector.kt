package com.landfall.hybriddetector

import android.os.SystemClock

data class CveHit(val cveId: String, val description: String)

data class CveResult(val hits: List<CveHit>, val scanTimeMs: Float) {
    val hasCveHits get() = hits.isNotEmpty()
}

/**
 * Static CVE rule evaluator. No ML dependencies.
 *
 * Evaluates known CVE patterns against pre-parsed TiffFeatures.
 * This runs as Tier 1 before ML models — if any rule triggers,
 * the file is flagged without needing ML inference.
 */
object CveDetector {
    private const val EXTREME_DIM_THRESHOLD = 0xFFFE7960L

    fun evaluate(features: TiffFeatures): CveResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        val hits = mutableListOf<CveHit>()

        // Rule 1: CVE-2025-21043 — Declared opcode list count > 1,000,000
        if (features.maxDeclaredOpcodeCount > 1_000_000) {
            hits.add(CveHit(
                cveId = "CVE-2025-21043",
                description = "Declared opcode count ${features.maxDeclaredOpcodeCount} exceeds 1M limit"
            ))
        }

        // Rule 2: CVE-2025-43300 — SOF3 component count mismatch
        // SubIFD has SPP=2, Compression=7 (JPEG Lossless), AND embedded JPEG SOF3
        // marker has component_count != SPP
        if (features.sof3ComponentMismatch) {
            hits.add(CveHit(
                cveId = "CVE-2025-43300",
                description = "JPEG SOF3 component count mismatch (SPP=2, Compression=7)"
            ))
        }

        // Rule 3: TILE-CONFIG — Tile configuration anomalies
        // 3a: tile_offsets_count != tile_byte_counts_count
        if (features.tileOffsetsCount > 0 || features.tileByteCountsCount > 0) {
            if (features.tileOffsetsCount != features.tileByteCountsCount) {
                hits.add(CveHit(
                    cveId = "TILE-CONFIG",
                    description = "Tile offsets count (${features.tileOffsetsCount}) != " +
                        "byte counts count (${features.tileByteCountsCount})"
                ))
            }
        }

        // 3b: actual tile count != expected from geometry
        if (features.expectedTileCount > 0 && features.tileOffsetsCount > 0) {
            if (features.tileOffsetsCount != features.expectedTileCount) {
                hits.add(CveHit(
                    cveId = "TILE-CONFIG",
                    description = "Tile count (${features.tileOffsetsCount}) != " +
                        "expected from geometry (${features.expectedTileCount})"
                ))
            }
        }

        // 3c: extreme dimensions
        val allDims = mutableListOf<Long>()
        allDims.add(features.maxWidth.toLong())
        allDims.add(features.maxHeight.toLong())
        for (tw in features.tileWidths) allDims.add(tw.toLong())
        for (th in features.tileHeights) allDims.add(th.toLong())
        val maxDim = allDims.maxOrNull() ?: 0L
        if (maxDim > EXTREME_DIM_THRESHOLD) {
            hits.add(CveHit(
                cveId = "TILE-DIM",
                description = "Extreme dimension $maxDim exceeds threshold"
            ))
        }

        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0f
        return CveResult(hits = hits, scanTimeMs = elapsedMs)
    }
}
