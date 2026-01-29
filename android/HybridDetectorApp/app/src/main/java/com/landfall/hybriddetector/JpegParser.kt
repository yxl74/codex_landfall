package com.landfall.hybriddetector

import java.io.File
import java.io.RandomAccessFile
import kotlin.math.max
import kotlin.math.min

data class JpegFeatures(
    val fileSize: Int,
    val hasEoi: Int,
    val bytesAfterEoi: Int,
    val bytesAfterEoiRatioPermille: Int,
    val tailZipMagic: Int,
    val tailPdfMagic: Int,
    val tailElfMagic: Int,
    val invalidLen: Int,
    val width: Int,
    val height: Int,
    val components: Int,
    val precision: Int,
    val driInterval: Int,
    val appCount: Int,
    val appBytes: Int,
    val appBytesRatioPermille: Int,
    val appMaxLen: Int,
    val comCount: Int,
    val comBytes: Int,
    val dqtCount: Int,
    val dqtBytes: Int,
    val dqtBytesRatioPermille: Int,
    val dqtMaxLen: Int,
    val dhtCount: Int,
    val dhtBytes: Int,
    val dhtBytesRatioPermille: Int,
    val dhtMaxLen: Int,
    val sof0Count: Int,
    val sof2Count: Int,
    val sosCount: Int,
    val maxSegLen: Int,
    val dqtTables: Int,
    val dqtInvalid: Int,
    val dhtTables: Int,
    val dhtInvalid: Int,
) {
    fun toFeatureMap(): Map<String, Int> {
        return mapOf(
            "file_size" to fileSize,
            "has_eoi" to hasEoi,
            "bytes_after_eoi" to bytesAfterEoi,
            "bytes_after_eoi_ratio_permille" to bytesAfterEoiRatioPermille,
            "tail_zip_magic" to tailZipMagic,
            "tail_pdf_magic" to tailPdfMagic,
            "tail_elf_magic" to tailElfMagic,
            "invalid_len" to invalidLen,
            "width" to width,
            "height" to height,
            "components" to components,
            "precision" to precision,
            "dri_interval" to driInterval,
            "app_count" to appCount,
            "app_bytes" to appBytes,
            "app_bytes_ratio_permille" to appBytesRatioPermille,
            "app_max_len" to appMaxLen,
            "com_count" to comCount,
            "com_bytes" to comBytes,
            "dqt_count" to dqtCount,
            "dqt_bytes" to dqtBytes,
            "dqt_bytes_ratio_permille" to dqtBytesRatioPermille,
            "dqt_max_len" to dqtMaxLen,
            "dht_count" to dhtCount,
            "dht_bytes" to dhtBytes,
            "dht_bytes_ratio_permille" to dhtBytesRatioPermille,
            "dht_max_len" to dhtMaxLen,
            "sof0_count" to sof0Count,
            "sof2_count" to sof2Count,
            "sos_count" to sosCount,
            "max_seg_len" to maxSegLen,
            "dqt_tables" to dqtTables,
            "dqt_invalid" to dqtInvalid,
            "dht_tables" to dhtTables,
            "dht_invalid" to dhtInvalid,
        )
    }
}

object JpegParser {
    private const val SOI0 = 0xFF
    private const val SOI1 = 0xD8
    private const val EOI0 = 0xFF
    private const val EOI1 = 0xD9

    private val SOF_MARKERS = setOf(
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    )

    fun parse(path: String): JpegFeatures? {
        val file = File(path)
        if (!file.exists() || !file.isFile) return null
        val sizeLong = file.length()
        if (sizeLong < 4L) return null
        val size = clampToInt(sizeLong)

        RandomAccessFile(file, "r").use { raf ->
            val head = ByteArray(2)
            raf.seek(0L)
            raf.readFully(head)
            if ((head[0].toInt() and 0xFF) != SOI0 || (head[1].toInt() and 0xFF) != SOI1) {
                return null
            }

            // Tail magics
            val tailSize = min(4096, size)
            val tail = ByteArray(tailSize)
            raf.seek(sizeLong - tailSize.toLong())
            raf.readFully(tail)
            val tailZipMagic = if (indexOf(tail, byteArrayOf(0x50, 0x4B, 0x03, 0x04)) != -1 ||
                indexOf(tail, byteArrayOf(0x50, 0x4B, 0x05, 0x06)) != -1
            ) 1 else 0
            val tailPdfMagic = if (indexOf(tail, "%PDF".toByteArray()) != -1) 1 else 0
            val tailElfMagic = if (indexOf(tail, byteArrayOf(0x7F, 0x45, 0x4C, 0x46)) != -1) 1 else 0

            // Find last EOI within a tail window. If absent, treat as no EOI.
            val eoiWindow = 1024 * 1024
            val eoiStart = max(0, (sizeLong - eoiWindow).toInt())
            val eoiTailSize = (sizeLong - eoiStart).toInt()
            val eoiTail = ByteArray(eoiTailSize)
            raf.seek(eoiStart.toLong())
            raf.readFully(eoiTail)
            val eoiRel = lastIndexOf(eoiTail, byteArrayOf(EOI0.toByte(), EOI1.toByte()))
            val hasEoi = if (eoiRel != -1) 1 else 0
            val bytesAfterEoi = if (hasEoi == 1) {
                val eoiAbs = eoiStart.toLong() + eoiRel.toLong()
                clampToInt(sizeLong - (eoiAbs + 2L))
            } else {
                clampToInt(sizeLong)
            }
            val bytesAfterEoiRatioPermille = if (sizeLong > 0L) {
                clampToInt(bytesAfterEoi.toLong() * 1000L / sizeLong)
            } else {
                0
            }

            // Parse segments up to SOS
            var invalidLen = 0
            var width = 0
            var height = 0
            var components = 0
            var precision = 0
            var driInterval = 0

            var appCount = 0
            var appBytes = 0
            var appMaxLen = 0
            var comCount = 0
            var comBytes = 0
            var dqtCount = 0
            var dqtBytes = 0
            var dqtMaxLen = 0
            var dhtCount = 0
            var dhtBytes = 0
            var dhtMaxLen = 0
            var sof0Count = 0
            var sof2Count = 0
            var sosCount = 0
            var maxSegLen = 0
            var dqtTables = 0
            var dqtInvalid = 0
            var dhtTables = 0
            var dhtInvalid = 0

            var offset = 2L
            while (offset < sizeLong - 1L) {
                raf.seek(offset)
                val b = raf.read()
                if (b != 0xFF) {
                    offset += 1L
                    continue
                }

                // Skip 0xFF fill bytes
                var marker = raf.read()
                while (marker == 0xFF) {
                    marker = raf.read()
                }
                if (marker < 0) break
                offset = raf.filePointer

                if (marker == 0x00) {
                    continue
                }
                if (marker == 0xD9) {
                    break
                }
                if (marker in 0xD0..0xD7) {
                    continue
                }

                if (marker in 0xE0..0xEF) appCount += 1
                if (marker == 0xFE) comCount += 1
                if (marker == 0xDB) dqtCount += 1
                if (marker == 0xC4) dhtCount += 1
                if (marker == 0xC0) sof0Count += 1
                if (marker == 0xC2) sof2Count += 1
                if (marker == 0xDA) sosCount += 1

                // Markers without a length field.
                if (marker == 0xD8 || marker == 0x01) {
                    continue
                }

                val lenHi = raf.read()
                val lenLo = raf.read()
                if (lenHi < 0 || lenLo < 0) {
                    invalidLen = 1
                    break
                }
                val segLen = (lenHi shl 8) or lenLo
                if (segLen < 2) {
                    invalidLen = 1
                    break
                }
                if (segLen > maxSegLen) maxSegLen = segLen

                val payloadLen = segLen - 2
                val payloadOff = raf.filePointer
                val segEnd = payloadOff + payloadLen.toLong()
                if (segEnd > sizeLong) {
                    invalidLen = 1
                    break
                }

                // Track bytes/max for key segments (includes the 2 length bytes, like Python extractor)
                if (marker in 0xE0..0xEF) {
                    appBytes += segLen
                    if (segLen > appMaxLen) appMaxLen = segLen
                }
                if (marker == 0xFE) {
                    comBytes += segLen
                }
                if (marker == 0xDB) {
                    dqtBytes += segLen
                    if (segLen > dqtMaxLen) dqtMaxLen = segLen
                }
                if (marker == 0xC4) {
                    dhtBytes += segLen
                    if (segLen > dhtMaxLen) dhtMaxLen = segLen
                }

                // Parse payload for selected markers
                if (payloadLen > 0 && (marker == 0xDB || marker == 0xC4 || marker == 0xDD || marker in SOF_MARKERS)) {
                    val payload = ByteArray(payloadLen)
                    raf.readFully(payload)

                    if (marker in SOF_MARKERS && width == 0 && height == 0 && payloadLen >= 6) {
                        precision = payload[0].toInt() and 0xFF
                        height = u16be(payload, 1)
                        width = u16be(payload, 3)
                        components = payload[5].toInt() and 0xFF
                    }

                    if (marker == 0xDD && payloadLen >= 2) {
                        driInterval = u16be(payload, 0)
                    }

                    if (marker == 0xDB) {
                        var p = 0
                        while (p < payload.size) {
                            if (p >= payload.size) break
                            val pqTq = payload[p].toInt() and 0xFF
                            p += 1
                            val pq = (pqTq shr 4) and 0x0F
                            val tableBytes = 64 * (if (pq == 0) 1 else 2)
                            if (p + tableBytes > payload.size) {
                                dqtInvalid += 1
                                break
                            }
                            dqtTables += 1
                            p += tableBytes
                        }
                    }

                    if (marker == 0xC4) {
                        var p = 0
                        while (p < payload.size) {
                            if (p + 1 + 16 > payload.size) {
                                dhtInvalid += 1
                                break
                            }
                            p += 1 // tc/th
                            var totalSyms = 0
                            for (i in 0 until 16) {
                                totalSyms += payload[p + i].toInt() and 0xFF
                            }
                            p += 16
                            if (p + totalSyms > payload.size) {
                                dhtInvalid += 1
                                break
                            }
                            dhtTables += 1
                            p += totalSyms
                        }
                    }
                } else {
                    raf.seek(segEnd)
                }

                offset = segEnd
                if (marker == 0xDA) {
                    break
                }
            }

            val appBytesRatioPermille = if (sizeLong > 0L) clampToInt(appBytes.toLong() * 1000L / sizeLong) else 0
            val dqtBytesRatioPermille = if (sizeLong > 0L) clampToInt(dqtBytes.toLong() * 1000L / sizeLong) else 0
            val dhtBytesRatioPermille = if (sizeLong > 0L) clampToInt(dhtBytes.toLong() * 1000L / sizeLong) else 0

            return JpegFeatures(
                fileSize = size,
                hasEoi = hasEoi,
                bytesAfterEoi = bytesAfterEoi,
                bytesAfterEoiRatioPermille = bytesAfterEoiRatioPermille,
                tailZipMagic = tailZipMagic,
                tailPdfMagic = tailPdfMagic,
                tailElfMagic = tailElfMagic,
                invalidLen = invalidLen,
                width = width,
                height = height,
                components = components,
                precision = precision,
                driInterval = driInterval,
                appCount = appCount,
                appBytes = appBytes,
                appBytesRatioPermille = appBytesRatioPermille,
                appMaxLen = appMaxLen,
                comCount = comCount,
                comBytes = comBytes,
                dqtCount = dqtCount,
                dqtBytes = dqtBytes,
                dqtBytesRatioPermille = dqtBytesRatioPermille,
                dqtMaxLen = dqtMaxLen,
                dhtCount = dhtCount,
                dhtBytes = dhtBytes,
                dhtBytesRatioPermille = dhtBytesRatioPermille,
                dhtMaxLen = dhtMaxLen,
                sof0Count = sof0Count,
                sof2Count = sof2Count,
                sosCount = sosCount,
                maxSegLen = maxSegLen,
                dqtTables = dqtTables,
                dqtInvalid = dqtInvalid,
                dhtTables = dhtTables,
                dhtInvalid = dhtInvalid,
            )
        }
    }

    private fun clampToInt(value: Long): Int {
        return when {
            value > Int.MAX_VALUE -> Int.MAX_VALUE
            value < Int.MIN_VALUE -> Int.MIN_VALUE
            else -> value.toInt()
        }
    }

    private fun u16be(buf: ByteArray, off: Int): Int {
        return ((buf[off].toInt() and 0xFF) shl 8) or (buf[off + 1].toInt() and 0xFF)
    }

    private fun indexOf(haystack: ByteArray, needle: ByteArray, startFrom: Int = 0): Int {
        if (needle.isEmpty() || haystack.size < needle.size) return -1
        outer@ for (i in startFrom..haystack.size - needle.size) {
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }

    private fun lastIndexOf(haystack: ByteArray, needle: ByteArray): Int {
        if (needle.isEmpty() || haystack.size < needle.size) return -1
        for (i in haystack.size - needle.size downTo 0) {
            var match = true
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) {
                    match = false
                    break
                }
            }
            if (match) return i
        }
        return -1
    }
}

