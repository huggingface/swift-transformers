//
//  BOMDoubling.swift
//  swift-transformers
//
//  Created by Pedro Cuenca on 20250912
//

import Foundation

extension Data {
    /// Workaround for https://github.com/huggingface/swift-transformers/issues/116
    /// Duplicate a BOM sequence that follows a quote. The first BOM is swallowed by JSONSerialization.jsonObject
    /// because it thinks it marks the encoding.
    var duplicatingBOMsAfterQuotes: Data {
        withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let src = raw.bindMemory(to: UInt8.self)
            var out = [UInt8]()
            // We expect very few matches (only 6 for Gemma)
            out.reserveCapacity(src.count + 1000)

            var i = 0
            while i < src.count {
                let b = src[i]
                out.append(b)

                // Check for \u{feff} BOM (observed in Gemma tokenizers), which is encoded as 0xef 0xbb 0xbf.
                // We may need more combinations.
                if b == 0x22, i + 3 < src.count,
                   src[i + 1] == 0xEF, src[i + 2] == 0xBB, src[i + 3] == 0xBF
                {
                    // Duplicate BOM
                    out.append(0xEF); out.append(0xBB); out.append(0xBF)
                    out.append(0xEF); out.append(0xBB); out.append(0xBF)
                    i += 4
                } else {
                    i += 1
                }
            }
            return Data(out)
        }
    }
}

extension JSONSerialization {
    class func bomPreservingJsonObject(with data: Data, options: JSONSerialization.ReadingOptions = []) throws -> Any {
        try JSONSerialization.jsonObject(with: data.duplicatingBOMsAfterQuotes, options: options)
    }
}
