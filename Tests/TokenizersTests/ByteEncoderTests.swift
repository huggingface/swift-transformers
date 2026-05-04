//
//  ByteEncoderTests.swift
//
//  Defensive coverage for the byte-level encoder data tables that the BPE
//  hot path indexes into. The lookup is array-indexed (no fallback), so a
//  silently dropped entry would surface as an empty string in tokenizer
//  output rather than a crash; these tests pin the invariants.
//

import Foundation
import Testing

@testable import Tokenizers

@Suite("Byte encoder tables")
struct ByteEncoderTests {
    /// `byteEncoderTable` is indexed directly by every input byte on the BPE
    /// pre-tokenize hot path, so the array must be exactly 256 long and every
    /// entry must be a non-empty mapping. Anything missing would silently emit
    /// `""` for that byte during encode.
    @Test("byteEncoderTable is dense over 0..<256")
    func byteEncoderTableIsDense() {
        #expect(byteEncoderTable.count == 256)
        for byte in 0..<256 {
            #expect(
                !byteEncoderTable[byte].isEmpty,
                "byteEncoderTable[\(byte)] is empty — the canonical GPT-2 byte-level alphabet covers the full 0..<256 range"
            )
        }
    }

    /// The fast-path `byteEncoderTable` must agree with the canonical
    /// `byteEncoder` dictionary for every byte. This guards against drift if
    /// either definition is edited in isolation.
    @Test("byteEncoderTable agrees with byteEncoder dictionary")
    func byteEncoderTableMatchesDictionary() {
        #expect(byteEncoder.count == 256)
        for (byte, expected) in byteEncoder {
            #expect(byteEncoderTable[Int(byte)] == expected)
        }
    }

    /// `byteEncoder` and `byteDecoder` must be exact inverses; a regression on
    /// either side breaks BPE round-trips. The asserts below would already
    /// surface as decode errors in real-world tokenization, but the explicit
    /// check makes the failure mode obvious.
    @Test("byteEncoder and byteDecoder are inverse mappings")
    func byteEncoderDecoderRoundTrip() {
        for (byte, str) in byteEncoder {
            #expect(byteDecoder[str] == byte)
        }
        for (str, byte) in byteDecoder {
            #expect(byteEncoder[byte] == str)
        }
    }
}
