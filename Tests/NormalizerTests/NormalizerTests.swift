import XCTest
@testable import Tokenizers
@testable import Hub

class NormalizerTests: XCTestCase {

    func testLowercaseNormalizer() {
        let testCases: [(String, String)] = [
            ("Café", "café"),
            ("François", "françois"),
            ("Ωmega", "ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "häagen-dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00E5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = LowercaseNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }
        
        let config = Config(["type": NormalizerType.Lowercase.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? LowercaseNormalizer)
    }

    func testNFDNormalizer() {
        let testCases: [(String, String)] = [
            ("caf\u{65}\u{301}", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{0041}\u{030A}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = NFDNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.NFD.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? NFDNormalizer)
    }

    func testNFCNormalizer() {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = NFCNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }
        
        let config = Config(["type": NormalizerType.NFC.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? NFCNormalizer)
    }
    
    func testNFKDNormalizer() {
        let testCases: [(String, String)] = [
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
            ("\u{00C5}", "Å"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = NFKDNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.NFKD.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? NFKDNormalizer)
    }

    func testNFKCINormalizer() {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = NFKCNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.NFKC.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? NFKCNormalizer)
    }
    
    func testBertNormalizer() {
        let testCases: [(String, String)] = [
            ("Café", "café"),
            ("François", "françois"),
            ("Ωmega", "ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen\tDazs", "häagen dazs"),
            ("你好!", " 你  好 !"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00E5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = BertNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.Bert.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? BertNormalizer)
    }
    func testPrecompiledNormalizer() {
        let testCases: [(String, String)] = [
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = PrecompiledNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.Precompiled.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? PrecompiledNormalizer)
    }

    func testStripAccentsINormalizer() {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = StripAccentsNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.StripAccents.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? StripAccentsNormalizer)
    }
}
