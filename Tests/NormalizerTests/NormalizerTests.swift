import XCTest
@testable import Tokenizers
@testable import Hub

class NormalizerTests: XCTestCase {

    func testLowercaseNormalizer() {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "françois"),
            ("Ωmega", "ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "häagen-dazs"),
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
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
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
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
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
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
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
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
        ]

        for (arg, expect) in testCases {
            let config = Config([:])
            let normalizer = NFKCNormalizer(config: config)
            XCTAssertEqual(normalizer.normalize(text: arg), expect)
        }

        let config = Config(["type": NormalizerType.NFKC.rawValue])
        XCTAssertNotNil(NormalizerFactory.fromConfig(config: config) as? NFKCNormalizer)
    }
}
