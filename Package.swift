// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
        .executable(name: "transformers", targets: ["TransformersCLI"]),
        .executable(name: "hub-cli", targets: ["HubCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/apple/swift-collections.git", .upToNextMinor(from: "1.1.4")),
        .package(url: "https://github.com/johnmai-dev/Jinja", .upToNextMinor(from: "1.1.0")),
    ],
    targets: [
        .executableTarget(
            name: "TransformersCLI",
            dependencies: [
                "Models", "Generation", "Tokenizers",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        ),
        .executableTarget(
            name: "HubCLI",
            dependencies: ["Hub", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .target(
            name: "Hub",
            dependencies: [.product(name: "OrderedCollections", package: "swift-collections")],
            resources: [.process("FallbackConfigs")],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .unsafeFlags(["-strict-concurrency=complete"]),
            ]
        ),
        .target(
            name: "Tokenizers",
            dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .target(
            name: "TensorUtils",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .target(
            name: "Generation",
            dependencies: ["Tokenizers", "TensorUtils"],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .target(
            name: "Models",
            dependencies: ["Tokenizers", "Generation", "TensorUtils"],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .testTarget(
            name: "TokenizersTests",
            dependencies: ["Tokenizers", "Models", "Hub"],
            resources: [.process("Resources"), .process("Vocabs")],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        ),
        .testTarget(
            name: "HubTests",
            dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .unsafeFlags(["-strict-concurrency=complete"]),
            ]
        ),
        .testTarget(
            name: "PreTokenizerTests",
            dependencies: ["Tokenizers", "Hub"],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .testTarget(
            name: "TensorUtilsTests",
            dependencies: ["TensorUtils", "Models", "Hub"],
            resources: [.process("Resources")],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .testTarget(
            name: "NormalizerTests",
            dependencies: ["Tokenizers", "Hub"],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
        .testTarget(
            name: "PostProcessorTests",
            dependencies: ["Tokenizers", "Hub"],
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]),
    ]
)
