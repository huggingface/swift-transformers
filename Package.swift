// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

/// Define the strict concurrency settings to be applied to all targets.
let swiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency"),
]

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Hub", targets: ["Hub"]),
        // ^ Hub client library
        .library(name: "Tokenizers", targets: ["Tokenizers"]),
        // ^ Tokenizers with optional chat template support via traits
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
        // ^ Everything, including Core ML inference
        .executable(name: "transformers", targets: ["TransformersCLI"]),
        .executable(name: "hub-cli", targets: ["HubCLI"]),
    ],
    traits: [
        .trait(
            name: "ChatTemplates",
            description:
            "Enables chat template support with Jinja templating engine (Swift 6.1+ only)"
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-argument-parser.git", .upToNextMinor(from: "1.4.0")
        ),
        .package(url: "https://github.com/johnmai-dev/Jinja", .upToNextMinor(from: "1.2.1")),
    ],
    targets: [
        .executableTarget(
            name: "TransformersCLI",
            dependencies: [
                "Models", .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "HubCLI",
            dependencies: [
                "Hub", .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "Hub", resources: [.process("FallbackConfigs")], swiftSettings: swiftSettings
        ),
        .target(
            name: "Tokenizers",
            dependencies: [
                "Hub",
                .product(
                    name: "Jinja", package: "Jinja", condition: .when(traits: ["ChatTemplates"])
                ),
            ],
            swiftSettings: swiftSettings
        ),
        .target(name: "TensorUtils"),
        .target(name: "Generation", dependencies: ["Tokenizers", "TensorUtils"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation", "TensorUtils"]),
        .testTarget(
            name: "TokenizersTests", dependencies: ["Tokenizers", "Models", "Hub"],
            resources: [.process("Resources"), .process("Vocabs")]
        ),
        .testTarget(
            name: "HubTests", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")],
            swiftSettings: swiftSettings
        ),
        .testTarget(name: "PreTokenizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(
            name: "TensorUtilsTests", dependencies: ["TensorUtils", "Models", "Hub"],
            resources: [.process("Resources")]
        ),
        .testTarget(name: "NormalizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(name: "PostProcessorTests", dependencies: ["Tokenizers", "Hub"]),
    ]
)
