// swift-tools-version: 5.9
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
        // ^ Tokenizers. Includes `Hub` to download config files
        .library(name: "TokenizersTemplates", targets: ["TokenizersTemplates"]),
        // ^ Optionally depend on this to add chat template support to Tokenizers
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
        // ^ Everything, including Core ML inference
        .executable(name: "transformers", targets: ["TransformersCLI"]),
        .executable(name: "hub-cli", targets: ["HubCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/johnmai-dev/Jinja", .upToNextMinor(from: "1.2.1")),
    ],
    targets: [
        .executableTarget(
            name: "TransformersCLI",
            dependencies: [ "Models", .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .executableTarget(name: "HubCLI", dependencies: ["Hub", .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .target(name: "Hub", resources: [.process("FallbackConfigs")], swiftSettings: swiftSettings),
        .target(name: "TokenizersCore", dependencies: ["Hub"], path: "Sources/Tokenizers"),
        .target(name: "TokenizersTemplates", dependencies: ["TokenizersCore", .product(name: "Jinja", package: "Jinja")]),
        .target(name: "Tokenizers", dependencies: ["TokenizersCore", .product(name: "Jinja", package: "Jinja")], path: "Sources/TokenizersWrapper"),
        // ^ This is just a wrapper or façade against TokenizersCore, but adds templates if available
        .target(name: "TensorUtils"),
        .target(name: "Generation", dependencies: ["Tokenizers", "TensorUtils"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation", "TensorUtils"]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "Models", "Hub"], resources: [.process("Resources"), .process("Vocabs")]),
        .testTarget(name: "HubTests", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")], swiftSettings: swiftSettings),
        .testTarget(name: "PreTokenizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(name: "TensorUtilsTests", dependencies: ["TensorUtils", "Models", "Hub"], resources: [.process("Resources")]),
        .testTarget(name: "NormalizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(name: "PostProcessorTests", dependencies: ["Tokenizers", "Hub"]),
    ]
)
