// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

/// Define the strict concurrency settings to be applied to all targets.
let swiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency")
]

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Hub", targets: ["Hub"]),
        .library(name: "Tokenizers", targets: ["Tokenizers"]),
        .library(name: "TokenizersCore", targets: ["TokenizersCore"]),
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.4.2"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.8.1"),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
    ],
    targets: [
        .target(name: "Generation", dependencies: ["Tokenizers"]),
        .target(
            name: "Hub",
            dependencies: [
                "TokenizerConfig",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "yyjson", package: "yyjson"),
            ],
            resources: [
                .process("Resources")
            ],
            swiftSettings: swiftSettings
        ),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation"]),
        .target(
            name: "TokenizerConfig",
            dependencies: [.product(name: "Jinja", package: "swift-jinja")]
        ),
        .target(
            name: "TokenizersCore",
            dependencies: ["TokenizerConfig", .product(name: "Jinja", package: "swift-jinja")],
            path: "Sources/Tokenizers"
        ),
        .target(name: "Tokenizers", dependencies: ["Hub", "TokenizersCore"], path: "Sources/TokenizersShim"),
        .testTarget(name: "Benchmarks", dependencies: ["Hub", "Tokenizers", "TokenizersCore", .product(name: "yyjson", package: "yyjson")]),
        .testTarget(name: "GenerationTests", dependencies: ["Generation"]),
        .testTarget(name: "HubTests", dependencies: ["Hub", .product(name: "Jinja", package: "swift-jinja")], swiftSettings: swiftSettings),
        .testTarget(name: "ModelsTests", dependencies: ["Models", "Hub"], resources: [.process("Resources")]),
        .testTarget(name: "TokenizersCoreTests", dependencies: ["TokenizersCore"]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "TokenizersCore", "Models", "Hub"], resources: [.process("Resources")]),
    ]
)
