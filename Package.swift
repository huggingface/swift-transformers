// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

/// Define the strict concurrency settings to be applied to all targets.
let swiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency")
]

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"])
    ],
    dependencies: [
        .package(url: "https://github.com/johnmai-dev/Jinja", .upToNextMinor(from: "1.3.0"))
    ],
    targets: [
        .target(name: "Generation", dependencies: ["Tokenizers"]),
        .target(name: "Hub", resources: [.process("Resources")], swiftSettings: swiftSettings),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation"]),
        .target(name: "Tokenizers", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")]),
        .testTarget(name: "HubTests", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")], swiftSettings: swiftSettings),
        .testTarget(name: "ModelsTests", dependencies: ["Models", "Hub"], resources: [.process("Resources")]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "Models", "Hub"], resources: [.process("Resources")]),
    ]
)
