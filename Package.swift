// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
        .executable(name: "transformers", targets: ["TransformersCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.2.0")
    ],
    targets: [
        .target(name: "Hub"),
        .target(name: "Tokenizers", dependencies: ["Hub"], resources: [.process("FallbackConfigs")]),
        .target(name: "TensorUtils"),
        .target(name: "Generation", dependencies: ["Tokenizers", "TensorUtils"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation", "TensorUtils"]),
        .executableTarget(
            name: "TransformersCLI",
            dependencies: [
                "Models", "Generation", "Tokenizers",
                .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "Models"], resources: [.process("Resources"), .process("Vocabs")]),
        .testTarget(name: "HubTests", dependencies: ["Hub"]),
    ]
)
