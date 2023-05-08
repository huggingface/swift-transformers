// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
    ],
    targets: [
        .target(name: "Tokenizers", resources: [.process("Vocabs")]),
        .target(name: "Generation", dependencies: ["Tokenizers"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation"]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers"], resources: [.process("Resources")]),
    ]
)
