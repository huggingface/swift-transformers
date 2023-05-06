// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    products: [
        .library(name: "Transformers", targets: ["Tokenizers"]),
    ],
    targets: [
        .target(name: "Tokenizers"),
        .testTarget(name: "tokenizers-tests", dependencies: ["Tokenizers"]),
    ]
)
