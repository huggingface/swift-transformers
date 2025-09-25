// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "transformers-cli",
    platforms: [.iOS(.v18), .macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "transformers-cli",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        )
    ]
)
