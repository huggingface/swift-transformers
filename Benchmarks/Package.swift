// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "benchmarks",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(name: "swift-transformers", path: ".."),
        .package(url: "https://github.com/ordo-one/benchmark", from: "1.29.7"),
    ],
    targets: [
        .executableTarget(
            name: "TokenizerBenchmarks",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Benchmark", package: "benchmark"),
            ],
            path: "Benchmarks/TokenizerBenchmarks",
            plugins: [
                .plugin(name: "BenchmarkPlugin", package: "benchmark")
            ]
        )
    ]
)
