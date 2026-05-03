//
//  BenchmarkSupport.swift
//  swift-transformers
//
//  Shared benchmark helpers for the `Tests/Benchmarks` suite. Each benchmark
//  file gates its `@Suite` on `RUN_BENCHMARKS=1`, so these helpers are only
//  exercised when explicitly opted in.
//

import Dispatch
import Foundation

/// Summary statistics for a sequence of per-iteration benchmark times (ms).
struct BenchmarkStats {
    let mean: Double
    let stdDev: Double
    let p50: Double
    let p95: Double
    let min: Double
    let max: Double

    var formatted: String {
        String(format: "%7.3f ms (± %5.3f, p50 %6.3f, p95 %6.3f)", mean, stdDev, p50, p95)
    }
}

/// Compute mean / stddev / p50 / p95 from a list of per-iteration times.
func benchmarkStats(_ times: [Double]) -> BenchmarkStats {
    let sorted = times.sorted()
    let mean = sorted.reduce(0, +) / Double(sorted.count)
    let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(sorted.count)
    let stdDev = variance.squareRoot()
    let p50 = sorted[sorted.count / 2]
    let p95 = sorted[Swift.min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
    return BenchmarkStats(
        mean: mean, stdDev: stdDev, p50: p50, p95: p95,
        min: sorted.first ?? 0, max: sorted.last ?? 0
    )
}

/// Run `block` `iterations` times after `warmup` warm-up runs and return the
/// aggregated stats. Each iteration's wall-clock time is captured via
/// `DispatchTime.now()`. The label is printed alongside the stats so a single
/// benchmark suite can group multiple measurements in its output.
@discardableResult
func benchmarkMeasure(
    label: String, iterations: Int, warmup: Int = 3, _ block: () -> Void
) -> BenchmarkStats {
    for _ in 0..<warmup { block() }
    var times: [Double] = []
    times.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = DispatchTime.now()
        block()
        let end = DispatchTime.now()
        times.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
    }
    let s = benchmarkStats(times)
    print("  \(label.padding(toLength: 26, withPad: " ", startingAt: 0)) \(s.formatted)")
    return s
}
