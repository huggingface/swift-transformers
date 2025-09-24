#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

/// Type-erased random number generator.
internal class AnyRandomNumberGenerator: RandomNumberGenerator {
    private var rng: RandomNumberGenerator

    /// Creates a type-erased random number generator.
    ///
    /// - Parameters:
    ///   - rng: A random number generator.
    init(_ rng: RandomNumberGenerator) {
        self.rng = rng
    }

    func next() -> UInt64 {
        rng.next()
    }
}

extension AnyRandomNumberGenerator: ParallelRandomNumberGenerator {
    func next(count: Int) -> [UInt64] {
        if let rng = rng as? ParallelRandomNumberGenerator {
            return rng.next(count: count)
        }
        return (0..<count).map { _ in rng.next() }
    }

    func next<T: FixedWidthInteger & UnsignedInteger>(count: Int, upperBound: T) -> [T] {
        if let rng = rng as? ParallelRandomNumberGenerator {
            return rng.next(count: count, upperBound: upperBound)
        }
        return (0..<count).map { _ in rng.next(upperBound: upperBound) }
    }
}

// MARK: - Random number generators

typealias DefaultRandomNumberGeneratorForTensor = PhiloxRandomNumberGenerator

/// A type that provides seedable deterministic pseudo-random data.
///
/// A SeedableRandomNumberGenerator can be used anywhere where a
/// RandomNumberGenerator would be used. It is useful when the pseudo-random
/// data needs to be reproducible across runs.
///
/// Conforming to the SeedableRandomNumberGenerator Protocol
/// ========================================================
///
/// To make a custom type conform to the `SeedableRandomNumberGenerator`
/// protocol, implement the `init(seed: [UInt8])` initializer, as well as the
/// requirements for `RandomNumberGenerator`. The values returned by `next()`
/// must form a deterministic sequence that depends only on the seed provided
/// upon initialization.
protocol SeedableRandomNumberGenerator: RandomNumberGenerator {
    init(seed: [UInt8])
    init<T: BinaryInteger>(seed: T)
}

extension SeedableRandomNumberGenerator {
    init<T: BinaryInteger>(seed: T) {
        var newSeed: [UInt8] = []
        for i in 0..<seed.bitWidth / UInt8.bitWidth {
            newSeed.append(UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * i)))
        }
        self.init(seed: newSeed)
    }
}

extension RandomNumberGenerator {
    mutating func next(count: Int) -> [UInt64] {
        if let generator = self as? ParallelRandomNumberGenerator {
            return generator.next(count: count)
        } else {
            return (0..<count).map { _ in next() }
        }
    }

    mutating func next<T: FixedWidthInteger & UnsignedInteger>(
        count: Int,
        upperBound: T
    ) -> [T] {
        if let generator = self as? ParallelRandomNumberGenerator {
            return generator.next(count: count, upperBound: upperBound)
        } else {
            return (0..<count).map { _ in next(upperBound: upperBound) }
        }
    }
}

/// An implementation of `SeedableRandomNumberGenerator` using ARC4.
///
/// ARC4 is a stream cipher that generates a pseudo-random stream of bytes. This
/// PRNG uses the seed as its key.
///
/// ARC4 is described in Schneier, B., "Applied Cryptography: Protocols,
/// Algorithms, and Source Code in C", 2nd Edition, 1996.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
struct ARC4RandomNumberGenerator: SeedableRandomNumberGenerator {
    public static var global = ARC4RandomNumberGenerator(seed: UInt32(time(nil)))
    var state: [UInt8] = Array(0...255)
    var iPos: UInt8 = 0
    var jPos: UInt8 = 0

    /// Initialize ARC4RandomNumberGenerator using an array of UInt8. The array
    /// must have length between 1 and 256 inclusive.
    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 256, "Length of seed must be at most 256")
        var j: UInt8 = 0
        for i: UInt8 in 0...255 {
            j &+= S(i) &+ seed[Int(i) % seed.count]
            swapAt(i, j)
        }
    }

    // Produce the next random UInt64 from the stream, and advance the internal
    // state.
    public mutating func next() -> UInt64 {
        var result: UInt64 = 0
        for _ in 0..<UInt64.bitWidth / UInt8.bitWidth {
            result <<= UInt8.bitWidth
            result += UInt64(nextByte())
        }
        return result
    }

    // Helper to access the state.
    private func S(_ index: UInt8) -> UInt8 {
        return state[Int(index)]
    }

    // Helper to swap elements of the state.
    private mutating func swapAt(_ i: UInt8, _ j: UInt8) {
        state.swapAt(Int(i), Int(j))
    }

    // Generates the next byte in the keystream.
    private mutating func nextByte() -> UInt8 {
        iPos &+= 1
        jPos &+= S(iPos)
        swapAt(iPos, jPos)
        return S(S(iPos) &+ S(jPos))
    }
}

/// An implementation of `SeedableRandomNumberGenerator` using Threefry.
/// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
/// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
///
/// This struct implements a 20-round Threefry2x32 PRNG. It must be seeded with
/// a 64-bit value.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
struct ThreefryRandomNumberGenerator: SeedableRandomNumberGenerator {
    private let rot: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
        = (13, 15, 26, 6, 17, 29, 16, 24)

    private func rotl32(value: UInt32, n: UInt32) -> UInt32 {
        return (value << (n & 31)) | (value >> ((32 - n) & 31))
    }

    private var ctr: UInt64 = 0
    private let key: SIMD2<UInt32>

    private func random(forCtr ctr: SIMD2<UInt32>, key: SIMD2<UInt32>) -> SIMD2<UInt32> {
        let skeinKsParity32: UInt32 = 0x1BD11BDA

        let ks0 = key.x
        let ks1 = key.y
        let ks2 = skeinKsParity32 ^ key.x ^ key.y
        var X0 = ctr.x
        var X1 = ctr.y

        // 20 rounds
        // Key injection (r = 0)
        X0 &+= ks0
        X1 &+= ks1
        // R1
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R2
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R3
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R4
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 1)
        X0 &+= ks1
        X1 &+= (ks2 + 1)
        // R5
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.4)
        X1 ^= X0
        // R6
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.5)
        X1 ^= X0
        // R7
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.6)
        X1 ^= X0
        // R8
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.7)
        X1 ^= X0
        // Key injection (r = 2)
        X0 &+= ks2
        X1 &+= (ks0 + 2)
        // R9
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R10
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R11
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R12
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 3)
        X0 &+= ks0
        X1 &+= (ks1 + 3)
        // R13
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.4)
        X1 ^= X0
        // R14
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.5)
        X1 ^= X0
        // R15
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.6)
        X1 ^= X0
        // R16
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.7)
        X1 ^= X0
        // Key injection (r = 4)
        X0 &+= ks1
        X1 &+= (ks2 + 4)
        // R17
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.0)
        X1 ^= X0
        // R18
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.1)
        X1 ^= X0
        // R19
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.2)
        X1 ^= X0
        // R20
        X0 &+= X1
        X1 = rotl32(value: X1, n: rot.3)
        X1 ^= X0
        // Key injection (r = 5)
        X0 &+= ks2
        X1 &+= (ks0 + 5)

        return [X0, X1]
    }

    internal init(uint64Seed seed: UInt64) {
        key = seed.vector2
    }

    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 8, "Length of seed must be at most 8")
        var combinedSeed: UInt64 = 0
        for (i, byte) in seed.enumerated() {
            combinedSeed += UInt64(byte) << UInt64(8 * i)
        }
        self.init(uint64Seed: combinedSeed)
    }

    public mutating func next() -> UInt64 {
        defer { ctr += 1 }
        return UInt64(highAndLow: random(forCtr: ctr.vector2, key: key))
    }
}

/// An implementation of `SeedableRandomNumberGenerator` using Philox.
/// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
/// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
///
/// This struct implements a 10-round Philox4x32 PRNG. It must be seeded with
/// a 64-bit value.
///
/// An individual generator is not thread-safe, but distinct generators do not
/// share state. The random data generated is of high-quality, but is not
/// suitable for cryptographic applications.
struct PhiloxRandomNumberGenerator: SeedableRandomNumberGenerator {
    @usableFromInline
    var counter: UInt64 = 0
    @usableFromInline
    let key: SIMD2<UInt32>

    // Since we generate two 64-bit values at a time, we only need to run the
    // generator every other invocation.
    @usableFromInline
    var useNextValue = false
    @usableFromInline
    var nextValue: UInt64 = 0

    @inlinable
    func bump(key: SIMD2<UInt32>) -> SIMD2<UInt32> {
        SIMD2<UInt32>(0x9E3779B9, 0xBB67AE85) &+ key
    }

    @inlinable
    func round(counter: SIMD4<UInt32>, key: SIMD2<UInt32>) -> SIMD4<UInt32> {
        let roundConstants = SIMD2<UInt64>(0xD2511F53, 0xCD9E8D57)
        let products = roundConstants &* SIMD2<UInt64>(UInt64(counter[0]), UInt64(counter[2]))

        let hi = SIMD2<UInt32>(truncatingIfNeeded: products &>> 32)
        let lo = SIMD2<UInt32>(truncatingIfNeeded: products & 0x0000_0000_FFFF_FFFF)
        return [
            hi[1] ^ counter[1] ^ key[0],
            lo[1],
            hi[0] ^ counter[3] ^ key[1],
            lo[0]
        ]
    }

    @inlinable
    func random(
        forCounter initialCounter: SIMD4<UInt32>,
        key initialKey: SIMD2<UInt32>
    ) -> SIMD4<UInt32> {
        var counter = initialCounter
        var key = initialKey
        // 10 rounds
        // R1
        counter = round(counter: counter, key: key)
        // R2
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R3
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R4
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R5
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R6
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R7
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R8
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R9
        key = bump(key: key)
        counter = round(counter: counter, key: key)
        // R10
        key = bump(key: key)
        counter = round(counter: counter, key: key)

        return counter
    }

    @inlinable
    public init(uint64Seed seed: UInt64) {
        key = seed.vector2
    }

    @inlinable
    public init(seed: [UInt8]) {
        precondition(seed.count > 0, "Length of seed must be positive")
        precondition(seed.count <= 8, "Length of seed must be at most 8")
        var combinedSeed: UInt64 = 0
        for (i, byte) in seed.enumerated() {
            combinedSeed += UInt64(byte) << UInt64(8 * i)
        }
        self.init(uint64Seed: combinedSeed)
    }

    @inlinable
    public mutating func next() -> UInt64 {
        if useNextValue {
            useNextValue = false
            return nextValue
        }
        let pair = random(forCounter: counter.vector4, key: key).reinterpretedUInt64Vector
        useNextValue = true
        nextValue = pair.y
        counter += 1
        return pair.x
    }
}

/// Private helpers.
extension UInt64 {
    @inlinable
    var vector2: SIMD2<UInt32> {
        let msb = UInt32(truncatingIfNeeded: self >> 32)
        let lsb = UInt32(truncatingIfNeeded: self & 0x0000_0000_FFFF_FFFF)
        return [msb, lsb]
    }

    @inlinable
    var vector4: SIMD4<UInt32> {
        let msb = UInt32(truncatingIfNeeded: self >> 32)
        let lsb = UInt32(truncatingIfNeeded: self)
        return [0, 0, msb, lsb]
    }

    @inlinable
    init(highAndLow: SIMD2<UInt32>) {
        self = (UInt64(highAndLow.x) << 32) + UInt64(highAndLow.y)
    }
}

extension SIMD4 where Scalar == UInt32 {
    @inlinable
    var reinterpretedUInt64Vector: SIMD2<UInt64> {
        let a = (UInt64(x) << 32) + UInt64(y)
        let b = (UInt64(z) << 32) + UInt64(w)
        return [a, b]
    }
}

// MARK: - Random distributions

import Dispatch

protocol RandomDistribution {
    associatedtype Sample
    func next<G: RandomNumberGenerator>(using generator: inout G) -> Sample
    func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [Sample]
}

extension RandomDistribution {
    @_specialize(
        where Self == UniformFloatingPointDistribution<Float>,
              G == DefaultRandomNumberGeneratorForTensor)
    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [Sample] {
        return Array(
            unsafeUninitializedCapacity: count
        ) { buffer, initializedCount in
            for i in 0..<count {
                buffer[i] = next(using: &generator)
            }
            initializedCount = count
        }
    }
}

struct UniformBooleanDistribution: RandomDistribution {
    public init() {}
    public func next<G: RandomNumberGenerator>(using generator: inout G) -> Bool {
        Bool.random(using: &generator)
    }

    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [Bool] {
        Array.random(count: count, using: &generator)
    }
}

struct UniformIntegerDistribution<T: FixedWidthInteger>: RandomDistribution {
    public let bounds: ClosedRange<T>

    public init(bounds: ClosedRange<T> = T.min...T.max) {
        self.bounds = bounds
    }

    public func next<G: RandomNumberGenerator>(using generator: inout G) -> T {
        return T.random(in: bounds, using: &generator)
    }

    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [T] {
        Array.random(count: count, in: bounds, using: &generator)
    }
}

struct UniformFloatingPointDistribution<T: BinaryFloatingPoint>: RandomDistribution
    where T.RawSignificand: FixedWidthInteger
{
    public let bounds: ClosedRange<T>

    public init(bounds: ClosedRange<T> = 0...1) {
        self.bounds = bounds
    }

    @_specialize(where T == Float, G == DefaultRandomNumberGeneratorForTensor)
    public func next<G: RandomNumberGenerator>(using generator: inout G) -> T {
        return T.random(in: bounds, using: &generator)
    }

    @_specialize(where T == Float, G == DefaultRandomNumberGeneratorForTensor)
    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [T] {
        Array.random(count: count, in: bounds, using: &generator)
    }
}

struct NormalDistribution<T: BinaryFloatingPoint>: RandomDistribution
    where T.RawSignificand: FixedWidthInteger
{
    public let mean: T
    public let standardDeviation: T
    @usableFromInline
    let uniformDistribution = UniformFloatingPointDistribution<T>()

    public init(mean: T = 0, standardDeviation: T = 1) {
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    @_specialize(where T == Float)
    @inlinable
    func normalized(_ u1: T, _ u2: T) -> T {
        let r = (-2 * T(log(Float(u1)))).squareRoot()
        let theta = 2 * T.pi * u2
        let normal01 = r * T(cos(Float(theta)))
        return mean + standardDeviation * normal01
    }

    @_specialize(where T == Float, G == DefaultRandomNumberGeneratorForTensor)
    @inlinable
    public func next<G: RandomNumberGenerator>(using generator: inout G) -> T {
        // FIXME: Box-Muller can generate two values for only a little more than the
        // cost of one.
        normalized(
            uniformDistribution.next(using: &generator),
            uniformDistribution.next(using: &generator))
    }

    @_specialize(where T == Float, G == DefaultRandomNumberGeneratorForTensor)
    @inlinable
    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [T] {
        let uniformNumbers = uniformDistribution.next(count * 2, using: &generator)
        return Array(unsafeUninitializedCapacity: count) { buffer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: count) { i in
                let offset = i * 2
                buffer[i] = normalized(uniformNumbers[offset], uniformNumbers[offset + 1])
            }
            initializedCount = count
        }
    }
}

struct TruncatedNormalDistribution<T: BinaryFloatingPoint>: RandomDistribution
    where T.RawSignificand: FixedWidthInteger
{
    public let mean: T
    public let standardDeviation: T
    private let normalDistribution = NormalDistribution(mean: 0, standardDeviation: 1)

    public init(mean: T = 0, standardDeviation: T = 1) {
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    public func next<G: RandomNumberGenerator>(using generator: inout G) -> T {
        // FIXME: Implement this. See
        // https://github.com/tensorflow/tensorflow/blob/b1a6b315a63bb29b4593bfb98095da4397d8cd5a/tensorflow/compiler/tf2xla/lib/random.cc#L42.
        fatalError("Unimplemented")
    }

    public func next<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [T] {
        // FIXME: Implement this. See
        // https://github.com/tensorflow/tensorflow/blob/b1a6b315a63bb29b4593bfb98095da4397d8cd5a/tensorflow/compiler/tf2xla/lib/random.cc#L42.
        fatalError("Unimplemented")
    }
}

struct BetaDistribution: RandomDistribution {
    public let alpha: Float
    public let beta: Float
    private let uniformDistribution = UniformFloatingPointDistribution<Float>()

    public init(alpha: Float = 0, beta: Float = 1) {
        self.alpha = alpha
        self.beta = beta
    }

    public func next<G: RandomNumberGenerator>(using generator: inout G) -> Float {
        // Generate a sample using Cheng's sampling algorithm from:
        // R. C. H. Cheng, "Generating beta variates with nonintegral shape
        // parameters.". Communications of the ACM, 21, 317-322, 1978.
        let a = min(alpha, beta)
        let b = max(alpha, beta)
        if a > 1 {
            return BetaDistribution.chengsAlgorithmBB(alpha, a, b, using: &generator)
        } else {
            return BetaDistribution.chengsAlgorithmBC(alpha, b, a, using: &generator)
        }
    }

    /// Returns one sample from a Beta(alpha, beta) distribution using Cheng's BB
    /// algorithm, when both alpha and beta are greater than 1.
    ///
    /// - Parameters:
    ///   - alpha: First Beta distribution shape parameter.
    ///   - a: `min(alpha, beta)`.
    ///   - b: `max(alpha, beta)`.
    ///   - generator: Random number generator.
    ///
    /// - Returns: Sample obtained using Cheng's BB algorithm.
    private static func chengsAlgorithmBB<G: RandomNumberGenerator>(
        _ alpha0: Float,
        _ a: Float,
        _ b: Float,
        using generator: inout G
    ) -> Float {
        let alpha = a + b
        let beta  = sqrt((alpha - 2) / (2 * a * b - alpha))
        let gamma = a + 1 / beta

        var r: Float = 0.0
        var w: Float = 0.0
        var t: Float = 0.0

        repeat {
            let u1 = Float.random(in: 0.0...1.0, using: &generator)
            let u2 = Float.random(in: 0.0...1.0, using: &generator)
            let v = beta * (log(u1) - log1p(-u1))
            r = gamma * v - 1.3862944
            let z = u1 * u1 * u2
            w = a * exp(v)

            let s = a + r - w
            if s + 2.609438 >= 5 * z {
                break
            }

            t = log(z)
            if s >= t {
                break
            }
        } while r + alpha * (log(alpha) - log(b + w)) < t

        w = min(w, Float.greatestFiniteMagnitude)
        return a == alpha0 ? w / (b + w) : b / (b + w)
    }

    /// Returns one sample from a Beta(alpha, beta) distribution using Cheng's BC
    /// algorithm, when at least one of alpha and beta is less than 1.
    ///
    /// - Parameters:
    ///     - alpha: First Beta distribution shape parameter.
    ///     - a: `max(alpha, beta)`.
    ///     - b: `min(alpha, beta)`.
    ///     - generator: Random number generator.
    ///
    /// - Returns: Sample obtained using Cheng's BB algorithm.
    private static func chengsAlgorithmBC<G: RandomNumberGenerator>(
        _ alpha0: Float,
        _ a: Float,
        _ b: Float,
        using generator: inout G
    ) -> Float {
        let alpha = a + b
        let beta  = 1 / b
        let delta = 1 + a - b
        let k1    = delta * (0.0138889 + 0.0416667 * b) / (a * beta - 0.777778)
        let k2    = 0.25 + (0.5 + 0.25 / delta) * b

        var w: Float = 0.0

        while true {
            let u1 = Float.random(in: 0.0...1.0, using: &generator)
            let u2 = Float.random(in: 0.0...1.0, using: &generator)
            let y = u1 * u2
            let z = u1 * y

            if u1 < 0.5 {
                if 0.25 * u2 + z - y >= k1 {
                    continue
                }
            } else {
                if z <= 0.25 {
                    let v = beta * (log(u1) - log1p(-u1))
                    w = a * exp(v)
                    break
                }
                if z >= k2 {
                    continue
                }
            }

            let v = beta * (log(u1) - log1p(-u1))
            w = a * exp(v)
            if alpha * (log(alpha) - log(b + 1) + v) - 1.3862944 >= log(z) {
                break
            }
        }

        w = min(w, Float.greatestFiniteMagnitude)
        return a == alpha0 ? w / (b + w): b / (b + w)
    }
}

// MARK: Parallel Random Number Generators

protocol ParallelRandomNumberGenerator: RandomNumberGenerator {
    func next(count: Int) -> [UInt64]
    func next<T: FixedWidthInteger & UnsignedInteger>(count: Int, upperBound: T) -> [T]
}

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
public struct SystemArc4RandomNumberGenerator: ParallelRandomNumberGenerator {
    public mutating func next() -> UInt64 {
        var result: UInt64 = 0
        arc4random_buf(&result, MemoryLayout<UInt64>.size)
        return result
    }

    public func next(count: Int) -> [UInt64] {
        return Array(unsafeUninitializedCapacity: count) { buffer, size in
            size = count
            arc4random_buf(
                UnsafeMutableRawPointer(buffer.baseAddress),
                count * MemoryLayout<UInt64>.stride)
        }
    }

    public func next<T: FixedWidthInteger & UnsignedInteger>(count: Int, upperBound: T) -> [T] {
        let rands = next(count: count)
        return Array(unsafeUninitializedCapacity: count) {
            bufferPointer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: count) {
                bufferPointer[$0] = self.upperBound(
                    rands[$0],
                    to: upperBound)
            }
            initializedCount = count
        }
    }

    // Implementation based on https://github.com/apple/swift/blob/master/stdlib/public/core/Random.swift#L93
    private func upperBound<T: FixedWidthInteger & UnsignedInteger>(
        _ val: UInt64,
        to upperBound: T
    ) -> T {
        precondition(upperBound != 0, "upperBound cannot be zero.")
        #if arch(i386) || arch(arm) || arch(arm64_32) // TODO(FIXME) SR-10912
            let tmp = (T.max % upperBound) + 1
            let range = tmp == upperBound ? 0 : tmp
            var random = T(truncatingIfNeeded: val)

            while random < range {
                withUnsafeMutablePointer(to: &random) {
                    arc4random_buf($0, MemoryLayout<T>.size)
                }
            }

            return random % upperBound
        #else
            var random = T(truncatingIfNeeded: val)
            var m = random.multipliedFullWidth(by: upperBound)
            if m.low < upperBound {
              let t = (0 &- upperBound) % upperBound
              while m.low < t {
                withUnsafeMutablePointer(to: &random) {
                    arc4random_buf($0, MemoryLayout<T>.size)
                }
                m = random.multipliedFullWidth(by: upperBound)
              }
            }
            return m.high
        #endif
    }
}
#endif

// MARK: Random Array Generators

extension Array where Element == Bool {
    static func random<RNG: RandomNumberGenerator>(
        count: Int,
        using generator: inout RNG
    ) -> Self {
        let rands = generator.next(count: count)
        return Self(unsafeUninitializedCapacity: count) { bufferPointer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: count) {
                bufferPointer[$0] = (rands[$0] >> 17) & 1 == 0
            }
            initializedCount = count
        }
    }
}

extension Array where Element: BinaryFloatingPoint, Element.RawSignificand: FixedWidthInteger {
    // Implementation based on https://github.com/apple/swift/blob/master/stdlib/public/core/FloatingPoint.swift#L2052
    static func random<RNG: RandomNumberGenerator>(
        count: Int,
        in range: Range<Element>,
        using generator: inout RNG
    ) -> Self {
        let delta = range.upperBound - range.lowerBound
        precondition(delta.isFinite, "There is no uniform distribution on an infinite range")
        func randArray(_ count: Int) -> [UInt64] {
            if Element.RawSignificand.bitWidth == Element.significandBitCount + 1 {
                return generator.next(count: count)
            } else {
                let significandCount = Element.significandBitCount + 1
                let maxSignificand: Element.RawSignificand = 1 << significandCount
                return generator.next(count: count).map { $0 & UInt64(maxSignificand - 1) }
            }
        }
        return Self(unsafeUninitializedCapacity: count) {
            resultBufferPointer, initializedCount in
            var indicesNeedingResampling = [Int](resultBufferPointer.indices)
            while !indicesNeedingResampling.isEmpty {
                let rands = randArray(indicesNeedingResampling.count)
                indicesNeedingResampling.withUnsafeMutableBufferPointer {
                    indicesNeedingResamplingBufferPtr in
                    DispatchQueue.concurrentPerform(
                        iterations: indicesNeedingResamplingBufferPtr.count
                    ) { i in
                        let rand = rands[i]
                        let unitRandom = Element(rand) * (Element.ulpOfOne / 2)
                        let randFloat = delta * unitRandom + range.lowerBound
                        if randFloat != range.upperBound {
                            let index = indicesNeedingResamplingBufferPtr[i]
                            resultBufferPointer[index] = randFloat
                            indicesNeedingResamplingBufferPtr[i] = -1
                        }
                    }
                }
                indicesNeedingResampling.removeAll(where: { $0 == -1 })
            }
            initializedCount = count
        }
    }

    // Implementation based on https://github.com/apple/swift/blob/master/stdlib/public/core/FloatingPoint.swift#L2152
    static func random<RNG: RandomNumberGenerator>(
        count: Int,
        in range: ClosedRange<Element>,
        using generator: inout RNG
    ) -> Self {
        let delta = range.upperBound - range.lowerBound
        precondition(delta.isFinite, "There is no uniform distribution on an infinite range")
        func randArrays(_ count: Int) -> (rand: [UInt64], tmp: [UInt64]?) {
            let rands: [UInt64]
            var tmp: [UInt64]? = nil
            if Element.RawSignificand.bitWidth == Element.significandBitCount + 1 {
                rands = generator.next(count: count)
                tmp = generator.next(count: count)
            } else {
                let significandCount = Element.significandBitCount + 1
                let maxSignificand: Element.RawSignificand = 1 << significandCount
                rands = generator.next(count: count, upperBound: UInt64(maxSignificand + 1))
            }
            return (rands, tmp)
        }

        return Self(unsafeUninitializedCapacity: count) {
            resultBufferPointer, initializedCount in
            let indicesNeedingResampling = [Int](resultBufferPointer.indices)
            let (rands, tmp) = randArrays(indicesNeedingResampling.count)
            DispatchQueue.concurrentPerform(
                iterations: indicesNeedingResampling.count
            ) { i in
                if Element.RawSignificand.bitWidth == Element.significandBitCount + 1 {
                    guard let tmp = tmp else {
                        fatalError("Expected the 'tmp' array to be initialized.")
                    }
                    if rands[i] == Element.RawSignificand.max && (tmp[i] & 1) == 1 {
                        let index = indicesNeedingResampling[i]
                        resultBufferPointer[index] = range.upperBound
                        return
                    }
                } else {
                    let significandCount = Element.significandBitCount + 1
                    let maxSignificand: Element.RawSignificand = 1 << significandCount
                    if rands[i] == maxSignificand {
                        let index = indicesNeedingResampling[i]
                        resultBufferPointer[index] = range.upperBound
                        return
                    }
                }
                let unitRandom = Element(rands[i]) * (Element.ulpOfOne / 2)
                let randFloat = delta * unitRandom + range.lowerBound
                let index = indicesNeedingResampling[i]
                resultBufferPointer[index] = randFloat
            }
            initializedCount = count
        }
    }
}

extension Array where Element: FixedWidthInteger {
    // Implementation based on https://github.com/apple/swift/blob/master/stdlib/public/core/Integers.swift#L2663
    static func random<RNG: RandomNumberGenerator>(
        count: Int,
        in range: Range<Element>,
        using generator: inout RNG
    ) -> Self {
        precondition(!range.isEmpty, "Can't get random value with an empty range")
        // Compute delta, the distance between the lower and upper bounds. This
        // value may not representable by the type Bound if Bound is signed, but
        // is always representable as Bound.Magnitude.
        let delta = Element.Magnitude(truncatingIfNeeded: range.upperBound &- range.lowerBound)
        let rands = generator.next(count: count, upperBound: UInt64(delta))
        return Self(unsafeUninitializedCapacity: count) { bufferPointer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: count) {
                // The mathematical result we want is lowerBound plus a random value in
                // 0 ..< delta. We need to be slightly careful about how we do this
                // arithmetic; the Bound type cannot generally represent the random value,
                // so we use a wrapping addition on Bound.Magnitude. This will often
                // overflow, but produces the correct bit pattern for the result when
                // converted back to Bound.
                bufferPointer[$0] = Element(truncatingIfNeeded:
                    Element.Magnitude(truncatingIfNeeded: range.lowerBound) &+
                        Element.Magnitude(rands[$0]))
            }
            initializedCount = count
        }
    }

    // Implementation based on https://github.com/apple/swift/blob/master/stdlib/public/core/Integers.swift#L2732
    static func random<RNG: RandomNumberGenerator>(
        count: Int,
        in range: ClosedRange<Element>,
        using generator: inout RNG
    ) -> Self {
        precondition(!range.isEmpty, "Can't get random value with an empty range")
        // Compute delta, the distance between the lower and upper bounds. This
        // value may not representable by the type Bound if Bound is signed, but
        // is always representable as Bound.Magnitude.
        var delta = Element.Magnitude(truncatingIfNeeded: range.upperBound &- range.lowerBound)
        // Subtle edge case: if the range is the whole set of representable values,
        // then adding one to delta to account for a closed range will overflow.
        // If we used &+ instead, the result would be zero, which isn't helpful,
        // so we actually need to handle this case separately.
        if delta == Element.Magnitude.max {
            return Self(unsafeUninitializedCapacity: count) {
                bufferPointer, initializedCount in
                DispatchQueue.concurrentPerform(iterations: count) {
                    bufferPointer[$0] =
                        Element(truncatingIfNeeded: generator.next() as Element.Magnitude)
                }
                initializedCount = count
            }
        }
        // Need to widen delta to account for the right-endpoint of a closed range.
        delta += 1
        let rands = generator.next(count: count, upperBound: UInt64(delta))
        return Self(unsafeUninitializedCapacity: count) {
            bufferPointer, initializedCount in
            DispatchQueue.concurrentPerform(iterations: count) {
                // The mathematical result we want is lowerBound plus a random value in
                // 0 ..< delta. We need to be slightly careful about how we do this
                // arithmetic; the Bound type cannot generally represent the random value,
                // so we use a wrapping addition on Bound.Magnitude. This will often
                // overflow, but produces the correct bit pattern for the result when
                // converted back to Bound.
                bufferPointer[$0] = Element(truncatingIfNeeded:
                    Element.Magnitude(truncatingIfNeeded: range.lowerBound) &+
                        Element.Magnitude(rands[$0]))
            }
            initializedCount = count
        }
    }
}

extension RandomNumberGenerator {
    /// Returns a random value within the specified range.
    mutating func next<T>(as type: T.Type, in bounds: Range<T>) -> T
        where T: BinaryFloatingPoint, T.RawSignificand: FixedWidthInteger {
        return T.random(in: bounds, using: &self)
    }

    /// Returns a random value within the specified range.
    mutating func next<T>(as type: T.Type, in bounds: ClosedRange<T>) -> T
        where T: BinaryFloatingPoint, T.RawSignificand: FixedWidthInteger {
        return T.random(in: bounds, using: &self)
    }
}
