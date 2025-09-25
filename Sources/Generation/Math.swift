//
//  Math.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

#if canImport(CoreML) && canImport(Accelerate)
import Accelerate
import CoreML
import Foundation

/// Mathematical utilities for text generation and tensor operations.
///
/// Provides optimized implementations of common mathematical operations
/// used in text generation, including argmax, softmax, sampling, and cumulative sum.
///
/// - Note: From M.I. Hollemans - https://github.com/hollance/CoreMLHelpers
public enum Math {
    /**
     Returns the index and value of the largest element in the array.
    
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    public static func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    /**
     Returns the index and value of the largest element in the array.
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    public static func argmax(_ ptr: UnsafePointer<Double>, count: Int, stride: Int = 1) -> (Int, Double) {
        var maxValue: Double = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxviD(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    /// Returns the index and value of the largest element in a Float array.
    ///
    /// - Parameters:
    ///   - ptr: Pointer to the first element in memory
    ///   - count: How many elements to look at
    ///   - stride: The distance between two elements in memory
    /// - Returns: Tuple of (index, value) of the maximum element
    public static func argmax32(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    /// Returns the index and value of the largest element in an MLMultiArray of doubles.
    ///
    /// - Parameter multiArray: Input MLMultiArray with double precision values
    /// - Returns: Tuple of (index, value) of the maximum element
    public static func argmax(_ multiArray: MLMultiArray) -> (Int, Double) {
        assert(multiArray.dataType == .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(multiArray.dataPointer))
        return Math.argmax(ptr, count: multiArray.count)
    }

    /// Returns the index and value of the largest element in an MLMultiArray of floats.
    ///
    /// - Parameter multiArray: Input MLMultiArray with single precision values
    /// - Returns: Tuple of (index, value) of the maximum element
    public static func argmax32(_ multiArray: MLMultiArray) -> (Int, Float) {
        assert(multiArray.dataType == .float32)
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        return Math.argmax32(ptr, count: multiArray.count)
    }

    /// Returns the cumulative sum of the array.
    ///
    /// Computes the cumulative sum where each element is the sum of all previous elements
    /// plus the current element.
    ///
    /// - Parameter arr: Input array of Float values
    /// - Returns: Array of cumulative sums
    public static func cumsum(_ arr: [Float]) -> [Float] {
        guard !arr.isEmpty else {
            return []
        }
        let arrCount = vDSP_Length(arr.count)
        var weight: Float = 1.0
        var result: [Float] = Array(repeating: 0.0, count: arr.count)
        var firstItem = arr[0]
        vDSP_vrsum(arr, 1, &weight, &result, 1, arrCount)
        vDSP_vsadd(result, 1, &firstItem, &result, 1, arrCount)
        return result
    }

    /// Performs multinomial sampling from probability distributions.
    ///
    /// Selects an index based on probability weights, commonly used for token sampling
    /// in text generation after applying logits warpers.
    ///
    /// - Parameters:
    ///   - indexes: Array of indices to sample from
    ///   - probs: Probability weights for each index
    /// - Returns: Selected index based on probability distribution
    public static func sample(indexes: [Int], probs: [Float]) -> Int {
        let i = randomNumber(probabilities: probs)
        return indexes[i]
    }

    /// Computes the softmax function over an array.
    ///
    /// Converts logits into a probability distribution by applying exponential normalization.
    /// Uses numerical stability techniques by shifting values to prevent overflow.
    ///
    /// The implementation follows this algorithm:
    /// 1. Subtract maximum value for numerical stability
    /// 2. Apply exponential function to all elements
    /// 3. Normalize by dividing by the sum of all exponentials
    ///
    /// - Parameter x: Input logits array
    /// - Returns: Probability distribution (sums to 1.0)
    ///
    /// - Note: Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
    public static func softmax(_ x: [Float]) -> [Float] {
        var x = x
        let len = vDSP_Length(x.count)

        // Find the maximum value in the input array.
        var max: Float = 0
        vDSP_maxv(x, 1, &max, len)

        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsadd(x, 1, &max, &x, 1, len)

        // Exponentiate all the elements in the array.
        var count = Int32(x.count)
        vvexpf(&x, x, &count)

        // Compute the sum of all exponentiated values.
        var sum: Float = 0
        vDSP_sve(x, 1, &sum, len)

        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdiv(x, 1, &sum, &x, 1, len)

        return x
    }

    /// Generates a random index based on probability weights.
    ///
    /// Uses the roulette wheel selection algorithm to choose an index where
    /// the probability of selection is proportional to the weight at that index.
    ///
    /// - Parameter probabilities: Array of probability weights (need not sum to 1.0)
    /// - Returns: Selected index based on probability distribution
    ///
    /// - Note: From https://stackoverflow.com/questions/30309556/generate-random-numbers-with-a-given-distribution
    public static func randomNumber(probabilities: [Float]) -> Int {
        // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
        let sum = probabilities.reduce(0, +)
        // Random number in the range 0.0 <= rnd < sum :
        let rnd = sum * Float(arc4random_uniform(UInt32.max)) / Float(UInt32.max)
        // Find the first interval of accumulated probabilities into which `rnd` falls:
        var accum: Float = 0.0
        for (i, p) in probabilities.enumerated() {
            accum += p
            if rnd < accum {
                return i
            }
        }
        // This point might be reached due to floating point inaccuracies:
        return probabilities.count - 1
    }
}

/// MLShapedArray extensions for Math operations.
public extension Math {
    /// Returns the index and value of the largest element in an MLShapedArray of floats.
    ///
    /// - Parameter shapedArray: Input MLShapedArray containing Float values
    /// - Returns: Tuple of (index, value) of the maximum element
    static func argmax(_ shapedArray: MLShapedArray<Float>) -> (Int, Float) {
        shapedArray.withUnsafeShapedBufferPointer { ptr, shape, strides in
            assert(shape.count == 1, "Only supported for 1-dimensional arrays or slices")
            return Math.argmax32(ptr.baseAddress!, count: shapedArray.count, stride: strides.first!)
        }
    }

    /// Returns the index and value of the largest element in a generic MLShapedArray.
    ///
    /// - Parameter shapedArray: Input shaped array conforming to MLShapedArrayProtocol
    /// - Returns: Tuple of (index, value) of the maximum element as Float
    ///
    /// - Note: Currently assumes Float data type
    static func argmax(_ shapedArray: some MLShapedArrayProtocol) -> (Int, Float) {
        shapedArray.withUnsafeShapedBufferPointer { ptr, shape, strides in
            assert(shape.count == 1, "Only supported for 1-dimensional arrays or slices")
            let floatsPtr = ptr.baseAddress as! UnsafePointer<Float>
            return Math.argmax32(floatsPtr, count: shapedArray.count, stride: strides.first!)
        }
    }
}
#endif // canImport(CoreML) && canImport(Accelerate)
