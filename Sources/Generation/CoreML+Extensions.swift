//
//  CoreML+Extensions.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

#if canImport(CoreML)
import CoreML
import Foundation

extension MLMultiArray {
    /// Creates an MLMultiArray from an array of integers.
    ///
    /// All values are stored in the last dimension of the MLMultiArray, with leading
    /// dimensions set to 1. For example, with dims=2, the shape becomes [1, arr.count].
    ///
    /// - Parameters:
    ///   - arr: Array of integers to convert
    ///   - dims: Number of dimensions for the resulting MLMultiArray
    /// - Returns: MLMultiArray containing the integer values
    static func from(_ arr: [Int], dims: Int = 1) -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = arr.count
        // Examples:
        // dims=1 : [arr.count]
        // dims=2 : [1, arr.count]
        //
        let o = try! MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for (i, item) in arr.enumerated() {
            ptr[i] = Int32(item)
        }
        return o
    }

    /// Creates an MLMultiArray from an array of doubles.
    ///
    /// All values are stored in the last dimension of the MLMultiArray, with leading
    /// dimensions set to 1. For example, with dims=2, the shape becomes [1, arr.count].
    ///
    /// - Parameters:
    ///   - arr: Array of doubles to convert
    ///   - dims: Number of dimensions for the resulting MLMultiArray
    /// - Returns: MLMultiArray containing the double values
    static func from(_ arr: [Double], dims: Int = 1) -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = arr.count
        // Examples:
        // dims=1 : [arr.count]
        // dims=2 : [1, arr.count]
        //
        let o = try! MLMultiArray(shape: shape as [NSNumber], dataType: .float64)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        for (i, item) in arr.enumerated() {
            ptr[i] = Double(item)
        }
        return o
    }

    /// Converts an MLMultiArray to a flat array of integers.
    ///
    /// Concatenates all dimensions into a single one-dimensional array by reading
    /// the MLMultiArray data in memory order.
    ///
    /// - Parameter o: MLMultiArray to convert
    /// - Returns: Flat array of integer values
    static func toIntArray(_ o: MLMultiArray) -> [Int] {
        var arr = Array(repeating: 0, count: o.count)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for i in 0..<o.count {
            arr[i] = Int(ptr[i])
        }
        return arr
    }

    /// Converts this MLMultiArray to a flat array of integers.
    ///
    /// - Returns: Flat array of integer values
    func toIntArray() -> [Int] { Self.toIntArray(self) }

    /// Converts an MLMultiArray to a flat array of doubles.
    ///
    /// Concatenates all dimensions into a single one-dimensional array by reading
    /// the MLMultiArray data in memory order.
    ///
    /// - Parameter o: MLMultiArray to convert
    /// - Returns: Flat array of double values
    static func toDoubleArray(_ o: MLMultiArray) -> [Double] {
        var arr: [Double] = Array(repeating: 0, count: o.count)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        for i in 0..<o.count {
            arr[i] = Double(ptr[i])
        }
        return arr
    }

    /// Converts this MLMultiArray to a flat array of doubles.
    ///
    /// - Returns: Flat array of double values
    func toDoubleArray() -> [Double] { Self.toDoubleArray(self) }

    /// Creates a test MLMultiArray with sequentially indexed values.
    ///
    /// Useful for debugging and unit tests. Values are assigned sequentially
    /// starting from 0, following the memory layout of the specified shape.
    ///
    /// Example output for shape [2, 3, 4]:
    /// ```
    /// [[[ 0, 1, 2, 3 ],
    ///   [ 4, 5, 6, 7 ],
    ///   [ 8, 9, 10, 11 ]],
    ///  [[ 12, 13, 14, 15 ],
    ///   [ 16, 17, 18, 19 ],
    ///   [ 20, 21, 22, 23 ]]]
    /// ```
    ///
    /// - Parameter shape: Desired shape of the test tensor
    /// - Returns: MLMultiArray with sequential values for testing
    static func testTensor(shape: [Int]) -> MLMultiArray {
        let arr = try! MLMultiArray(shape: shape as [NSNumber], dataType: .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
        for i in 0..<arr.count {
            ptr.advanced(by: i).pointee = Double(i)
        }
        return arr
    }
}

extension MLMultiArray {
    /// Provides a way to index n-dimensionals arrays a la numpy.
    enum Indexing: Equatable {
        case select(Int)
        case slice
    }

    /// Slice an array according to a list of `Indexing` enums.
    ///
    /// You must specify all dimensions.
    /// Note: only one slice is supported at the moment.
    static func slice(_ o: MLMultiArray, indexing: [Indexing]) -> MLMultiArray {
        assert(
            indexing.count == o.shape.count
        )
        assert(
            indexing.filter { $0 == Indexing.slice }.count == 1
        )
        var selectDims: [Int: Int] = [:]
        for (i, idx) in indexing.enumerated() {
            if case let .select(select) = idx {
                selectDims[i] = select
            }
        }
        return slice(
            o,
            sliceDim: indexing.firstIndex { $0 == Indexing.slice }!,
            selectDims: selectDims
        )
    }

    /// Slice an array according to a list, according to `sliceDim` (which dimension to slice on)
    /// and a dictionary of `dim` to `index`.
    ///
    /// You must select all other dimensions than the slice dimension (cf. the assert).
    static func slice(_ o: MLMultiArray, sliceDim: Int, selectDims: [Int: Int]) -> MLMultiArray {
        assert(
            selectDims.count + 1 == o.shape.count
        )
        var shape: [NSNumber] = Array(repeating: 1, count: o.shape.count)
        shape[sliceDim] = o.shape[sliceDim]
        // print("About to slice ndarray of shape \(o.shape) into ndarray of shape \(shape)")
        let arr = try! MLMultiArray(shape: shape, dataType: .double)

        // let srcPtr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        // TODO: use srcPtr instead of array subscripting.
        let dstPtr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
        for i in 0..<arr.count {
            var index: [Int] = []
            for j in 0..<shape.count {
                if j == sliceDim {
                    index.append(i)
                } else {
                    index.append(selectDims[j]!)
                }
            }
            // print("Accessing element \(index)")
            dstPtr[i] = o[index as [NSNumber]] as! Double
        }
        return arr
    }
}

extension MLMultiArray {
    var debug: String {
        debug([])
    }

    /// From https://twitter.com/mhollemans
    ///
    /// Slightly tweaked
    ///
    func debug(_ indices: [Int]) -> String {
        func indent(_ x: Int) -> String {
            String(repeating: " ", count: x)
        }

        // This function is called recursively for every dimension.
        // Add an entry for this dimension to the end of the array.
        var indices = indices + [0]

        let d = indices.count - 1 // the current dimension
        let N = shape[d].intValue // how many elements in this dimension
        var s = "["
        if indices.count < shape.count { // not last dimension yet?
            for i in 0..<N {
                indices[d] = i
                s += debug(indices) // then call recursively again
                if i != N - 1 {
                    s += ",\n" + indent(d + 1)
                }
            }
        } else { // the last dimension has actual data
            s += " "
            for i in 0..<N {
                indices[d] = i
                s += "\(self[indices as [NSNumber]])"
                if i != N - 1 { // not last element?
                    s += ", "
                    if i % 11 == 10 { // wrap long lines
                        s += "\n " + indent(d + 1)
                    }
                }
            }
            s += " "
        }
        return s + "]"
    }
}

extension MLShapedArray<Float> {
    /// Efficiently extracts float values from the shaped array.
    ///
    /// Uses optimized memory copying when possible (stride=1), falling back to
    /// slower scalar access for non-contiguous arrays.
    ///
    /// - Returns: Array of Float values from the shaped array
    var floats: [Float] {
        guard strides.first == 1, strides.count == 1 else {
            // For some reason this path is slow.
            // If strides is not 1, we can write a Metal kernel to copy the values properly.
            return scalars
        }

        // Fast path: memcpy
        let mlArray = MLMultiArray(self)
        return mlArray.floats ?? scalars
    }
}

extension MLShapedArraySlice<Float> {
    /// Efficiently extracts float values from the shaped array slice.
    ///
    /// Uses optimized memory copying when possible (stride=1), falling back to
    /// slower scalar access for non-contiguous slices.
    ///
    /// - Returns: Array of Float values from the shaped array slice
    var floats: [Float] {
        guard strides.first == 1, strides.count == 1 else {
            // For some reason this path is slow.
            // If strides is not 1, we can write a Metal kernel to copy the values properly.
            return scalars
        }

        // Fast path: memcpy
        let mlArray = MLMultiArray(self)
        return mlArray.floats ?? scalars
    }
}

extension MLMultiArray {
    /// Efficiently extracts float values from the MLMultiArray if it contains float32 data.
    ///
    /// Uses fast memory copying to extract all float values as a contiguous array.
    /// Returns nil if the array doesn't contain float32 data.
    ///
    /// - Returns: Array of Float values, or nil if not float32 type
    var floats: [Float]? {
        guard dataType == .float32 else { return nil }

        var result: [Float] = Array(repeating: 0, count: count)
        return withUnsafeBytes { ptr in
            guard let source = ptr.baseAddress else { return nil }
            result.withUnsafeMutableBytes { resultPtr in
                let dest = resultPtr.baseAddress!
                memcpy(dest, source, self.count * MemoryLayout<Float>.stride)
            }
            return result
        }
    }
}
#endif // canImport(CoreML)
