import Accelerate
import CoreML.MLMultiArray


public extension BNNSNDArrayDescriptor {
    func makeMultiArray<T: Numeric>(of numericType: T.Type, shape: [NSNumber]) throws -> MLMultiArray {
        assert(numericType == Float32.self)
        let strides = shape.dropFirst().reversed().reduce(into: [1]) { acc, a in
            acc.insert(acc[0].intValue * a.intValue as NSNumber, at: 0)
        }

        return try MLMultiArray(dataPointer: self.data!, shape: shape, dataType: .float32, strides: strides)
    }
}
