import Accelerate
import CoreML.MLMultiArray


public extension BNNSNDArrayDescriptor {
    func makeMultiArray<T: Numeric>(of: T.Type, shape: [NSNumber]) throws -> MLMultiArray {
        let dataType: MLMultiArrayDataType
        switch of {
        case is Int32.Type: dataType = .int32
        case is Float32.Type: dataType = .float32
        case is Double.Type: dataType = .double
        default: fatalError("type not supported")
        }

        let strides = shape.dropFirst().reversed().reduce(into: [1]) { acc, a in
            acc.insert(acc[0].intValue * a.intValue as NSNumber, at: 0)
        }

        return try MLMultiArray(dataPointer: self.data!, shape: shape, dataType: dataType, strides: strides)
    }
}
