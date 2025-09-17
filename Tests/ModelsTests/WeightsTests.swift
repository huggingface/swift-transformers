import Foundation
@testable import Hub
@testable import Models
import Testing

@Suite("Weights Tests")
struct WeightsTests {
    let downloadDestination: URL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appending(component: "huggingface-tests")

    var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    @Test("Load weights from file URL")
    func loadWeightsFromFileURL() async throws {
        let repo = "google/bert_uncased_L-2_H-128_A-2"
        let modelDir = try await hubApi.snapshot(from: repo, matching: ["config.json", "model.safetensors"])

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.isReadableKey])
        #expect(files.contains(where: { $0.lastPathComponent == "config.json" }))
        #expect(files.contains(where: { $0.lastPathComponent == "model.safetensors" }))

        let modelFile = modelDir.appending(path: "/model.safetensors")
        let weights = try Weights.from(fileURL: modelFile)
        #expect(weights["bert.embeddings.LayerNorm.bias"]!.dataType == .float32)
        #expect(weights["bert.embeddings.LayerNorm.bias"]!.count == 128)
        #expect(weights["bert.embeddings.LayerNorm.bias"]!.shape.count == 1)

        #expect(weights["bert.embeddings.word_embeddings.weight"]!.dataType == .float32)
        #expect(weights["bert.embeddings.word_embeddings.weight"]!.count == 3906816)
        #expect(weights["bert.embeddings.word_embeddings.weight"]!.shape.count == 2)

        #expect(abs(weights["bert.embeddings.word_embeddings.weight"]![[0, 0]].floatValue - -0.0041) < 1e-3)
        #expect(abs(weights["bert.embeddings.word_embeddings.weight"]![[3, 4]].floatValue - 0.0037) < 1e-3)
        #expect(abs(weights["bert.embeddings.word_embeddings.weight"]![[5, 3]].floatValue - -0.5371) < 1e-3)
        #expect(abs(weights["bert.embeddings.word_embeddings.weight"]![[7, 8]].floatValue - 0.0460) < 1e-3)
        #expect(abs(weights["bert.embeddings.word_embeddings.weight"]![[11, 7]].floatValue - -0.0058) < 1e-3)
    }

    @Test("Read 1-dimensional tensor from a safetensors file")
    func safetensorReadTensor1D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-1d-int32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        #expect(tensor.dataType == .int32)
        #expect(tensor[[0]] == 1)
        #expect(tensor[[1]] == 2)
        #expect(tensor[[2]] == 3)
    }

    @Test("Read 2-dimensional tensor from a safetensors file")
    func safetensorReadTensor2D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-2d-float64", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        #expect(tensor.dataType == .float64)
        #expect(tensor[[0, 0]] == 1)
        #expect(tensor[[0, 1]] == 2)
        #expect(tensor[[0, 2]] == 3)
        #expect(tensor[[1, 0]] == 24)
        #expect(tensor[[1, 1]] == 25)
        #expect(tensor[[1, 2]] == 26)
    }

    @Test("Read 3-dimensional tensor from a safetensors file")
    func safetensorReadTensor3D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-3d-float32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        #expect(tensor.dataType == .float32)
        #expect(tensor[[0, 0, 0]] == 22)
        #expect(tensor[[0, 0, 1]] == 23)
        #expect(tensor[[0, 0, 2]] == 24)
        #expect(tensor[[0, 1, 0]] == 11)
        #expect(tensor[[0, 1, 1]] == 12)
        #expect(tensor[[0, 1, 2]] == 13)
        #expect(tensor[[1, 0, 0]] == 2)
        #expect(tensor[[1, 0, 1]] == 3)
        #expect(tensor[[1, 0, 2]] == 4)
        #expect(tensor[[1, 1, 0]] == 1)
        #expect(tensor[[1, 1, 1]] == 2)
        #expect(tensor[[1, 1, 2]] == 3)
    }

    @Test("Read 4-dimensional tensor from a safetensors file")
    func safetensorReadTensor4D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-4d-float32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        #expect(tensor.dataType == .float32)
        #expect(tensor[[0, 0, 0, 0]] == 11)
        #expect(tensor[[0, 0, 0, 1]] == 12)
        #expect(tensor[[0, 0, 0, 2]] == 13)
        #expect(tensor[[0, 0, 1, 0]] == 1)
        #expect(tensor[[0, 0, 1, 1]] == 2)
        #expect(tensor[[0, 0, 1, 2]] == 3)
        #expect(tensor[[0, 0, 2, 0]] == 4)
        #expect(tensor[[0, 0, 2, 1]] == 5)
        #expect(tensor[[0, 0, 2, 2]] == 6)
        #expect(tensor[[1, 0, 0, 0]] == 22)
        #expect(tensor[[1, 0, 0, 1]] == 23)
        #expect(tensor[[1, 0, 0, 2]] == 24)
        #expect(tensor[[1, 0, 1, 0]] == 15)
        #expect(tensor[[1, 0, 1, 1]] == 16)
        #expect(tensor[[1, 0, 1, 2]] == 17)
        #expect(tensor[[1, 0, 2, 0]] == 17)
        #expect(tensor[[1, 0, 2, 1]] == 18)
        #expect(tensor[[1, 0, 2, 2]] == 19)
    }
}
