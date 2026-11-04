import HuggingFace
import XCTest

@testable import Hub

#if canImport(Network)
import Network

final class BackgroundSessionTests: XCTestCase {
    func testBackgroundSessionDownload() async throws {
        try await download(existingFile: false)
    }

    func testBackgroundSessionReplacesExistingFile() async throws {
        try await download(existingFile: true)
    }

    private func download(existingFile: Bool) async throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        defer { listener.cancel() }
        let ready = expectation(description: "Local Hub server is ready")
        listener.stateUpdateHandler = { state in
            if case .ready = state { ready.fulfill() }
        }
        listener.newConnectionHandler = { connection in
            connection.start(queue: .global())
            Self.receiveRequest(on: connection)
        }
        listener.start(queue: .global())
        await fulfillment(of: [ready], timeout: 5)
        let port = try XCTUnwrap(listener.port)

        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let hub = HubApi(
            downloadBase: directory,
            cache: HubCache(cacheDirectory: directory.appending(path: "cache")),
            hfToken: "",
            endpoint: "http://127.0.0.1:\(port.rawValue)",
            useBackgroundSession: true,
            useOfflineMode: false
        )
        let destination = directory.appending(path: "models/test/model/config.json")
        if existingFile {
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("old contents".utf8).write(to: destination)
        }

        let snapshot = try await hub.snapshot(from: "test/model", matching: "config.json")

        XCTAssertEqual(snapshot.appending(path: "config.json"), destination)
        XCTAssertEqual(try Data(contentsOf: destination), Self.contents)
    }

    private static let contents = Data("{\"model_type\":\"test\"}".utf8)

    private static func receiveRequest(on connection: NWConnection, buffered: Data = Data()) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { data, _, isComplete, error in
            let request = buffered + (data ?? Data())
            guard let end = request.range(of: Data("\r\n\r\n".utf8)) else {
                if error != nil || isComplete || request.count >= 16_384 {
                    connection.cancel()
                } else {
                    receiveRequest(on: connection, buffered: request)
                }
                return
            }
            let line = String(decoding: request[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")[0]
            let components = line.split(separator: " ")
            guard components.count == 3 else {
                connection.cancel()
                return
            }

            let body: Data
            let headers: String
            switch components[1] {
            case "/api/models/test/model/revision/main":
                body = Data("{\"siblings\":[{\"rfilename\":\"config.json\"}]}".utf8)
                headers = "Content-Type: application/json\r\nContent-Length: \(body.count)\r\n"
            case "/test/model/resolve/main/config.json":
                body = components[0] == "HEAD" ? Data() : contents
                headers = "X-Repo-Commit: \(String(repeating: "a", count: 40))\r\nETag: \"0123456789abcdef0123456789abcdef01234567\"\r\nContent-Length: \(contents.count)\r\n"
            default:
                connection.cancel()
                return
            }
            let response = Data("HTTP/1.1 200 OK\r\n\(headers)Connection: close\r\n\r\n".utf8) + body
            connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
#endif
