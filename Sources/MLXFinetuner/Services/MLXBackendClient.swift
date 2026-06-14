import Darwin
import Foundation

enum BackendClientError: LocalizedError {
    case backendMissing(String)
    case launchFailed(String)
    case commandFailed(String)
    case decodeFailed(String)

    var errorDescription: String? {
        switch self {
        case .backendMissing(let path):
            "Backend script not found at \(path)."
        case .launchFailed(let message):
            message
        case .commandFailed(let message):
            message
        case .decodeFailed(let message):
            message
        }
    }
}

final class MLXBackendClient {
    private var currentTrainingProcess: Process?

    func prepareEnvironment(pythonOverride: String) async throws -> String {
        try await Task.detached {
            try self.prepareEnvironmentBlocking(pythonOverride: pythonOverride)
        }.value
    }

    func runJSON<T: Decodable>(
        pythonOverride: String,
        command: String,
        arguments: [String]
    ) async throws -> T {
        try await Task.detached {
            let python = try self.prepareEnvironmentBlocking(pythonOverride: pythonOverride)
            let result = try self.runBlocking(
                python: python,
                arguments: [command] + arguments
            )
            guard result.status == 0 else {
                throw BackendClientError.commandFailed(cleanBackendError(stdout: result.stdout, stderr: result.stderr))
            }
            guard let data = result.stdout.data(using: .utf8) else {
                throw BackendClientError.decodeFailed("Backend returned non-UTF8 output.")
            }
            do {
                return try JSONDecoder().decode(T.self, from: data)
            } catch {
                throw BackendClientError.decodeFailed("Could not decode backend JSON: \(error.localizedDescription)\n\(result.stdout)")
            }
        }.value
    }

    func streamTraining(
        pythonOverride: String,
        config: TrainingConfig,
        onEvent: @escaping (TrainingEvent) -> Void,
        onRawLine: @escaping (String) -> Void
    ) async throws {
        let python = try await prepareEnvironment(pythonOverride: pythonOverride)
        let configURL = try writeTemporaryConfig(config)
        defer { try? FileManager.default.removeItem(at: configURL) }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            do {
                let process = try makeProcess(
                    python: python,
                    arguments: ["train", "--config", configURL.path]
                )
                let stdout = Pipe()
                let stderr = Pipe()
                process.standardOutput = stdout
                process.standardError = stderr

                let stdoutBuffer = LockedLineBuffer()
                let stderrBuffer = LockedLineBuffer()
                let outputTail = LockedLogTail(limit: 80)
                let resumeGate = LockedGate()

                @Sendable func resumeOnce(_ result: Result<Void, Error>) {
                    guard resumeGate.enter() else { return }
                    stdout.fileHandleForReading.readabilityHandler = nil
                    stderr.fileHandleForReading.readabilityHandler = nil
                    switch result {
                    case .success:
                        continuation.resume()
                    case .failure(let error):
                        continuation.resume(throwing: error)
                    }
                }

                stdout.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    let lines = stdoutBuffer.append(data)

                    for line in lines {
                        outputTail.append(line)
                        if let event = decodeEvent(line) {
                            onEvent(event)
                        } else {
                            onRawLine(line)
                        }
                    }
                }

                stderr.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    let lines = stderrBuffer.append(data)
                    for line in lines {
                        outputTail.append(line)
                        onRawLine(line)
                    }
                }

                process.terminationHandler = { proc in
                    let remainingOut = stdoutBuffer.remaining()
                    let remainingErr = stderrBuffer.remaining()
                    if !remainingOut.isEmpty {
                        for line in remainingOut.split(whereSeparator: \.isNewline) {
                            let text = String(line)
                            outputTail.append(text)
                            if let event = decodeEvent(text) {
                                onEvent(event)
                            } else {
                                onRawLine(text)
                            }
                        }
                    }
                    if !remainingErr.isEmpty {
                        for line in remainingErr.split(whereSeparator: \.isNewline) {
                            let text = String(line)
                            outputTail.append(text)
                            onRawLine(text)
                        }
                    }
                    self.currentTrainingProcess = nil
                    if proc.terminationStatus == 0 {
                        resumeOnce(.success(()))
                    } else {
                        resumeOnce(.failure(BackendClientError.commandFailed(trainingFailureMessage(for: proc, outputTail: outputTail.snapshot()))))
                    }
                }

                currentTrainingProcess = process
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    func stopTraining() {
        guard let process = currentTrainingProcess, process.isRunning else { return }
        process.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
            if process.isRunning {
                process.interrupt()
            }
        }
    }

    private func runBlocking(python: String, arguments: [String]) throws -> (status: Int32, stdout: String, stderr: String) {
        let process = try makeProcess(python: python, arguments: arguments)
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(data: out, encoding: .utf8) ?? "",
            String(data: err, encoding: .utf8) ?? ""
        )
    }

    private func makeProcess(python: String, arguments: [String]) throws -> Process {
        let backendPath = backendScriptPath()
        guard FileManager.default.fileExists(atPath: backendPath) else {
            throw BackendClientError.backendMissing(backendPath)
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = [backendPath] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        return process
    }

    private func prepareEnvironmentBlocking(pythonOverride: String) throws -> String {
        let trimmed = pythonOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        if let env = ProcessInfo.processInfo.environment["MLX_FINETUNER_PYTHON"], !env.isEmpty {
            return env
        }

        let venvPython = managedVenvPythonPath()
        let marker = URL(fileURLWithPath: venvPython)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".mlx-finetuner-ready")
            .path
        if FileManager.default.fileExists(atPath: venvPython),
           FileManager.default.fileExists(atPath: marker) {
            return venvPython
        }

        let bootstrapPython = pythonExecutable(pythonOverride: "")
        let venvRoot = URL(fileURLWithPath: venvPython)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        try FileManager.default.createDirectory(at: venvRoot, withIntermediateDirectories: true)

        if !FileManager.default.fileExists(atPath: venvPython) {
            try runSetupCommand(executable: bootstrapPython, arguments: ["-m", "venv", venvRoot.path])
        }

        try runSetupCommand(executable: venvPython, arguments: ["-m", "pip", "install", "-U", "pip"])
        try runSetupCommand(executable: venvPython, arguments: ["-m", "pip", "install", "-r", requirementsPath()])
        try "ready\n".write(toFile: marker, atomically: true, encoding: .utf8)
        return venvPython
    }

    private func runSetupCommand(executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-finetuner-setup-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: outputURL)
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        try? output.close()
        if process.terminationStatus != 0 {
            let data = (try? Data(contentsOf: outputURL)) ?? Data()
            let text = String(data: data, encoding: .utf8) ?? ""
            try? FileManager.default.removeItem(at: outputURL)
            throw BackendClientError.commandFailed("Python environment setup failed: \(text)")
        }
        try? FileManager.default.removeItem(at: outputURL)
    }

    private func pythonExecutable(pythonOverride: String) -> String {
        let trimmed = pythonOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        if let env = ProcessInfo.processInfo.environment["MLX_FINETUNER_PYTHON"], !env.isEmpty {
            return env
        }
        if let runtime = runtimeConfig()["python"] as? String, !runtime.isEmpty {
            return runtime
        }
        return "/usr/bin/python3"
    }

    private func managedVenvPythonPath() -> String {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return support
            .appendingPathComponent("MLXFinetuner")
            .appendingPathComponent("PythonEnv")
            .appendingPathComponent("bin/python")
            .path
    }

    private func requirementsPath() -> String {
        if let resourcePath = Bundle.main.resourcePath {
            let bundled = URL(fileURLWithPath: resourcePath)
                .appendingPathComponent("requirements.txt")
                .path
            if FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("requirements.txt")
            .path
    }

    private func backendScriptPath() -> String {
        if let env = ProcessInfo.processInfo.environment["MLX_FINETUNER_BACKEND"], !env.isEmpty {
            return env
        }
        if let resourcePath = Bundle.main.resourcePath {
            let bundled = URL(fileURLWithPath: resourcePath)
                .appendingPathComponent("Backend/mlx_finetuner_backend.py")
                .path
            if FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        let cwd = FileManager.default.currentDirectoryPath
        return URL(fileURLWithPath: cwd).appendingPathComponent("Backend/mlx_finetuner_backend.py").path
    }

    private func runtimeConfig() -> [String: Any] {
        guard let resourcePath = Bundle.main.resourcePath else { return [:] }
        let url = URL(fileURLWithPath: resourcePath).appendingPathComponent("runtime.json")
        guard
            let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return [:]
        }
        return json
    }

    private func writeTemporaryConfig(_ config: TrainingConfig) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-finetuner-\(UUID().uuidString).json")
        let data = try JSONEncoder().encode(config)
        try data.write(to: url, options: .atomic)
        return url
    }
}

private func splitCompleteLines(from buffer: inout Data) -> [String] {
    guard let text = String(data: buffer, encoding: .utf8) else {
        buffer.removeAll()
        return []
    }
    let parts = text.split(separator: "\n", omittingEmptySubsequences: false)
    guard text.hasSuffix("\n") else {
        buffer = Data(String(parts.last ?? "").utf8)
        return parts.dropLast().map(String.init)
    }
    buffer.removeAll()
    return parts.dropLast().map(String.init)
}

private func decodeEvent(_ line: String) -> TrainingEvent? {
    guard let data = line.data(using: .utf8) else { return nil }
    return try? JSONDecoder().decode(TrainingEvent.self, from: data)
}

private func cleanBackendError(stdout: String, stderr: String) -> String {
    for candidate in [stdout, stderr] where !candidate.isEmpty {
        for line in candidate.split(whereSeparator: \.isNewline).map(String.init).reversed() {
            guard let data = line.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let message = object["message"] as? String,
                  !message.isEmpty
            else {
                continue
            }
            return message
        }
    }
    let fallback = stderr.isEmpty ? stdout : stderr
    return fallback.trimmingCharacters(in: .whitespacesAndNewlines)
}

private func trainingFailureMessage(for process: Process, outputTail: [String]) -> String {
    let base: String
    if process.terminationReason == .uncaughtSignal {
        let signal = process.terminationStatus
        if signal == SIGABRT {
            base = "Training was aborted by SIGABRT while MLX/Metal was running. This usually means unified-memory pressure during allocation."
        } else {
            base = "Training stopped after signal \(signal)."
        }
    } else {
        base = "Training exited with status \(process.terminationStatus)."
    }

    let recent = outputTail
        .suffix(14)
        .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        .joined(separator: "\n")
    guard !recent.isEmpty else { return base }
    return "\(base)\nRecent backend output:\n\(recent)"
}

private final class LockedLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()

    func append(_ data: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(data)
        return splitCompleteLines(from: &buffer)
    }

    func remaining() -> String {
        lock.lock()
        defer {
            buffer.removeAll()
            lock.unlock()
        }
        return String(data: buffer, encoding: .utf8) ?? ""
    }
}

private final class LockedLogTail: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private var lines: [String] = []

    init(limit: Int) {
        self.limit = max(1, limit)
    }

    func append(_ line: String) {
        lock.lock()
        defer { lock.unlock() }
        lines.append(line)
        if lines.count > limit {
            lines.removeFirst(lines.count - limit)
        }
    }

    func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}

private final class LockedGate: @unchecked Sendable {
    private let lock = NSLock()
    private var didEnter = false

    func enter() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !didEnter else { return false }
        didEnter = true
        return true
    }
}
