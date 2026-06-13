import Foundation
import SwiftUI

@MainActor
final class AppStore: ObservableObject {
    @Published var selectedStep: FlowStep = .model
    @Published var modelId = "Qwen/Qwen2.5-0.5B"
    @Published var modelDirectory = "\(NSHomeDirectory())/MLXModels"
    @Published var datasetPath = ""
    @Published var outputDirectory = "\(NSHomeDirectory())/MLXFinetunes"
    @Published var resumeAdapterPath = ""
    @Published var hfRepoId = ""

    @Published var method: TrainingMethod = .qlora
    @Published var vlmComponent: VLMComponent = .languageModel
    @Published var precision: Precision = .bf16
    @Published var learningRate = 2e-5
    @Published var batchSize = 1
    @Published var maxSeqLength = 2048
    @Published var steps = 100
    @Published var epochs = 0.0
    @Published var gradientAccumulation = 1
    @Published var warmupSteps = 0
    @Published var imageResolution = 768
    @Published var maxPixels = 1_048_576
    @Published var qloraBits = 4
    @Published var loraRank = 8
    @Published var loraAlpha = 16
    @Published var loraDropout = 0.0
    @Published var targetModules = "q_proj,v_proj"
    @Published var pushToHF = false
    @Published var dryRun = false

    @Published var modelInspection: ModelInspection?
    @Published var datasetPreview: DatasetPreview?
    @Published var logs: [String] = []
    @Published var metrics: [TrainingMetric] = []
    @Published var currentStep = 0
    @Published var totalSteps = 0
    @Published var tokensPerSecond: Double?
    @Published var etaSeconds: Double?
    @Published var isWorking = false
    @Published var isTraining = false
    @Published var statusMessage = "Ready"
    @Published var lastOutputPath: String?

    @AppStorage("backendPython") var backendPython = ""

    private let backend = MLXBackendClient()
    private var didStartBackendPreparation = false

    var detectedBackend: ModelKind {
        modelInspection?.kind ?? datasetPreview?.kind ?? .unknown
    }

    var memoryEstimate: String {
        let seqFactor = Double(maxSeqLength) / 2048.0
        let batchFactor = Double(batchSize * gradientAccumulation)
        let base = method == .full ? 18.0 : 5.0
        let imageFactor = detectedBackend == .visionLanguage ? Double(maxPixels) / 1_048_576.0 : 0.0
        let gb = max(2.0, base * seqFactor * max(1.0, batchFactor) + imageFactor * 2.0)
        return String(format: "~%.1f GB unified memory", gb)
    }

    func prepareBackendEnvironment() async {
        guard !didStartBackendPreparation else { return }
        didStartBackendPreparation = true
        isWorking = true
        statusMessage = "Preparing app-managed Python environment..."
        do {
            let python = try await backend.prepareEnvironment(pythonOverride: backendPython)
            statusMessage = "Backend ready: \(python)"
        } catch {
            statusMessage = "Backend setup failed: \(error.localizedDescription)"
        }
        isWorking = false
    }

    func inspectModel(download: Bool = false) async {
        guard !modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusMessage = "Enter a Hugging Face model ID first."
            return
        }
        isWorking = true
        statusMessage = download ? "Downloading model..." : "Inspecting model config..."
        do {
            let inspection: ModelInspection = try await backend.runJSON(
                pythonOverride: backendPython,
                command: "inspect-model",
                arguments: [
                    "--model-id", modelId,
                    "--download-dir", modelDirectory
                ] + (download ? ["--download"] : [])
            )
            modelInspection = inspection
            statusMessage = inspection.kind == .unknown ? "Model inspected; backend is uncertain." : "Detected \(inspection.kind.title)."
            if inspection.kind == .visionLanguage {
                vlmComponent = .languageModel
            }
            selectedStep = .dataset
        } catch {
            statusMessage = "Model inspection failed: \(error.localizedDescription)"
        }
        isWorking = false
    }

    func previewDataset() async {
        guard !datasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusMessage = "Choose a dataset path first."
            return
        }
        isWorking = true
        statusMessage = "Parsing dataset preview..."
        do {
            let preview: DatasetPreview = try await backend.runJSON(
                pythonOverride: backendPython,
                command: "preview-dataset",
                arguments: [
                    "--path", datasetPath,
                    "--limit", "12"
                ]
            )
            datasetPreview = preview
            statusMessage = "Previewed \(preview.validRows) valid rows, \(preview.malformedRows) malformed."
            if modelInspection?.kind == nil || modelInspection?.kind == .unknown {
                selectedStep = .configure
            }
        } catch {
            statusMessage = "Dataset preview failed: \(error.localizedDescription)"
        }
        isWorking = false
    }

    func startTraining() async {
        guard !isTraining else { return }
        logs.removeAll()
        metrics.removeAll()
        currentStep = 0
        totalSteps = steps
        lastOutputPath = nil
        isTraining = true
        statusMessage = dryRun ? "Running dry-run training command..." : "Training started..."

        let config = currentTrainingConfig()
        do {
            try await backend.streamTraining(
                pythonOverride: backendPython,
                config: config,
                onEvent: { [weak self] event in
                    Task { @MainActor in self?.handleTrainingEvent(event) }
                },
                onRawLine: { [weak self] line in
                    Task { @MainActor in self?.appendLog(line) }
                }
            )
            if isTraining {
                statusMessage = lastOutputPath == nil ? "Training finished." : "Training finished: \(lastOutputPath!)"
            }
        } catch {
            if isTraining {
                statusMessage = "Training failed: \(error.localizedDescription)"
                appendLog("error: \(error.localizedDescription)")
            }
        }
        isTraining = false
    }

    func stopTraining() {
        backend.stopTraining()
        statusMessage = "Stopping training..."
        appendLog("Stopping training process.")
        isTraining = false
    }

    func pushOutput() async {
        guard let output = lastOutputPath, !hfRepoId.isEmpty else {
            statusMessage = "Set a repo ID and finish a run before pushing."
            return
        }
        isWorking = true
        do {
            let status: BackendStatus = try await backend.runJSON(
                pythonOverride: backendPython,
                command: "push",
                arguments: ["--path", output, "--repo-id", hfRepoId]
            )
            statusMessage = status.message
        } catch {
            statusMessage = "Push failed: \(error.localizedDescription)"
        }
        isWorking = false
    }

    private func handleTrainingEvent(_ event: TrainingEvent) {
        if let message = event.message {
            appendLog(message)
        }
        if let step = event.step {
            currentStep = step
        }
        if let total = event.totalSteps {
            totalSteps = total
        }
        if let tps = event.tokensPerSecond {
            tokensPerSecond = tps
        }
        if let eta = event.etaSeconds {
            etaSeconds = eta
        }
        if let loss = event.loss, let step = event.step {
            metrics.append(TrainingMetric(step: step, loss: loss, tokensPerSecond: event.tokensPerSecond))
            if metrics.count > 240 {
                metrics.removeFirst(metrics.count - 240)
            }
        }
        if let output = event.outputPath {
            lastOutputPath = output
        }
        if event.type == "error" {
            statusMessage = event.message ?? "Training failed."
        }
    }

    private func appendLog(_ line: String) {
        logs.append(line)
        if logs.count > 1000 {
            logs.removeFirst(logs.count - 1000)
        }
    }

    private func currentTrainingConfig() -> TrainingConfig {
        TrainingConfig(
            modelId: modelId,
            modelDirectory: modelDirectory,
            datasetPath: datasetPath,
            backend: detectedBackend == .unknown ? .textLLM : detectedBackend,
            method: method,
            vlmComponent: vlmComponent,
            precision: precision,
            learningRate: learningRate,
            batchSize: batchSize,
            maxSeqLength: maxSeqLength,
            steps: steps,
            epochs: epochs,
            gradientAccumulation: gradientAccumulation,
            warmupSteps: warmupSteps,
            imageResolution: imageResolution,
            maxPixels: maxPixels,
            qloraBits: qloraBits,
            loraRank: loraRank,
            loraAlpha: loraAlpha,
            loraDropout: loraDropout,
            targetModules: targetModules,
            outputDirectory: outputDirectory,
            resumeAdapterPath: resumeAdapterPath,
            pushToHF: pushToHF,
            hfRepoId: hfRepoId,
            dryRun: dryRun
        )
    }
}
