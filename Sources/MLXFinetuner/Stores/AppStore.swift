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
    @Published var gradCheckpoint = false
    @Published var warmupSteps = 0
    @Published var validationSplitPercent = 10
    @Published var imageResolution = 768
    @Published var maxPixels = 1_048_576
    @Published var qloraBits = 4
    @Published var loraLayers = 16
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
    private var lastConfigureDefaultsKey = ""

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

    var logText: String {
        logs.joined(separator: "\n")
    }

    var configureCompatibilitySummary: String? {
        guard configureRecommendedSettings() != nil else { return nil }
        let normalized = normalizedModelReference(
            [
                modelId,
                modelInspection?.family,
                modelInspection?.modelType,
                modelInspection?.architecture
            ]
                .compactMap { $0 }
                .joined(separator: " ")
        )
        if normalized.contains("qwen35") || normalized.contains("qwen36") {
            return "Qwen3.5/3.6 defaults active: VLM backend, QLoRA, checkpointing, and hybrid LoRA targets."
        }
        return "Qwen defaults active: QLoRA, checkpointing, batch size 1, and Qwen attention LoRA targets."
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
            applyRecommendedSettings(inspection.recommendedSettings)
            selectedStep = .dataset
        } catch {
            statusMessage = "Model inspection failed: \(error.localizedDescription)"
        }
        isWorking = false
    }

    private func applyRecommendedSettings(_ settings: ModelRecommendedSettings?) {
        guard let settings else { return }
        if let method = settings.method {
            self.method = method
        }
        if let vlmComponent = settings.vlmComponent {
            self.vlmComponent = vlmComponent
        }
        if let batchSize = settings.batchSize {
            self.batchSize = batchSize
        }
        if let maxSeqLength = settings.maxSeqLength, self.maxSeqLength <= 2048 {
            self.maxSeqLength = maxSeqLength
        }
        if let gradientAccumulation = settings.gradientAccumulation {
            self.gradientAccumulation = gradientAccumulation
        }
        if let gradCheckpoint = settings.gradCheckpoint {
            self.gradCheckpoint = gradCheckpoint
        }
        if let imageResolution = settings.imageResolution {
            self.imageResolution = imageResolution
        }
        if let maxPixels = settings.maxPixels {
            self.maxPixels = maxPixels
        }
        if let qloraBits = settings.qloraBits {
            self.qloraBits = qloraBits
        }
        if let loraLayers = settings.loraLayers {
            self.loraLayers = loraLayers
        }
        if let loraRank = settings.loraRank {
            self.loraRank = loraRank
        }
        if let loraAlpha = settings.loraAlpha {
            self.loraAlpha = loraAlpha
        }
        if let loraDropout = settings.loraDropout {
            self.loraDropout = loraDropout
        }
        if let targetModules = settings.targetModules, self.targetModules == "q_proj,v_proj" || self.targetModules.isEmpty {
            self.targetModules = targetModules
        }
    }

    func applyConfigureModelDefaults(force: Bool = false) {
        let key = "\(modelId)|\(modelInspection?.family ?? "")|\(modelInspection?.kind.rawValue ?? "")"
        guard (force || lastConfigureDefaultsKey != key), let settings = configureRecommendedSettings() else { return }
        applyRecommendedSettings(settings)
        lastConfigureDefaultsKey = key
        statusMessage = "Applied Qwen-compatible Configure defaults."
    }

    private func configureRecommendedSettings() -> ModelRecommendedSettings? {
        if let settings = modelInspection?.recommendedSettings, hasRecommendedSettings(settings) {
            return settings
        }

        let normalized = normalizedModelReference(
            [
                modelId,
                modelInspection?.family,
                modelInspection?.modelType,
                modelInspection?.architecture
            ]
                .compactMap { $0 }
                .joined(separator: " ")
        )
        guard normalized.contains("qwen") else { return nil }

        let isQwenHybrid = normalized.contains("qwen35") || normalized.contains("qwen36")
        let isQwenHybridMoE = isQwenHybrid && (normalized.contains("moe") || normalized.range(of: #"a\d+b"#, options: .regularExpression) != nil)
        let isVLM = detectedBackend == .visionLanguage || isQwenHybrid || normalized.contains("vl")

        if isQwenHybrid {
            return ModelRecommendedSettings(
                backend: .visionLanguage,
                method: .qlora,
                vlmComponent: .languageModel,
                batchSize: 1,
                maxSeqLength: isQwenHybridMoE ? 2048 : 4096,
                gradientAccumulation: 1,
                gradCheckpoint: true,
                imageResolution: 768,
                maxPixels: 1_048_576,
                qloraBits: 4,
                loraLayers: isQwenHybridMoE ? 8 : 16,
                loraRank: 8,
                loraAlpha: 16,
                loraDropout: 0.0,
                targetModules: qwenHybridTargetModules
            )
        }

        return ModelRecommendedSettings(
            backend: isVLM ? .visionLanguage : nil,
            method: .qlora,
            vlmComponent: isVLM ? .languageModel : nil,
            batchSize: 1,
            maxSeqLength: nil,
            gradientAccumulation: 1,
            gradCheckpoint: true,
            imageResolution: isVLM ? 768 : nil,
            maxPixels: isVLM ? 1_048_576 : nil,
            qloraBits: 4,
            loraLayers: 16,
            loraRank: 8,
            loraAlpha: 16,
            loraDropout: 0.0,
            targetModules: qwenTargetModules
        )
    }

    private func hasRecommendedSettings(_ settings: ModelRecommendedSettings) -> Bool {
        settings.backend != nil ||
            settings.method != nil ||
            settings.vlmComponent != nil ||
            settings.batchSize != nil ||
            settings.maxSeqLength != nil ||
            settings.gradientAccumulation != nil ||
            settings.gradCheckpoint != nil ||
            settings.imageResolution != nil ||
            settings.maxPixels != nil ||
            settings.qloraBits != nil ||
            settings.loraLayers != nil ||
            settings.loraRank != nil ||
            settings.loraAlpha != nil ||
            settings.loraDropout != nil ||
            settings.targetModules != nil
    }

    private func normalizedModelReference(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    private var qwenTargetModules: String {
        "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
    }

    private var qwenHybridTargetModules: String {
        [
            qwenTargetModules,
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj"
        ].joined(separator: ",")
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
        applyConfigureModelDefaults(force: true)
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
            gradCheckpoint: gradCheckpoint,
            warmupSteps: warmupSteps,
            validationSplitPercent: validationSplitPercent,
            imageResolution: imageResolution,
            maxPixels: maxPixels,
            qloraBits: qloraBits,
            loraLayers: loraLayers,
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
