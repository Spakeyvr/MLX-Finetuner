import Foundation

enum FlowStep: String, CaseIterable, Identifiable {
    case model
    case dataset
    case configure
    case train
    case export

    var id: String { rawValue }

    var title: String {
        switch self {
        case .model: "Model"
        case .dataset: "Dataset"
        case .configure: "Configure"
        case .train: "Train"
        case .export: "Export"
        }
    }

    var detail: String {
        switch self {
        case .model: "Hugging Face ID"
        case .dataset: "Preview samples"
        case .configure: "Method and limits"
        case .train: "Live metrics"
        case .export: "Adapters or model"
        }
    }

    var systemImage: String {
        switch self {
        case .model: "shippingbox"
        case .dataset: "folder"
        case .configure: "slider.horizontal.3"
        case .train: "waveform.path.ecg"
        case .export: "square.and.arrow.up"
        }
    }
}

enum ModelKind: String, Codable, CaseIterable {
    case unknown
    case textLLM = "text_llm"
    case visionLanguage = "vision_language"

    var title: String {
        switch self {
        case .unknown: "Unknown"
        case .textLLM: "Text LLM"
        case .visionLanguage: "Vision-language"
        }
    }
}

enum TrainingMethod: String, Codable, CaseIterable, Identifiable {
    case qlora
    case full

    var id: String { rawValue }
    var title: String { self == .qlora ? "QLoRA" : "Full-parameter" }
}

enum VLMComponent: String, Codable, CaseIterable, Identifiable {
    case languageModel = "language_model"
    case visionEncoder = "vision_encoder"
    case both

    var id: String { rawValue }
    var title: String {
        switch self {
        case .languageModel: "Language model"
        case .visionEncoder: "Vision encoder"
        case .both: "Both"
        }
    }
}

enum Precision: String, Codable, CaseIterable, Identifiable {
    case bf16
    case fp16

    var id: String { rawValue }
}

struct ModelInspection: Codable {
    var modelId: String
    var localPath: String?
    var kind: ModelKind
    var modelType: String?
    var architecture: String?
    var parameterEstimate: String?
    var warnings: [String]
}

struct DatasetPreview: Codable {
    var path: String
    var kind: ModelKind
    var totalRows: Int
    var validRows: Int
    var malformedRows: Int
    var examples: [PreviewExample]
    var issues: [String]
}

struct PreviewExample: Codable, Identifiable {
    var index: Int
    var format: String
    var prompt: String?
    var completion: String?
    var messagesSummary: String?
    var imageRefs: [ImageReference]
    var issues: [String]

    var id: Int { index }
}

struct ImageReference: Codable, Identifiable {
    var path: String?
    var url: String?
    var exists: Bool
    var width: Int?
    var height: Int?
    var thumbnailPath: String?
    var error: String?

    var id: String { path ?? url ?? UUID().uuidString }
}

struct TrainingMetric: Identifiable {
    let id = UUID()
    var step: Int
    var loss: Double
    var tokensPerSecond: Double?
}

struct TrainingEvent: Codable {
    var type: String
    var message: String?
    var level: String?
    var step: Int?
    var totalSteps: Int?
    var loss: Double?
    var tokensPerSecond: Double?
    var etaSeconds: Double?
    var outputPath: String?
}

struct BackendStatus: Codable {
    var ok: Bool
    var message: String
    var outputPath: String?
}

struct TrainingConfig: Codable {
    var modelId: String
    var modelDirectory: String
    var datasetPath: String
    var backend: ModelKind
    var method: TrainingMethod
    var vlmComponent: VLMComponent
    var precision: Precision
    var learningRate: Double
    var batchSize: Int
    var maxSeqLength: Int
    var steps: Int
    var epochs: Double
    var gradientAccumulation: Int
    var warmupSteps: Int
    var imageResolution: Int
    var maxPixels: Int
    var qloraBits: Int
    var loraRank: Int
    var loraAlpha: Int
    var loraDropout: Double
    var targetModules: String
    var outputDirectory: String
    var resumeAdapterPath: String
    var pushToHF: Bool
    var hfRepoId: String
    var dryRun: Bool
}
