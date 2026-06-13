import SwiftUI

struct ConfigureStepView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HeaderView(
                    title: "Configure",
                    subtitle: "Choose full fine-tuning or QLoRA explicitly. VLM runs can train the language model, vision encoder, or both."
                )

                FormSection(title: "Training Method") {
                    Picker("Method", selection: $store.method) {
                        ForEach(TrainingMethod.allCases) { method in
                            Text(method.title).tag(method)
                        }
                    }
                    .pickerStyle(.segmented)

                    if store.method == .full {
                        WarningStrip(text: "Full-parameter fine-tuning updates every weight and can exhaust unified memory quickly. Start with batch size 1 and short sequences.")
                        Picker("Precision", selection: $store.precision) {
                            ForEach(Precision.allCases) { precision in
                                Text(precision.rawValue).tag(precision)
                            }
                        }
                        Text("Estimated pressure: \(store.memoryEstimate)")
                            .foregroundStyle(.secondary)
                    } else {
                        Stepper("Quantization: \(store.qloraBits)-bit", value: $store.qloraBits, in: 2...8)
                        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 12) {
                            GridRow {
                                NumberField(label: "Rank", value: $store.loraRank, range: 1...256)
                                NumberField(label: "Alpha", value: $store.loraAlpha, range: 1...512)
                            }
                            GridRow {
                                DoubleField(label: "Dropout", value: $store.loraDropout, range: 0...0.9)
                                LabeledContent("Target modules") {
                                    TextField("q_proj,v_proj", text: $store.targetModules)
                                        .textFieldStyle(.roundedBorder)
                                }
                            }
                        }
                    }
                }

                if store.detectedBackend == .visionLanguage {
                    FormSection(title: "VLM Components") {
                        Picker("Train", selection: $store.vlmComponent) {
                            ForEach(VLMComponent.allCases) { component in
                                Text(component.title).tag(component)
                            }
                        }
                        .pickerStyle(.segmented)

                        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 12) {
                            GridRow {
                                NumberField(label: "Image resolution", value: $store.imageResolution, range: 128...4096)
                                NumberField(label: "Max pixels", value: $store.maxPixels, range: 65_536...16_777_216)
                            }
                        }
                    }
                }

                FormSection(title: "Hyperparameters") {
                    Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 12) {
                        GridRow {
                            ScientificField(label: "Learning rate", value: $store.learningRate)
                            NumberField(label: "Batch size", value: $store.batchSize, range: 1...128)
                        }
                        GridRow {
                            NumberField(label: "Max sequence", value: $store.maxSeqLength, range: 128...131_072)
                            NumberField(label: "Steps", value: $store.steps, range: 1...1_000_000)
                        }
                        GridRow {
                            DoubleField(label: "Epochs", value: $store.epochs, range: 0...100)
                            NumberField(label: "Grad accumulation", value: $store.gradientAccumulation, range: 1...256)
                        }
                        GridRow {
                            NumberField(label: "Warmup steps", value: $store.warmupSteps, range: 0...100_000)
                            Toggle("Dry run", isOn: $store.dryRun)
                        }
                    }
                }

                FormSection(title: "Output") {
                    LabeledContent("Output directory") {
                        PathPicker(path: $store.outputDirectory, canChooseFiles: false, canChooseDirectories: true)
                    }
                    LabeledContent("Resume adapter") {
                        PathPicker(path: $store.resumeAdapterPath, canChooseFiles: true, canChooseDirectories: false, allowEmpty: true)
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 980, alignment: .leading)
        }
    }
}
