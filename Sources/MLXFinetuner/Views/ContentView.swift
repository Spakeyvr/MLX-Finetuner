import SwiftUI

struct ContentView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        NavigationSplitView {
            SidebarView(selection: $store.selectedStep)
        } detail: {
            VStack(spacing: 0) {
                detailView
                Divider()
                StatusBarView(store: store)
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button {
                    Task { await store.inspectModel(download: false) }
                } label: {
                    Label("Inspect", systemImage: "magnifyingglass")
                }
                .help("Inspect model configuration")
                .disabled(store.isWorking || store.modelId.isEmpty)

                Button {
                    Task { await store.previewDataset() }
                } label: {
                    Label("Preview", systemImage: "eye")
                }
                .help("Preview dataset")
                .disabled(store.isWorking || store.datasetPath.isEmpty)

                Button {
                    if store.isTraining {
                        store.stopTraining()
                    } else {
                        Task { await store.startTraining() }
                    }
                } label: {
                    Label(store.isTraining ? "Stop" : "Train", systemImage: store.isTraining ? "stop.fill" : "play.fill")
                }
                .help(store.isTraining ? "Stop training" : "Start training")
                .disabled(!store.isTraining && (store.modelId.isEmpty || store.datasetPath.isEmpty))
            }
        }
        .task {
            await store.prepareBackendEnvironment()
        }
    }

    @ViewBuilder
    private var detailView: some View {
        switch store.selectedStep {
        case .model:
            ModelStepView(store: store)
        case .dataset:
            DatasetStepView(store: store)
        case .configure:
            ConfigureStepView(store: store)
        case .train:
            TrainStepView(store: store)
        case .export:
            ExportStepView(store: store)
        }
    }
}

struct SidebarView: View {
    @Binding var selection: FlowStep

    var body: some View {
        List(selection: $selection) {
            Section("Fine-tune") {
                ForEach(FlowStep.allCases) { step in
                    HStack(spacing: 10) {
                        Image(systemName: step.systemImage)
                            .foregroundStyle(.secondary)
                            .frame(width: 18)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(step.title)
                                .lineLimit(1)
                            Text(step.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .tag(step)
                }
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("MLX Finetuner")
    }
}

struct StatusBarView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        HStack(spacing: 12) {
            if store.isWorking {
                ProgressView()
                    .controlSize(.small)
            }
            Text(store.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
            Text(store.detectedBackend.title)
                .font(.caption)
                .foregroundStyle(.secondary)
            if store.isTraining {
                Text("\(store.currentStep)/\(store.totalSteps)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}
