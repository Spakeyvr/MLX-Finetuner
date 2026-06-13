import SwiftUI

struct ModelStepView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HeaderView(
                    title: "Pick Model",
                    subtitle: "Paste a Hugging Face model ID or local path. The backend reads config.json and routes text LLMs to mlx-lm and VLMs to mlx-vlm."
                )

                FormSection {
                    LabeledContent("Model ID") {
                        TextField("Qwen/Qwen2.5-0.5B", text: $store.modelId)
                            .textFieldStyle(.roundedBorder)
                    }
                    LabeledContent("Download directory") {
                        PathPicker(path: $store.modelDirectory, canChooseFiles: false, canChooseDirectories: true)
                    }
                    HStack {
                        Button {
                            Task { await store.inspectModel(download: false) }
                        } label: {
                            Label("Inspect", systemImage: "magnifyingglass")
                        }
                        .keyboardShortcut("i", modifiers: [.command])

                        Button {
                            Task { await store.inspectModel(download: true) }
                        } label: {
                            Label("Download", systemImage: "arrow.down.circle")
                        }
                        .buttonStyle(.borderedProminent)

                        Spacer()
                    }
                }

                if let inspection = store.modelInspection {
                    FormSection(title: "Detection") {
                        InfoGrid(rows: [
                            ("Backend", inspection.kind.title),
                            ("Model type", inspection.modelType ?? "Unknown"),
                            ("Architecture", inspection.architecture ?? "Unknown"),
                            ("Local path", inspection.localPath ?? "Not downloaded"),
                            ("Estimate", inspection.parameterEstimate ?? "Unavailable")
                        ])
                        if !inspection.warnings.isEmpty {
                            IssueList(issues: inspection.warnings)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 920, alignment: .leading)
        }
    }
}
