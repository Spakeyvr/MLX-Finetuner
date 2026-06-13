import SwiftUI

struct ExportStepView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HeaderView(
                    title: "Export",
                    subtitle: "Use the saved adapter or full model locally, resume from a checkpoint, or push the output to Hugging Face."
                )

                FormSection(title: "Latest Output") {
                    InfoGrid(rows: [
                        ("Output", store.lastOutputPath ?? "No completed run yet"),
                        ("Method", store.method.title),
                        ("Backend", store.detectedBackend.title)
                    ])
                }

                FormSection(title: "Hugging Face") {
                    Toggle("Push after training", isOn: $store.pushToHF)
                    LabeledContent("Repo ID") {
                        TextField("username/model-name", text: $store.hfRepoId)
                            .textFieldStyle(.roundedBorder)
                    }
                    HStack {
                        Button {
                            Task { await store.pushOutput() }
                        } label: {
                            Label("Push Output", systemImage: "arrow.up.circle")
                        }
                        .disabled(store.lastOutputPath == nil || store.hfRepoId.isEmpty)
                        Spacer()
                    }
                }

                FormSection(title: "Resume") {
                    LabeledContent("Adapter checkpoint") {
                        PathPicker(path: $store.resumeAdapterPath, canChooseFiles: true, canChooseDirectories: false, allowEmpty: true)
                    }
                    Text("Resume passes the adapter path to mlx-lm or mlx-vlm on the next run.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(24)
            .frame(maxWidth: 920, alignment: .leading)
        }
    }
}

struct SettingsView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        Form {
            TextField("Python executable", text: $store.backendPython)
            Text("Leave blank to use MLX_FINETUNER_PYTHON or /usr/bin/python3.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .frame(width: 520)
    }
}
