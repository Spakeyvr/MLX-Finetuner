import SwiftUI

struct DatasetStepView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HeaderView(
                    title: "Preview Dataset",
                    subtitle: "Choose a folder or .jsonl file. Text formats and VLM message blocks are normalized before training; bad rows are flagged here."
                )

                FormSection {
                    LabeledContent("Dataset path") {
                        PathPicker(path: $store.datasetPath, canChooseFiles: true, canChooseDirectories: true)
                    }
                    HStack {
                        Button {
                            Task { await store.previewDataset() }
                        } label: {
                            Label("Preview", systemImage: "eye")
                        }
                        .buttonStyle(.borderedProminent)
                        Spacer()
                    }
                }

                if let preview = store.datasetPreview {
                    FormSection(title: "Summary") {
                        InfoGrid(rows: [
                            ("Rows", "\(preview.totalRows)"),
                            ("Valid", "\(preview.validRows)"),
                            ("Malformed", "\(preview.malformedRows)"),
                            ("Detected type", preview.kind.title)
                        ])
                        if !preview.issues.isEmpty {
                            IssueList(issues: preview.issues)
                        }
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Examples")
                            .font(.headline)
                        ForEach(preview.examples) { example in
                            PreviewExampleRow(example: example)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 980, alignment: .leading)
        }
    }
}

struct PreviewExampleRow: View {
    var example: PreviewExample

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("#\(example.index + 1)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Text(example.format)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
                Spacer()
            }

            if let prompt = example.prompt {
                Text(prompt)
                    .font(.callout)
                    .lineLimit(4)
            }
            if let completion = example.completion {
                Text(completion)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
            }
            if let summary = example.messagesSummary {
                Text(summary)
                    .font(.callout)
                    .lineLimit(4)
            }

            if !example.imageRefs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(example.imageRefs) { image in
                            ImageThumb(reference: image)
                        }
                    }
                }
            }

            if !example.issues.isEmpty {
                IssueList(issues: example.issues)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

struct ImageThumb: View {
    var reference: ImageReference

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let thumbnail = reference.thumbnailPath, let nsImage = NSImage(contentsOfFile: thumbnail) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 96, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(.quaternary)
                    Image(systemName: reference.exists ? "photo" : "exclamationmark.triangle")
                        .foregroundStyle(reference.exists ? AnyShapeStyle(.secondary) : AnyShapeStyle(Color.orange))
                }
                .frame(width: 96, height: 72)
            }
            Text(reference.path ?? reference.url ?? "image")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: 96, alignment: .leading)
        }
    }
}
