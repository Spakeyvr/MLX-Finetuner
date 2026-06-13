import AppKit
import SwiftUI

struct HeaderView: View {
    var title: String
    var subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.largeTitle.weight(.semibold))
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct FormSection<Content: View>: View {
    var title: String?
    @ViewBuilder var content: Content

    init(title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let title {
                Text(title)
                    .font(.headline)
            }
            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

struct InfoGrid: View {
    var rows: [(String, String)]

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
            ForEach(rows, id: \.0) { key, value in
                GridRow {
                    Text(key)
                        .foregroundStyle(.secondary)
                    Text(value)
                        .textSelection(.enabled)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)
                }
            }
        }
    }
}

struct IssueList: View {
    var issues: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(issues, id: \.self) { issue in
                Label(issue, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
            }
        }
    }
}

struct WarningStrip: View {
    var text: String

    var body: some View {
        Label(text, systemImage: "memorychip")
            .font(.callout)
            .foregroundStyle(.orange)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
    }
}

struct PathPicker: View {
    @Binding var path: String
    var canChooseFiles: Bool
    var canChooseDirectories: Bool
    var allowEmpty = false

    var body: some View {
        HStack {
            TextField(allowEmpty ? "Optional path" : "Path", text: $path)
                .textFieldStyle(.roundedBorder)
            Button {
                choosePath()
            } label: {
                Image(systemName: "folder")
            }
            .help("Choose path")
        }
    }

    private func choosePath() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = canChooseFiles
        panel.canChooseDirectories = canChooseDirectories
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = canChooseDirectories
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

struct NumberField: View {
    var label: String
    @Binding var value: Int
    var range: ClosedRange<Int>

    var body: some View {
        LabeledContent(label) {
            Stepper(value: $value, in: range) {
                TextField(label, value: $value, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
            }
        }
    }
}

struct DoubleField: View {
    var label: String
    @Binding var value: Double
    var range: ClosedRange<Double>

    var body: some View {
        LabeledContent(label) {
            TextField(label, value: $value, format: .number.precision(.fractionLength(0...4)))
                .textFieldStyle(.roundedBorder)
                .frame(width: 120)
                .onChange(of: value) { _, newValue in
                    value = min(max(newValue, range.lowerBound), range.upperBound)
                }
        }
    }
}

struct ScientificField: View {
    var label: String
    @Binding var value: Double

    var body: some View {
        LabeledContent(label) {
            TextField(label, value: $value, format: .number.notation(.scientific))
                .textFieldStyle(.roundedBorder)
                .frame(width: 140)
        }
    }
}

func formatDuration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "-" }
    let total = Int(seconds.rounded())
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let secs = total % 60
    if hours > 0 {
        return "\(hours)h \(minutes)m"
    }
    if minutes > 0 {
        return "\(minutes)m \(secs)s"
    }
    return "\(secs)s"
}
