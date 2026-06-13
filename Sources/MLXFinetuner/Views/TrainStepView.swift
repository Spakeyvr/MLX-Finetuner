import SwiftUI

struct TrainStepView: View {
    @ObservedObject var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    HeaderView(
                        title: "Train",
                        subtitle: "Watch loss, throughput, ETA, and logs while the Python backend runs. Stop sends a termination signal to the training process."
                    )

                    FormSection(title: "Live Metrics") {
                        HStack(spacing: 18) {
                            MetricTile(title: "Step", value: "\(store.currentStep)/\(max(store.totalSteps, store.steps))")
                            MetricTile(title: "Tokens/sec", value: store.tokensPerSecond.map { String(format: "%.1f", $0) } ?? "-")
                            MetricTile(title: "ETA", value: store.etaSeconds.map(formatDuration) ?? "-")
                            MetricTile(title: "Memory", value: store.memoryEstimate)
                        }
                        ProgressView(value: Double(store.currentStep), total: Double(max(store.totalSteps, store.steps)))
                        LossCurveView(metrics: store.metrics)
                            .frame(height: 220)
                    }

                    FormSection(title: "Run") {
                        HStack {
                            Button {
                                if store.isTraining {
                                    store.stopTraining()
                                } else {
                                    Task { await store.startTraining() }
                                }
                            } label: {
                                Label(store.isTraining ? "Stop Training" : "Start Training", systemImage: store.isTraining ? "stop.fill" : "play.fill")
                            }
                            .buttonStyle(.borderedProminent)

                            Toggle("Dry run", isOn: $store.dryRun)
                            Spacer()
                        }
                    }
                }
                .padding(24)
                .frame(maxWidth: 1040, alignment: .leading)
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Log")
                    .font(.headline)
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 3) {
                            ForEach(Array(store.logs.enumerated()), id: \.offset) { index, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(index)
                            }
                        }
                    }
                    .onChange(of: store.logs.count) { _, newValue in
                        if newValue > 0 {
                            proxy.scrollTo(newValue - 1, anchor: .bottom)
                        }
                    }
                }
                .frame(minHeight: 150, maxHeight: 220)
            }
            .padding(16)
        }
    }
}

struct LossCurveView: View {
    var metrics: [TrainingMetric]

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.quaternary.opacity(0.35))
                if metrics.count < 2 {
                    Text("Loss points will appear as training logs arrive.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Canvas { context, size in
                        let losses = metrics.map(\.loss)
                        let minLoss = losses.min() ?? 0
                        let maxLoss = losses.max() ?? 1
                        let span = max(0.0001, maxLoss - minLoss)
                        var path = Path()
                        for (index, metric) in metrics.enumerated() {
                            let x = size.width * CGFloat(index) / CGFloat(max(1, metrics.count - 1))
                            let normalized = (metric.loss - minLoss) / span
                            let y = size.height - CGFloat(normalized) * size.height
                            if index == 0 {
                                path.move(to: CGPoint(x: x, y: y))
                            } else {
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                        context.stroke(path, with: .color(.accentColor), lineWidth: 2)
                    }
                    .padding(12)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
    }
}

struct MetricTile: View {
    var title: String
    var value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}
