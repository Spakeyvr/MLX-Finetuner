// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXFinetuner",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MLXFinetuner", targets: ["MLXFinetuner"])
    ],
    targets: [
        .executableTarget(
            name: "MLXFinetuner",
            path: "Sources/MLXFinetuner"
        )
    ]
)
