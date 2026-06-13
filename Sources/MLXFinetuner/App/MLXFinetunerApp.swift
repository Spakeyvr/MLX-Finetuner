import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct MLXFinetunerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup("MLX Finetuner", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 1080, minHeight: 720)
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("Preview Dataset") {
                    Task { await store.previewDataset() }
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])

                Button(store.isTraining ? "Stop Training" : "Start Training") {
                    if store.isTraining {
                        store.stopTraining()
                    } else {
                        Task { await store.startTraining() }
                    }
                }
                .keyboardShortcut(.return, modifiers: [.command])
            }
        }

        Settings {
            SettingsView(store: store)
        }
    }
}
