// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg(target_os = "windows")]
fn hide_helper_windows() {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetClassNameW, GetWindowThreadProcessId, ShowWindow, SW_HIDE,
    };

    unsafe extern "system" fn enum_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let target_pid = lparam as u32;
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == target_pid {
            let mut class_name = [0u16; 256];
            let len = GetClassNameW(hwnd, class_name.as_mut_ptr(), 256);
            if len > 0 {
                let class_str = String::from_utf16_lossy(&class_name[..len as usize]);
                if class_str.contains("UAC") || class_str.contains("Tao Thread") {
                    ShowWindow(hwnd, SW_HIDE);
                }
            }
        }
        1
    }

    let pid = std::process::id();
    unsafe {
        EnumWindows(Some(enum_windows_proc), pid as isize);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }

            #[cfg(target_os = "windows")]
            {
                // Run window sanitizer to guarantee Process.MainWindowTitle matches primary window
                hide_helper_windows();
                std::thread::spawn(|| {
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    hide_helper_windows();
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    hide_helper_windows();
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
