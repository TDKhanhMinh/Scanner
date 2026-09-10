import type { ProductDetectorMode } from "@/types/scanner";

const STORAGE_KEY = "attendance-scanner.detector-settings.v1";

export interface DetectorPreferences {
  detectorMode: ProductDetectorMode;
  debugDiagnostics: boolean;
}

export const DEFAULT_DETECTOR_PREFERENCES: DetectorPreferences = {
  detectorMode: "ai_enhanced",
  debugDiagnostics: false,
};

function isProductDetectorMode(value: unknown): value is ProductDetectorMode {
  return value === "ai_enhanced" || value === "classic";
}

export function loadDetectorPreferences(): DetectorPreferences {
  if (typeof window === "undefined") return DEFAULT_DETECTOR_PREFERENCES;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_DETECTOR_PREFERENCES;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return DEFAULT_DETECTOR_PREFERENCES;
    const record = parsed as Record<string, unknown>;
    return {
      detectorMode: isProductDetectorMode(record.detectorMode)
        ? record.detectorMode
        : DEFAULT_DETECTOR_PREFERENCES.detectorMode,
      debugDiagnostics:
        typeof record.debugDiagnostics === "boolean"
          ? record.debugDiagnostics
          : DEFAULT_DETECTOR_PREFERENCES.debugDiagnostics,
    };
  } catch {
    return DEFAULT_DETECTOR_PREFERENCES;
  }
}

export function saveDetectorPreferences(preferences: DetectorPreferences): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Storage can be disabled by the host; in-memory settings still work.
  }
}
