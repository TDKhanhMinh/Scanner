import type {
  ExportMode,
  PlanSettings,
  ProductDetectorMode,
  ScanMode,
} from "@/types/scanner";

export const USER_PREFERENCES_STORAGE_KEY = "attendance-scanner.user-preferences.v1";
export const LEGACY_DETECTOR_STORAGE_KEY = "attendance-scanner.detector-settings.v1";

export interface UserPreferencesV1 {
  schemaVersion: 1;
  mode: ScanMode;
  detectorMode: ProductDetectorMode;
  exportMode: ExportMode;
  debugDiagnostics: boolean;
}

export const DEFAULT_USER_PREFERENCES: UserPreferencesV1 = {
  schemaVersion: 1,
  mode: "gray",
  detectorMode: "ai_enhanced",
  exportMode: "PER_IMAGE",
  debugDiagnostics: false,
};

function isScanMode(value: unknown): value is ScanMode {
  return (
    value === "gray" ||
    value === "bw" ||
    value === "color" ||
    value === "smart_document"
  );
}

function isProductDetectorMode(value: unknown): value is ProductDetectorMode {
  return value === "ai_enhanced" || value === "classic";
}

function isExportMode(value: unknown): value is ExportMode {
  return value === "PER_IMAGE" || value === "GROUPED";
}

export function loadUserPreferences(): UserPreferencesV1 {
  if (typeof window === "undefined" || !window.localStorage) {
    return { ...DEFAULT_USER_PREFERENCES };
  }

  try {
    const raw = window.localStorage.getItem(USER_PREFERENCES_STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        const rec = parsed as Record<string, unknown>;
        if (rec.schemaVersion === 1) {
          return {
            schemaVersion: 1,
            mode: isScanMode(rec.mode) ? rec.mode : DEFAULT_USER_PREFERENCES.mode,
            detectorMode: isProductDetectorMode(rec.detectorMode)
              ? rec.detectorMode
              : DEFAULT_USER_PREFERENCES.detectorMode,
            exportMode: isExportMode(rec.exportMode)
              ? rec.exportMode
              : DEFAULT_USER_PREFERENCES.exportMode,
            debugDiagnostics:
              typeof rec.debugDiagnostics === "boolean"
                ? rec.debugDiagnostics
                : DEFAULT_USER_PREFERENCES.debugDiagnostics,
          };
        }
      }
    }

    // Migration from legacy detector settings key
    const legacyRaw = window.localStorage.getItem(LEGACY_DETECTOR_STORAGE_KEY);
    if (legacyRaw) {
      const legacyParsed: unknown = JSON.parse(legacyRaw);
      if (typeof legacyParsed === "object" && legacyParsed !== null) {
        const legacyRec = legacyParsed as Record<string, unknown>;
        const migrated: UserPreferencesV1 = {
          ...DEFAULT_USER_PREFERENCES,
          detectorMode: isProductDetectorMode(legacyRec.detectorMode)
            ? legacyRec.detectorMode
            : DEFAULT_USER_PREFERENCES.detectorMode,
          debugDiagnostics:
            typeof legacyRec.debugDiagnostics === "boolean"
              ? legacyRec.debugDiagnostics
              : DEFAULT_USER_PREFERENCES.debugDiagnostics,
        };
        try {
          window.localStorage.setItem(
            USER_PREFERENCES_STORAGE_KEY,
            JSON.stringify(migrated)
          );
        } catch {
          // Ignore localStorage write failures during migration
        }
        return migrated;
      }
    }
  } catch {
    // Return safe fallback on JSON parse or storage access error
    return { ...DEFAULT_USER_PREFERENCES };
  }

  return { ...DEFAULT_USER_PREFERENCES };
}

export function saveUserPreferences(preferences: Partial<UserPreferencesV1>): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    const current = loadUserPreferences();
    const updated: UserPreferencesV1 = {
      schemaVersion: 1,
      mode: isScanMode(preferences.mode) ? preferences.mode : current.mode,
      detectorMode: isProductDetectorMode(preferences.detectorMode)
        ? preferences.detectorMode
        : current.detectorMode,
      exportMode: isExportMode(preferences.exportMode)
        ? preferences.exportMode
        : current.exportMode,
      debugDiagnostics:
        typeof preferences.debugDiagnostics === "boolean"
          ? preferences.debugDiagnostics
          : current.debugDiagnostics,
    };
    window.localStorage.setItem(
      USER_PREFERENCES_STORAGE_KEY,
      JSON.stringify(updated)
    );
  } catch {
    // Storage can be disabled by the host; in-memory settings still work.
  }
}

export function normalizePathForFingerprint(path: string): string {
  const trimmed = path.trim().replace(/\\/g, "/");
  if (/^[a-zA-Z]:\//.test(trimmed)) {
    return trimmed.toLowerCase();
  }
  return trimmed;
}

export function createPlanFingerprint(
  inputRoot: string,
  outputRoot: string,
  settings: PlanSettings
): string {
  return JSON.stringify({
    input: normalizePathForFingerprint(inputRoot),
    output: normalizePathForFingerprint(outputRoot),
    mode: settings.mode,
    detector: settings.detectorMode,
    year: settings.period.year,
    month: settings.period.month,
    export: settings.exportMode,
    reprocess: settings.reprocess,
  });
}
