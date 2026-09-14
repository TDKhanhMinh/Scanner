import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_USER_PREFERENCES,
  LEGACY_DETECTOR_STORAGE_KEY,
  USER_PREFERENCES_STORAGE_KEY,
  createPlanFingerprint,
  loadUserPreferences,
  normalizePathForFingerprint,
  saveUserPreferences,
} from "@/lib/userPreferences";

describe("userPreferences", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns default preferences when storage is empty", () => {
    const preferences = loadUserPreferences();
    expect(preferences).toEqual(DEFAULT_USER_PREFERENCES);
    expect(preferences.schemaVersion).toBe(1);
    expect(preferences.mode).toBe("gray");
    expect(preferences.detectorMode).toBe("ai_enhanced");
    expect(preferences.exportMode).toBe("PER_IMAGE");
    expect(preferences.debugDiagnostics).toBe(false);
  });

  it("loads valid v1 preferences from storage", () => {
    window.localStorage.setItem(
      USER_PREFERENCES_STORAGE_KEY,
      JSON.stringify({
        schemaVersion: 1,
        mode: "smart_document",
        detectorMode: "classic",
        exportMode: "GROUPED",
        debugDiagnostics: true,
      }),
    );

    const loaded = loadUserPreferences();
    expect(loaded).toEqual({
      schemaVersion: 1,
      mode: "smart_document",
      detectorMode: "classic",
      exportMode: "GROUPED",
      debugDiagnostics: true,
    });
  });

  it("migrates legacy detector settings from v1 detector key", () => {
    window.localStorage.setItem(
      LEGACY_DETECTOR_STORAGE_KEY,
      JSON.stringify({
        detectorMode: "classic",
        debugDiagnostics: true,
      }),
    );

    const migrated = loadUserPreferences();
    expect(migrated.detectorMode).toBe("classic");
    expect(migrated.debugDiagnostics).toBe(true);
    expect(migrated.mode).toBe(DEFAULT_USER_PREFERENCES.mode);
    expect(migrated.exportMode).toBe(DEFAULT_USER_PREFERENCES.exportMode);
    expect(migrated.schemaVersion).toBe(1);

    // Verify it was written back to the new storage key
    const rawNew = window.localStorage.getItem(USER_PREFERENCES_STORAGE_KEY);
    expect(rawNew).not.toBeNull();
    const parsed = JSON.parse(rawNew!);
    expect(parsed.schemaVersion).toBe(1);
    expect(parsed.detectorMode).toBe("classic");
    expect(parsed.debugDiagnostics).toBe(true);
  });

  it("safely falls back to defaults when storage contains corrupted JSON or invalid fields", () => {
    window.localStorage.setItem(USER_PREFERENCES_STORAGE_KEY, "{corrupted-json...");
    expect(loadUserPreferences()).toEqual(DEFAULT_USER_PREFERENCES);

    window.localStorage.setItem(
      USER_PREFERENCES_STORAGE_KEY,
      JSON.stringify({
        schemaVersion: 1,
        mode: "non_existent_mode",
        detectorMode: "invalid_detector",
        exportMode: "INVALID_EXPORT",
        debugDiagnostics: "not_a_boolean",
      }),
    );
    expect(loadUserPreferences()).toEqual(DEFAULT_USER_PREFERENCES);
  });

  it("persists partial updates without overwriting existing fields", () => {
    saveUserPreferences({ mode: "bw", exportMode: "GROUPED" });

    const loaded = loadUserPreferences();
    expect(loaded.mode).toBe("bw");
    expect(loaded.exportMode).toBe("GROUPED");
    expect(loaded.detectorMode).toBe(DEFAULT_USER_PREFERENCES.detectorMode);
    expect(loaded.debugDiagnostics).toBe(DEFAULT_USER_PREFERENCES.debugDiagnostics);
  });

  it("normalizes paths for fingerprint deterministically", () => {
    expect(normalizePathForFingerprint("C:\\Attendance\\Input")).toBe("c:/attendance/input");
    expect(normalizePathForFingerprint("D:/Scanner/data")).toBe("d:/scanner/data");
    expect(normalizePathForFingerprint("./relative/path")).toBe("./relative/path");
  });

  it("generates different fingerprints when any setting changes", () => {
    const baseSettings = {
      mode: "gray" as const,
      detectorMode: "ai_enhanced" as const,
      period: { year: 2026, month: 9 },
      exportMode: "PER_IMAGE" as const,
      reprocess: false,
    };

    const fp1 = createPlanFingerprint("C:/Input", "C:/Output", baseSettings);
    const fp2 = createPlanFingerprint("C:/Input", "C:/Output", {
      ...baseSettings,
      mode: "smart_document",
    });
    const fp3 = createPlanFingerprint("C:/Input", "C:/Output", {
      ...baseSettings,
      period: { year: 2026, month: 10 },
    });
    const fp4 = createPlanFingerprint("C:/Input", "C:/Output", {
      ...baseSettings,
      exportMode: "GROUPED",
    });
    const fp5 = createPlanFingerprint("C:/Input", "C:/Output", {
      ...baseSettings,
      reprocess: true,
    });
    const fpBackslash = createPlanFingerprint("c:\\input", "c:\\output", baseSettings);

    expect(fp1).not.toBe(fp2);
    expect(fp1).not.toBe(fp3);
    expect(fp1).not.toBe(fp4);
    expect(fp1).not.toBe(fp5);
    // Backslash and lowercase Windows path should match normalized forward-slash
    expect(fp1).toBe(fpBackslash);
  });
});
