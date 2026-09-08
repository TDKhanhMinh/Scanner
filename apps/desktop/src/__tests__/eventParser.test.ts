import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseScannerEvent,
  parseScannerEvents,
  parseEventChunk,
} from "@/lib/eventParser";
import {
  PROTOCOL_VERSION,
  isScanPlanEvent,
  isFileStartedEvent,
  isFileCompletedEvent,
  isFileFailedEvent,
  isScanCompletedEvent,
} from "@/types/scanner";

describe("eventParser", () => {
  it("parses valid scan_plan event with System Design canonical fields", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "scan_plan",
      timestamp: "2026-09-07T00:00:00Z",
      inputRoot: "D:/ChamCong/all",
      outputRoot: "D:/ChamCong/output",
      employees: 12,
      totalImages: 48,
      new: 40,
      modified: 8,
      unchanged: 0,
      filesToProcess: 48,
      collisions: [],
      unsupportedCount: 2,
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    if (event && isScanPlanEvent(event)) {
      expect(event.type).toBe("scan_plan");
      expect(event.inputRoot).toBe("D:/ChamCong/all");
      expect(event.employees).toBe(12);
      expect(event.totalEmployees).toBe(12);
      expect(event.new).toBe(40);
      expect(event.newCount).toBe(40);
      expect(event.modified).toBe(8);
      expect(event.unchanged).toBe(0);
      expect(event.filesToProcess).toBe(48);
      expect(event.unsupportedCount).toBe(2);
    } else {
      throw new Error("Expected ScanPlanEvent");
    }
  });

  it("rejects a scan_plan with inconsistent process counters", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "scan_plan",
      timestamp: "2026-09-07T00:00:00Z",
      inputRoot: "D:/ChamCong/all",
      outputRoot: "D:/ChamCong/output",
      employees: 1,
      totalImages: 1,
      new: 1,
      modified: 0,
      rebuild: 0,
      unchanged: 0,
      filesToProcess: 0,
      collisions: [],
    });

    expect(parseScannerEvent(json)).toBeNull();
  });

  it("parses valid file_started event", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "file_started",
      timestamp: "2026-09-07T00:00:01Z",
      relativePath: "NguyenVanA/2026-03-01.jpg",
      employeeName: "NguyenVanA",
      index: 1,
      total: 48,
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    if (event && isFileStartedEvent(event)) {
      expect(event.type).toBe("file_started");
      expect(event.employeeName).toBe("NguyenVanA");
      expect(event.index).toBe(1);
      expect(event.total).toBe(48);
    } else {
      throw new Error("Expected FileStartedEvent");
    }
  });

  it("parses valid file_completed event", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "file_completed",
      timestamp: "2026-09-07T00:00:02Z",
      relativePath: "NguyenVanA/2026-03-01.jpg",
      employeeName: "NguyenVanA",
      outputRelativePath: "NguyenVanA/2026-03-01.pdf",
      documentDetected: true,
      warning: null,
      durationMs: 420,
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    if (event && isFileCompletedEvent(event)) {
      expect(event.type).toBe("file_completed");
      expect(event.documentDetected).toBe(true);
      expect(event.durationMs).toBe(420);
    } else {
      throw new Error("Expected FileCompletedEvent");
    }
  });

  it("parses valid file_failed event", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "file_failed",
      timestamp: "2026-09-07T00:00:03Z",
      relativePath: "TranVanB/corrupted.png",
      employeeName: "TranVanB",
      errorCode: "IMAGE_DECODE_FAILED",
      message: "Failed to decode image: corrupted header",
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    if (event && isFileFailedEvent(event)) {
      expect(event.type).toBe("file_failed");
      expect(event.errorCode).toBe("IMAGE_DECODE_FAILED");
      expect(event.message).toContain("corrupted");
    } else {
      throw new Error("Expected FileFailedEvent");
    }
  });

  it("parses valid scan_completed event", () => {
    const json = JSON.stringify({
      protocolVersion: PROTOCOL_VERSION,
      type: "scan_completed",
      timestamp: "2026-09-07T00:00:10Z",
      totalProcessed: 48,
      success: 47,
      failed: 1,
      warning: 0,
      skipped: 0,
      durationMs: 9500,
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    if (event && isScanCompletedEvent(event)) {
      expect(event.type).toBe("scan_completed");
      expect(event.totalProcessed).toBe(48);
      expect(event.success).toBe(47);
      expect(event.durationMs).toBe(9500);
    } else {
      throw new Error("Expected ScanCompletedEvent");
    }
  });

  it("safely ignores unknown extra fields for forward compatibility", () => {
    const json = JSON.stringify({
      protocolVersion: 1,
      type: "file_completed",
      timestamp: "2026-09-07T00:00:02Z",
      relativePath: "NV01/img.jpg",
      employeeName: "NV01",
      outputRelativePath: "NV01/img.pdf",
      documentDetected: false,
      durationMs: 120,
      futureMetricA: 99.5,
      futureMetadata: { isCloudSynced: false },
    });

    const event = parseScannerEvent(json);
    expect(event).not.toBeNull();
    expect(event?.type).toBe("file_completed");
    expect((event as unknown as Record<string, unknown>).futureMetricA).toBe(99.5);
  });

  it("strictly validates protocolVersion, timestamp, and enum errorCode (P2)", () => {
    const baseValid = {
      protocolVersion: 1,
      timestamp: "2026-09-07T00:00:00Z",
      type: "file_started",
      relativePath: "A/1.jpg",
      employeeName: "A",
      index: 1,
      total: 10,
    };

    // Missing protocolVersion
    const noProto = { ...baseValid };
    delete (noProto as Record<string, unknown>).protocolVersion;
    expect(parseScannerEvent(JSON.stringify(noProto))).toBeNull();

    // Invalid protocolVersion
    expect(
      parseScannerEvent(JSON.stringify({ ...baseValid, protocolVersion: 2 }))
    ).toBeNull();
    expect(
      parseScannerEvent(JSON.stringify({ ...baseValid, protocolVersion: "1" }))
    ).toBeNull();

    // Missing timestamp
    const noTs = { ...baseValid };
    delete (noTs as Record<string, unknown>).timestamp;
    expect(parseScannerEvent(JSON.stringify(noTs))).toBeNull();

    // Empty timestamp
    expect(
      parseScannerEvent(JSON.stringify({ ...baseValid, timestamp: "   " }))
    ).toBeNull();

    // Invalid errorCode
    const invalidErrEvent = {
      protocolVersion: 1,
      timestamp: "2026-09-07T00:00:00Z",
      type: "file_failed",
      relativePath: "A/1.jpg",
      employeeName: "A",
      errorCode: "NOT_AN_ENUM_ERROR_CODE",
      message: "some err",
    };
    expect(parseScannerEvent(JSON.stringify(invalidErrEvent))).toBeNull();

    // Incomplete counter fields in scan_plan
    const incompletePlan = {
      protocolVersion: 1,
      timestamp: "2026-09-07T00:00:00Z",
      type: "scan_plan",
      inputRoot: "/in",
      outputRoot: "/out",
      employees: 5,
      // missing totalImages, new, modified, unchanged
    };
    expect(parseScannerEvent(JSON.stringify(incompletePlan))).toBeNull();
  });

  it("returns null for malformed or empty inputs", () => {
    expect(parseScannerEvent("")).toBeNull();
    expect(parseScannerEvent("   \n  ")).toBeNull();
    expect(parseScannerEvent("not a json")).toBeNull();
    expect(parseScannerEvent("123")).toBeNull();
    expect(parseScannerEvent("[]")).toBeNull();
    expect(
      parseScannerEvent(
        JSON.stringify({
          protocolVersion: 1,
          timestamp: "2026-09-07T00:00:00Z",
          type: "unknown_type",
        })
      )
    ).toBeNull();
  });

  it("parses multiple lines using parseScannerEvents", () => {
    const lines = [
      "",
      JSON.stringify({
        protocolVersion: 1,
        type: "file_started",
        timestamp: "2026-09-07T00:00:01Z",
        relativePath: "A/1.jpg",
        employeeName: "A",
        index: 1,
        total: 2,
      }),
      "invalid-log-line-ignored",
      JSON.stringify({
        protocolVersion: 1,
        type: "file_started",
        timestamp: "2026-09-07T00:00:02Z",
        relativePath: "A/2.jpg",
        employeeName: "A",
        index: 2,
        total: 2,
      }),
      "   ",
    ];

    const parsed = parseScannerEvents(lines);
    expect(parsed).toHaveLength(2);
    expect(parsed[0].type).toBe("file_started");
    expect(parsed[1].type).toBe("file_started");
  });

  it("handles streaming chunks with parseEventChunk", () => {
    const part1 =
      '{"protocolVersion":1,"type":"file_started","timestamp":"t1","relativePath":"A/1.jpg","employeeName":"A","index":1,"total":2}\n{"protocolVersion":1,"type":"file_comp';
    const part2 =
      'leted","timestamp":"t2","relativePath":"A/1.jpg","employeeName":"A","outputRelativePath":"A/1.pdf","documentDetected":true,"durationMs":300}\n';

    const firstResult = parseEventChunk(part1, "");
    expect(firstResult.events).toHaveLength(1);
    expect(firstResult.events[0].type).toBe("file_started");
    expect(firstResult.remainder).toBe('{"protocolVersion":1,"type":"file_comp');

    const secondResult = parseEventChunk(part2, firstResult.remainder);
    expect(secondResult.events).toHaveLength(1);
    expect(secondResult.events[0].type).toBe("file_completed");
    expect(secondResult.remainder).toBe("");
  });

  it("loads and parses all JSON event fixture files", () => {
    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const fixturesDir = path.resolve(
      currentDir,
      "../../../../fixtures/scanner/events"
    );
    expect(fs.existsSync(fixturesDir)).toBe(true);

    const files = fs
      .readdirSync(fixturesDir)
      .filter((f: string) => f.endsWith(".json"));
    expect(files.length).toBeGreaterThanOrEqual(5);

    for (const file of files) {
      const raw = fs.readFileSync(path.join(fixturesDir, file), "utf-8");
      const event = parseScannerEvent(raw);
      expect(event).not.toBeNull();
      expect(event?.protocolVersion).toBe(1);
    }
  });
});
