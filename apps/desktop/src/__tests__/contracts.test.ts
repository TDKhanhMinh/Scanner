import { describe, expect, it } from "vitest";
import {
  VALID_COMPLETENESS_STATUSES,
  VALID_EXPORT_MODES,
  VALID_PAGE_TYPES,
  type DocumentGroup,
} from "@/types/scanner";

describe("shared document contracts", () => {
  it("keeps period/export/group enum values stable across JSON round trips", () => {
    const group: DocumentGroup = {
      key: { employeeRelativeDir: "NV01", year: 2026, month: 9 },
      sourcePages: [
        {
          sourceRelativePath: "NV01/random-page.png",
          identity: {
            pageType: "UNKNOWN",
            pageOrder: null,
            confidence: null,
            detectionMethod: "manual_review",
          },
        },
      ],
      completenessStatus: "AMBIGUOUS",
      reviewRequired: true,
    };

    const restored = JSON.parse(JSON.stringify(group)) as DocumentGroup;

    expect(VALID_EXPORT_MODES).toEqual(["PER_IMAGE", "GROUPED"]);
    expect(VALID_PAGE_TYPES).toEqual(["FIRST_HALF", "SECOND_HALF", "UNKNOWN"]);
    expect(VALID_COMPLETENESS_STATUSES).toEqual([
      "COMPLETE",
      "INCOMPLETE",
      "AMBIGUOUS",
    ]);
    expect(restored).toEqual(group);
  });

  it("does not treat a source filename as document identity", () => {
    const group: DocumentGroup = {
      key: { employeeRelativeDir: "NV01", year: 2026, month: 9 },
      sourcePages: [
        {
          sourceRelativePath: "NV01/random-a.png",
          identity: { pageType: "FIRST_HALF", pageOrder: 1 },
        },
      ],
      completenessStatus: "COMPLETE",
      reviewRequired: false,
    };

    expect(group.key.employeeRelativeDir).toBe("NV01");
    expect(group.sourcePages[0].sourceRelativePath).toBe("NV01/random-a.png");
    expect(group.key).not.toHaveProperty("sourceFilename");
  });
});
