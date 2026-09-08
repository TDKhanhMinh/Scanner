import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { PageOrderReviewPanel } from "@/components/scanner/PageOrderReviewPanel";
import type { ReviewGroup } from "@/types/scanner";

const reviewGroup: ReviewGroup = {
  key: { employeeRelativeDir: "NV01", year: 2026, month: 9 },
  sourcePages: [
    {
      sourceRelativePath: "NV01/page-one-random.png",
      identity: { pageType: "UNKNOWN", pageOrder: null, confidence: 0.42 },
    },
    {
      sourceRelativePath: "NV01/page-two-random.png",
      identity: { pageType: "UNKNOWN", pageOrder: null, confidence: 0.4 },
    },
  ],
  reasons: ["unknown_page_identity", "duplicate_page_order"],
  completenessStatus: "AMBIGUOUS",
  reviewRequired: true,
  manualOrder: [],
};

describe("PageOrderReviewPanel", () => {
  it("supports accessible reordering and only resolves on explicit confirmation", () => {
    const onResolve = vi.fn();
    const onPreview = vi.fn();
    render(
      <PageOrderReviewPanel
        groups={[reviewGroup]}
        onResolve={onResolve}
        onSkip={vi.fn()}
        onPreview={onPreview}
      />,
    );

    expect(screen.getByText(/Không nhận diện chắc chắn loại page/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Xem preview NV01\/page-one-random\.png/i }));
    expect(onPreview).toHaveBeenCalledWith("NV01/page-one-random.png");
    fireEvent.click(screen.getByRole("button", { name: /Đưa NV01\/page-one-random\.png xuống sau/i }));
    expect(onResolve).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Xác nhận thứ tự page/i }));
    expect(onResolve).toHaveBeenCalledWith("NV01:2026-09", [
      "NV01/page-two-random.png",
      "NV01/page-one-random.png",
    ]);
  });

  it("allows skipping incomplete groups and blocks an invalid resolve", () => {
    const onResolve = vi.fn();
    const onSkip = vi.fn();
    render(
      <PageOrderReviewPanel
        groups={[{ ...reviewGroup, completenessStatus: "INCOMPLETE" }]}
        onResolve={onResolve}
        onSkip={onSkip}
      />,
    );

    expect(screen.getByRole("button", { name: /Xác nhận thứ tự page/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /Bỏ qua group này/i }));
    expect(onSkip).toHaveBeenCalledWith("NV01:2026-09");
    expect(onResolve).not.toHaveBeenCalled();
  });
});
