import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { DetectorModeSelector } from "@/components/scanner/DetectorModeSelector";

describe("DetectorModeSelector", () => {
  it("defaults to AI Enhanced without exposing model names", () => {
    render(<DetectorModeSelector mode="ai_enhanced" onSelectMode={vi.fn()} />);

    expect(screen.getByRole("button", { name: /AI Enhanced/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByText(/Không cần chọn tên model/i)).toBeInTheDocument();
    expect(screen.queryByText(/MobileNet|DeepLab|DocAligner/i)).not.toBeInTheDocument();
  });

  it("selects Classic as the explicit fallback", () => {
    const onSelectMode = vi.fn();
    render(<DetectorModeSelector mode="ai_enhanced" onSelectMode={onSelectMode} />);

    fireEvent.click(screen.getByRole("button", { name: /Classic/i }));
    expect(onSelectMode).toHaveBeenCalledWith("classic");
  });
});
