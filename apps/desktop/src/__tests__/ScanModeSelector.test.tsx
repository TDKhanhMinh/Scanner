import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { ScanModeSelector } from "@/components/scanner/ScanModeSelector";

describe("ScanModeSelector", () => {
  it("offers Smart Document alongside the existing scan modes", () => {
    const onSelectMode = vi.fn();

    render(<ScanModeSelector mode="gray" onSelectMode={onSelectMode} />);

    expect(screen.getAllByRole("button")).toHaveLength(4);
    expect(screen.getByRole("button", { name: /Smart Document/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));
    expect(onSelectMode).toHaveBeenCalledWith("smart_document");
  });
});
