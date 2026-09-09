import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { FileResultList } from "@/components/scanner/FileResultList";

describe("FileResultList", () => {
  it("formats clipped warnings and exposes the source preview action", () => {
    const onPreview = vi.fn();

    render(
      <FileResultList
        items={[
          {
            id: "clipped-1",
            relativePath: "NV01/clipped.jpg",
            employeeName: "NV01",
            sourceFile: "clipped.jpg",
            targetPdf: "NV01/clipped.pdf",
            status: "warning",
            documentDetected: false,
            message: "DOCUMENT_CLIPPED",
          },
        ]}
        onPreview={onPreview}
      />,
    );

    expect(
      screen.getByText("Ảnh chụp sát biên/thiếu góc — đã giữ nguyên ảnh gốc, chưa cắt gọt"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Chụp lại để lấy trọn 4 mép giấy/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Xem trước clipped\.jpg/i }));
    expect(onPreview).toHaveBeenCalledWith("NV01/clipped.jpg");
  });
});
