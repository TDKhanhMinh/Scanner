import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it } from "vitest";
import {
  DetectionPreviewPanel,
  previewPointToSvgPoint,
} from "@/components/scanner/DetectionPreviewPanel";
import type { FileResultItem } from "@/components/scanner/FileResultList";
import type { DetectionPreview } from "@/types/scanner";

const preview: DetectionPreview = {
  sourceWidth: 2000,
  sourceHeight: 1000,
  coordinateSpace: "original_pixels",
  finalCorners: [
    { x: 100, y: 50 },
    { x: 1900, y: 70 },
    { x: 1850, y: 950 },
    { x: 120, y: 930 },
  ],
  candidateCorners: [
    {
      source: "mask_fit",
      confidence: 0.86,
      corners: [
        { x: 110, y: 60 },
        { x: 1890, y: 80 },
        { x: 1840, y: 940 },
        { x: 130, y: 920 },
      ],
    },
  ],
  maskAvailable: true,
  previewImageDataUrl: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP",
  confidence: 0.91,
  confidenceIsCalibrated: false,
  fallbackUsed: false,
  detectorName: "hybrid",
  modelVersion: "configured-v1",
  reasonCodes: [],
  warningCodes: [],
};

const item: FileResultItem = {
  id: "preview-1",
  relativePath: "NV01/page.jpg",
  employeeName: "NV01",
  sourceFile: "page.jpg",
  targetPdf: "NV01/page.pdf",
  status: "success",
  documentDetected: true,
  detectionPreview: preview,
};

describe("DetectionPreviewPanel", () => {
  it("scales source coordinates into the overlay coordinate system", () => {
    expect(previewPointToSvgPoint({ x: 1000, y: 500 }, preview)).toBe("50.000,50.000");
    expect(previewPointToSvgPoint({ x: -10, y: 1100 }, preview)).toBe("0.000,100.000");
  });

  it("renders final/candidate overlays and avoids probability language", () => {
    const { container } = render(
      <DetectionPreviewPanel item={item} onClose={() => undefined} />,
    );

    expect(container.querySelector('svg[viewBox="0 0 100 100"]')).toBeInTheDocument();
    expect(container.querySelectorAll("polygon")).toHaveLength(2);
    expect(screen.getByText(/không phải xác suất/i)).toBeInTheDocument();
    expect(screen.getByText(/mask evidence có sẵn/i)).toBeInTheDocument();
  });

  it("makes full-image fallback explicit when no final corners exist", () => {
    render(
      <DetectionPreviewPanel
        item={{
          ...item,
          status: "warning",
          documentDetected: false,
          detectionPreview: {
            ...preview,
            finalCorners: null,
            fallbackUsed: true,
            reasonCode: "CV_NO_CANDIDATE",
            reasonCodes: ["CV_NO_CANDIDATE"],
            warningCodes: ["DOCUMENT_NOT_DETECTED"],
          },
        }}
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText(/Fallback ảnh đầy đủ, không auto-crop/i)).toBeInTheDocument();
    expect(screen.getByText(/giữ nguyên ảnh để tránh cắt nhầm/i)).toBeInTheDocument();
  });
});
