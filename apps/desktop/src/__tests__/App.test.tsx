import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import App from "../App";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

describe("App with shadcn UI tests", () => {
  it("renders the application heading and tab triggers", () => {
    render(<App />);
    const heading = screen.getByRole("heading", { name: /Attendance Scanner Desktop/i });
    expect(heading).toBeDefined();

    expect(screen.getByRole("tab", { name: /Cấu hình & Quét/i })).toBeDefined();
    expect(screen.getByRole("tab", { name: /Kết quả chi tiết/i })).toBeDefined();
  });

  it("renders Button component with variant styles", () => {
    render(<Button variant="default">Test Button</Button>);
    const btn = screen.getByRole("button", { name: /Test Button/i });
    expect(btn).toBeDefined();
    expect(btn.className).toContain("min-h-[44px]");
  });

  it("renders Badge component with status variants", () => {
    render(<Badge variant="success">Success Badge</Badge>);
    expect(screen.getByText("Success Badge")).toBeDefined();
  });

  it("simulates folder selection demo in App", () => {
    render(<App />);
    const selectBtn = screen.getByRole("button", { name: /Chọn thư mục/i });
    fireEvent.click(selectBtn);
    expect(screen.getAllByText(/D:\\ChamCong\\all/i).length).toBeGreaterThan(0);
  });
});
