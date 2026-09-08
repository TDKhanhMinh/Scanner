import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
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

  it("supports direct user typing into folder path input", () => {
    render(<App />);
    const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
    fireEvent.change(input, { target: { value: "C:\\attendance\\custom_folder" } });
    expect(input).toHaveValue("C:\\attendance\\custom_folder");
    expect(screen.getByLabelText("Thư mục xuất PDF")).toHaveValue(
      "C:\\attendance\\custom_folder_pdf",
    );
  });
});
