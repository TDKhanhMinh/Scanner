import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import App from "../App";

describe("App foundation smoke test", () => {
  it("renders the application heading", () => {
    render(<App />);
    const heading = screen.getByRole("heading", { name: /Attendance Scanner Desktop/i });
    expect(heading).toBeDefined();
  });
});
