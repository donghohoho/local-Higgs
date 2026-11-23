import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "local-Higgs",
  description: "오픈소스 AI 영상 생성기",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
