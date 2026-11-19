import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "../globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "mlxcel | High-Performance LLM Inference on Apple Silicon",
  description:
    "Run 100+ LLM and VLM models natively on Apple Silicon with Metal acceleration. Rust-powered, zero Python dependencies, OpenAI-compatible API.",
  keywords: [
    "LLM",
    "VLM",
    "Apple Silicon",
    "MLX",
    "Metal",
    "inference",
    "Rust",
    "local AI",
    "OpenAI compatible",
    "llama",
    "qwen",
    "gemma",
    "deepseek",
    "mlxcel",
  ],
  authors: [{ name: "Lablup Inc.", url: "https://www.lablup.com" }],
  creator: "Lablup Inc.",
  publisher: "Lablup Inc.",
  openGraph: {
    type: "website",
    locale: "en_US",
    alternateLocale: "ko_KR",
    url: "https://mlxcel.ai",
    siteName: "mlxcel",
    title: "mlxcel | High-Performance LLM Inference on Apple Silicon",
    description:
      "Run 100+ LLM and VLM models natively on Apple Silicon with Metal acceleration. Rust-powered, zero Python dependencies, OpenAI-compatible API.",
    images: [
      {
        url: "https://mlxcel.ai/og-image.png",
        width: 1200,
        height: 630,
        alt: "mlxcel | High-Performance LLM Inference on Apple Silicon",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "mlxcel | High-Performance LLM Inference on Apple Silicon",
    description:
      "Run 100+ LLM and VLM models natively on Apple Silicon with Metal acceleration. Rust-powered, zero Python dependencies, OpenAI-compatible API.",
    images: ["https://mlxcel.ai/og-image.png"],
    creator: "@lablupinc",
  },
  robots: {
    index: true,
    follow: true,
  },
  metadataBase: new URL("https://mlxcel.ai"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
