"use client";

import { usePlatform, OS, Arch } from "@/hooks/use-os";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { Github } from "@/components/brand-icons";
import { MotionDiv } from "@/components/motion-wrapper";
import type { Dictionary } from "@/dictionaries/en";
import type { ReleaseData } from "@/lib/release";

interface SmartDownloadProps {
  dict: Dictionary["hero"];
  release: ReleaseData | null;
}

export function SmartDownload({ dict, release }: SmartDownloadProps) {
  const { os, arch } = usePlatform();

  // Helper to find asset based on OS and Architecture
  const getAssetUrl = (osName: OS, archName: Arch) => {
    if (!release) return null;
    if (osName === "unknown") return release.html_url;

    const osKeywords: Record<string, string[]> = {
      macos: ["darwin", "mac", "apple"],
      windows: ["windows", "win", ".exe"],
      linux: ["linux"],
    };

    const targetOsKeywords = osKeywords[osName];

    const archKeywords =
      archName === "arm64"
        ? ["arm64", "aarch64", "universal"]
        : ["x86_64", "amd64", "universal", "x64"];

    // Priority 1: Match OS AND specific architecture AND preferred extension
    const preferredExts =
      osName === "macos"
        ? [".dmg"]
        : osName === "windows"
          ? [".exe", ".msi"]
          : [];

    if (preferredExts.length > 0) {
      const bestMatchWithExt = release.assets.find((a) => {
        const name = a.name.toLowerCase();
        const osMatch = targetOsKeywords.some((k) => name.includes(k));
        const archMatch = archKeywords.some((k) => name.includes(k));
        const extMatch = preferredExts.some((ext) => name.endsWith(ext));
        return osMatch && archMatch && extMatch;
      });
      if (bestMatchWithExt) return bestMatchWithExt.browser_download_url;
    }

    // Priority 2: Match OS AND specific architecture (any extension)
    const bestMatch = release.assets.find((a) => {
      const name = a.name.toLowerCase();
      const osMatch = targetOsKeywords.some((k) => name.includes(k));
      const archMatch = archKeywords.some((k) => name.includes(k));
      return osMatch && archMatch;
    });

    if (bestMatch) return bestMatch.browser_download_url;

    // Priority 3: Fallback to OS match only
    const fallbackMatch = release.assets.find((a) => {
      const name = a.name.toLowerCase();
      return targetOsKeywords.some((k) => name.includes(k));
    });

    return fallbackMatch
      ? fallbackMatch.browser_download_url
      : release.html_url;
  };

  const primaryUrl = getAssetUrl(os, arch);
  const version = release?.tag_name || "Latest";

  if (!release) {
    return (
      <Button
        variant="glass"
        size="lg"
        onClick={() =>
          window.open(
            "https://github.com/lablup/mlxcel-releases",
            "_blank"
          )
        }
      >
        <Github className="mr-2 h-5 w-5" />
        {dict.view_releases}
      </Button>
    );
  }

  const osLabel = {
    macos: "macOS",
    windows: "Windows",
    linux: "Linux",
    unknown: "",
  }[os];

  const archLabel =
    arch === "arm64" ? " (ARM64)" : arch === "x86_64" ? " (x64)" : "";

  return (
    <div className="flex flex-col items-center gap-4">
      <MotionDiv
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full"
      >
        <Button
          variant="primary"
          size="lg"
          className="w-full rounded-2xl px-6 py-5 text-base shadow-[0_20px_60px_-28px_rgba(0,167,196,0.45)] sm:w-auto sm:px-10 sm:text-lg"
          onClick={() =>
            (window.location.href = primaryUrl || release.html_url)
          }
        >
          <Download className="mr-2 h-6 w-6" />
          {os === "unknown"
            ? dict.download_latest
            : `${dict.download_btn} ${osLabel}${archLabel}`}
        </Button>
      </MotionDiv>

      <div className="flex flex-col items-center gap-1 text-center">
        <p className="rounded-full border border-slate-200 bg-white px-3 py-1 font-mono text-xs text-cyan-700">
          {version}
        </p>
        <p className="text-xs tracking-wide text-slate-500">{dict.trust_line}</p>
        <p className="ko-keep-all text-[11px] tracking-[0.01em] text-slate-400">
          {dict.supporting_note}
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-3 text-sm text-slate-500">
        <a
          href={release.html_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 rounded-md transition-colors hover:text-brand-cyan focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]"
        >
          <Github className="w-3 h-3" /> {dict.release_notes}
        </a>
        <button
          onClick={() => {
            const el = document.getElementById("downloads");
            el?.scrollIntoView({ behavior: "smooth" });
          }}
          className="rounded-md transition-colors hover:text-brand-cyan focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]"
        >
          {dict.other_platforms}
        </button>
      </div>
    </div>
  );
}
