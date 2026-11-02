"use client";

import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { Download, Package } from "lucide-react";
import { Github } from "@/components/brand-icons";
import { MotionDiv } from "@/components/motion-wrapper";
import { useRef } from "react";
import type { Dictionary } from "@/dictionaries/en";
import type { ReleaseData } from "@/lib/release";

interface DownloadsProps {
  dict: Dictionary["downloads"];
  release: ReleaseData | null;
}

// OS-specific logo components
function AppleLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      role="img"
      aria-label="Apple logo"
    >
      <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
    </svg>
  );
}

export function Downloads({ dict, release }: DownloadsProps) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  });
  const bgY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 100, prefersReducedMotion ? 0 : -60]
  );
  const headerY = useTransform(
    scrollYProgress,
    [0, 1],
    [prefersReducedMotion ? 0 : 48, prefersReducedMotion ? 0 : -24]
  );

  if (!release) {
    return (
      <section id="downloads" ref={sectionRef} className="py-24 relative overflow-hidden">
        <motion.div
          aria-hidden="true"
          style={{ y: bgY }}
          className="absolute inset-x-0 top-10 mx-auto h-64 max-w-5xl rounded-full bg-gradient-to-r from-cyan-300/12 via-transparent to-purple-300/14 blur-3xl"
        />
        <div className="container mx-auto px-4">
          <motion.div style={{ y: headerY }} className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4 text-slate-950">{dict.title}</h2>
            <p className="text-slate-600">{dict.subtitle}</p>
          </motion.div>
          <div className="text-center">
            <a
              href="https://github.com/lablup/mlxcel-releases/releases/latest"
              target="_blank"
              className="inline-flex items-center gap-2 text-brand-cyan transition-colors hover:text-slate-900"
            >
              <Github className="w-5 h-5" />
              {dict.view_full}
            </a>
          </div>
        </div>
      </section>
    );
  }

  // Filter out checksums, signatures, and keep only relevant formats
  const filteredAssets = release.assets.filter((asset) => {
    const name = asset.name.toLowerCase();

    const excludedExtensions = [
      ".sha256",
      ".sha256sum",
      ".md5",
      ".sha1",
      ".sig",
      ".asc",
      ".pem",
      ".sbom",
      ".txt",
      ".json",
      ".yml",
      ".yaml",
    ];
    if (excludedExtensions.some((ext) => name.endsWith(ext))) return false;

    const allowedExtensions = [
      ".dmg",
      ".pkg",
      ".tar.gz",
      ".zip",
    ];

    return allowedExtensions.some((ext) => name.endsWith(ext));
  });

  const getIcon = (name: string) => {
    if (name.includes("mac") || name.includes("darwin") || name.includes("apple"))
      return <AppleLogo className="w-5 h-5 text-gray-300" />;
    return <Package className="w-5 h-5 text-gray-400" />;
  };

  const getFriendlyName = (filename: string) => {
    const lower = filename.toLowerCase();

    const os = "macOS";
    let arch = "Apple Silicon";

    if (lower.includes("arm64") || lower.includes("aarch64")) {
      arch = "Apple Silicon";
    } else if (lower.includes("x86_64") || lower.includes("amd64")) {
      arch = "Intel";
    } else if (lower.includes("universal")) {
      arch = "Universal";
    }

    let ext = "";
    if (lower.endsWith(".dmg")) ext = "DMG";
    else if (lower.endsWith(".pkg")) ext = "PKG";
    else if (lower.endsWith(".tar.gz")) ext = "Tarball";
    else if (lower.endsWith(".zip")) ext = "ZIP";

    return `${os} (${arch})${ext ? ` - ${ext}` : ""}`;
  };

  return (
    <section
      id="downloads"
      ref={sectionRef}
      className="py-24 relative overflow-hidden"
    >
      <motion.div
        aria-hidden="true"
        style={{ y: bgY }}
        className="absolute inset-x-0 top-10 mx-auto h-72 max-w-5xl rounded-full bg-gradient-to-r from-cyan-300/12 via-transparent to-purple-300/14 blur-3xl"
      />
      <div className="container mx-auto px-4">
        <motion.div style={{ y: headerY }} className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-4 text-slate-950">{dict.title}</h2>
          <p className="text-slate-600">
            {dict.subtitle}{" "}
            <span className="font-mono text-brand-cyan">
              {release.tag_name}
            </span>
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto">
          {filteredAssets.map((asset, i) => (
            <MotionDiv
              key={asset.name}
              initial={{
                opacity: 0,
                x: prefersReducedMotion ? 0 : i % 3 === 0 ? -28 : i % 3 === 2 ? 28 : 0,
                y: prefersReducedMotion ? 0 : 36,
                scale: prefersReducedMotion ? 1 : 0.97,
              }}
              whileInView={{ opacity: 1, x: 0, y: 0, scale: 1 }}
              transition={{ duration: 0.5, delay: i * 0.04, ease: "easeOut" }}
              viewport={{ once: true }}
              className="h-full"
              whileHover={
                prefersReducedMotion
                  ? undefined
                  : { y: -6, transition: { duration: 0.2 } }
              }
            >
              <a
                href={asset.browser_download_url}
                className="group flex h-full flex-col justify-between gap-4 rounded-2xl border border-slate-200/80 bg-white/92 p-5 backdrop-blur-md transition-all hover:-translate-y-0.5 hover:border-cyan-300/35 hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#f7fbfe]"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 overflow-hidden min-w-0 flex-1">
                    <div className="shrink-0 rounded-xl bg-cyan-50 p-2.5 text-slate-900 transition-colors group-hover:bg-cyan-100">
                      {getIcon(asset.name.toLowerCase())}
                    </div>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-slate-900">
                        {getFriendlyName(asset.name)}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {asset.name.toLowerCase().includes(".dmg") ||
                        asset.name.toLowerCase().includes(".pkg")
                          ? "Best for most people"
                          : "Archive or portable build"}
                      </p>
                    </div>
                  </div>
                  <span className="shrink-0 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 font-mono text-[11px] font-medium text-slate-600 transition-colors group-hover:border-cyan-200/40">
                    {(asset.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </div>

                <div className="flex items-center justify-between gap-4 min-w-0 border-t border-slate-200 pt-3">
                  <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-slate-500 transition-colors group-hover:text-slate-700">
                    {asset.name}
                  </span>
                  <div className="flex items-center gap-2 text-sm text-cyan-700">
                    <span className="hidden sm:inline">Get it</span>
                    <Download className="h-4 w-4 shrink-0 transition-colors group-hover:text-brand-cyan" />
                  </div>
                </div>
              </a>
            </MotionDiv>
          ))}
        </div>

        <div className="text-center mt-12">
          <a
            href={release.html_url}
            target="_blank"
            className="text-sm text-slate-500 transition-colors underline decoration-dotted hover:text-slate-900"
          >
            {dict.view_full}
          </a>
        </div>
      </div>
    </section>
  );
}
