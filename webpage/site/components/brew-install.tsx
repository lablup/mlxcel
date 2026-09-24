"use client";

import { useState } from "react";
import { Check, Copy, Terminal } from "lucide-react";
import { MotionDiv } from "@/components/motion-wrapper";
import type { Dictionary } from "@/dictionaries/en";

interface BrewInstallProps {
  dict: Dictionary["brew"];
}

export function BrewInstall({ dict }: BrewInstallProps) {
  const [copied, setCopied] = useState(false);
  const command = "brew install lablup/tap/mlxcel";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(`brew tap lablup/tap && ${command}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  return (
    <section className="py-16 relative">
      <div className="container mx-auto px-4">
        <MotionDiv
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto"
        >
          <div className="text-center mb-8">
            <div className="mb-3 inline-flex items-center gap-2 text-sm text-slate-500">
              <Terminal className="w-4 h-4" />
              {dict.badge}
            </div>
            <h3 className="mb-2 text-2xl font-bold text-slate-950">{dict.title}</h3>
            <p className="text-sm text-slate-600">{dict.subtitle}</p>
          </div>

          {/* Terminal window */}
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-[0_28px_90px_-38px_rgba(15,23,42,0.18)]">
            {/* Title bar */}
            <div className="flex items-center gap-2 border-b border-slate-300/80 bg-[#e8edf3] px-4 py-3">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/70" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                <div className="w-3 h-3 rounded-full bg-green-500/70" />
              </div>
              <span className="ml-2 font-mono text-[11px] text-slate-500">
                Terminal
              </span>
            </div>

            {/* Terminal body */}
            <div className="space-y-3 bg-[#0a0a0f] px-4 py-4 font-mono text-sm sm:px-5">
              {/* Step 1: tap */}
              <div className="flex items-center gap-2 text-gray-500">
                <span className="text-brand-cyan select-none">$</span>
                <span className="text-gray-300">brew tap lablup/tap</span>
              </div>

              {/* Step 2: install */}
              <div className="group flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1 overflow-hidden">
                  <div className="flex items-start gap-2 min-w-0">
                    <span className="text-brand-cyan select-none">$</span>
                    <span className="break-all text-white font-medium sm:break-normal">
                      {command}
                    </span>
                  </div>
                </div>
                <button
                  onClick={handleCopy}
                  className="shrink-0 rounded-md p-1.5 text-gray-600 transition-all cursor-pointer hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0f]"
                  aria-label="Copy commands"
                >
                  {copied ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          </div>

          <p className="mt-4 text-center text-[11px] text-slate-500">
            {dict.note}
          </p>
        </MotionDiv>
      </div>
    </section>
  );
}
