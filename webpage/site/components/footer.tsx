import Image from "next/image";
import { Github, Linkedin, Youtube } from "@/components/brand-icons";
import type { Dictionary } from "@/dictionaries/en";
import type { Locale } from "@/lib/dictionary";

interface FooterProps {
  dict: Dictionary["footer"];
  lang: Locale;
}

const LABLUP_LOGO_URL = "/brands/lablup-logo.svg";

export function Footer({ dict, lang }: FooterProps) {
  const isKorean = lang === "ko";
  const footerTitle = isKorean ? "본사 및 HPC 연구소" : "Headquarter & HPC Lab";
  const krOffice = isKorean
    ? "KR Office: 서울특별시 강남구 선릉로 577 CR타워 8층"
    : "KR Office: 8F, 577, Seolleung-ro, Gangnam-gu, Seoul, Republic of Korea";
  const usOffice = "US Office: 3003 N First st, Suite 221, San Jose, CA 95134";
  const privacyLabel = isKorean ? "개인정보취급방침" : "Privacy Policy";
  const termsLabel = isKorean ? "이용약관" : "Terms of Use";
  const policyBase = isKorean ? "https://www.backend.ai/ko" : "https://www.backend.ai";

  const socials = [
    {
      href: "https://www.youtube.com/c/LablupInc",
      label: "YouTube",
      icon: Youtube,
    },
    {
      href: "https://kr.linkedin.com/company/lablup",
      label: "LinkedIn",
      icon: Linkedin,
    },
    {
      href: "https://github.com/lablup",
      label: "GitHub",
      icon: Github,
    },
  ];

  return (
    <footer className="border-t border-slate-200/80 bg-white">
      <div className="container mx-auto flex flex-col gap-6 px-4 py-10 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-[42rem]">
          <a
            href="https://www.lablup.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center"
          >
            <Image
              src={LABLUP_LOGO_URL}
              alt="Lablup"
              width={120}
              height={32}
              className="h-7 w-auto sm:h-8"
            />
          </a>
          <address className="mt-5 not-italic text-slate-400">
            <p className="text-[1.05rem] font-semibold tracking-[-0.03em] text-slate-400 sm:text-[1.12rem]">
              {footerTitle}
            </p>
            <div className="mt-2.5 space-y-1 text-[0.74rem] leading-[1.5] sm:text-[0.8rem]">
              <p>{krOffice}</p>
              <p>{usOffice}</p>
            </div>
          </address>
        </div>

        <ul className="flex items-center gap-2.5 lg:pt-1">
          {socials.map(({ href, label, icon: Icon }) => (
            <li key={label}>
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={label}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-black text-white transition-transform duration-200 hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-white sm:h-10 sm:w-10"
              >
                <Icon className="h-4 w-4 sm:h-4.5 sm:w-4.5" />
              </a>
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-black">
        <div className="container mx-auto flex flex-col gap-3 px-4 py-4 text-[0.72rem] text-white sm:flex-row sm:items-center sm:justify-between sm:text-[0.78rem]">
          <p>{`© ${dict.rights}`}</p>
          <div className="flex items-center gap-3 sm:gap-5">
            <a
              href={`${policyBase}/privacy-policy`}
              target="_blank"
              rel="noopener noreferrer"
              className="transition-opacity hover:opacity-80"
            >
              {privacyLabel}
            </a>
            <span className="text-white/40">|</span>
            <a
              href={`${policyBase}/terms-of-use`}
              target="_blank"
              rel="noopener noreferrer"
              className="transition-opacity hover:opacity-80"
            >
              {termsLabel}
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
