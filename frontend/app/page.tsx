"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { TrendingUp, BarChart3, Briefcase, Search } from "lucide-react";
import { useAuth } from "@/context/auth-context";

// Headline copy options:
// Option 1: Prove your trading strategy can actually beat the S&P 500
// Option 2: Backtest your trading theories before risking real capital
// Option 3: Benchmark your stock portfolio performance against the actual market
// Selected: Option 1 (fits the active trader audience and highlights the core benchmark feature)

export default function LandingPage() {
  const { isAuthenticated } = useAuth();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen">
      <section className="relative flex items-center justify-center min-h-[85vh] px-6 py-12 md:py-20">
        {/* Glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-[radial-gradient(ellipse,rgba(59,130,246,0.1)_0%,transparent_70%)] pointer-events-none" />

        <div className="relative z-10 max-w-6xl w-full grid grid-cols-1 md:grid-cols-12 gap-12 md:gap-8 items-center">
          {/* Left Column (Content) */}
          <div className="md:col-span-7 flex flex-col items-center md:items-start text-center md:text-left">
            <h1 className="text-4xl sm:text-5xl md:text-[52px] lg:text-[58px] font-bold leading-[1.15] mb-5 tracking-tight text-foreground">
              Prove your trading strategy can actually beat the S&P 500
            </h1>

            <p className="text-base sm:text-lg text-muted-foreground leading-relaxed mb-8 max-w-xl">
              Simulate trades with a $1M virtual portfolio, compare performance against the S&P 500, and access real-time NASDAQ/NYSE stock data.
            </p>

            <div className="flex gap-4 justify-center md:justify-start flex-wrap">
              {mounted && isAuthenticated ? (
                <Link
                  href="/dashboard"
                  className="px-7 py-3 rounded-lg font-semibold text-white bg-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-blue-500/25 hover:bg-blue-700 transition-all"
                >
                  Open the dashboard
                </Link>
              ) : (
                <Link
                  href="/register"
                  className="px-7 py-3 rounded-lg font-semibold text-white bg-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-blue-500/25 hover:bg-blue-700 transition-all"
                >
                  Start a sim with $1M
                </Link>
              )}
            </div>
          </div>

          {/* Right Column (Product Chart Preview Card) */}
          <div className="md:col-span-5 w-full flex justify-center">
            {/* The real chart window mockup */}
            <div className="w-full max-w-[420px] bg-zinc-900/65 border border-zinc-800 rounded-2xl shadow-2xl shadow-black/75 md:rotate-[1.5deg] overflow-hidden transition-all duration-300 hover:rotate-0 hover:border-zinc-700/80">
              {/* Browser Chrome Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/80 bg-zinc-950/30">
                <div className="flex gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-800" />
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-800" />
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-800" />
                </div>
                <div className="font-mono text-[10px] text-zinc-500 tracking-wider">
                  AAPL · 1M
                </div>
                <div className="w-9" /> {/* spacer for visual symmetry */}
              </div>

              {/* Chart Body */}
              <div className="relative p-5 h-[230px] bg-gradient-to-b from-transparent to-zinc-950/20">
                {/* Legend Chips */}
                <div className="absolute top-4 left-5 z-10 flex gap-3 text-[10px] font-mono select-none">
                  <div className="flex items-center gap-1.5 bg-zinc-950/60 px-2 py-1 rounded border border-zinc-800/80">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                    <span className="text-zinc-300 font-bold">AAPL</span>
                    <span className="text-emerald-500 font-bold">+0.31%</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-zinc-950/60 px-2 py-1 rounded border border-zinc-800/80">
                    <span className="w-1.5 h-1.5 bg-zinc-400 rounded-full" />
                    <span className="text-zinc-400">S&P 500</span>
                    <span className="text-zinc-300 font-semibold">+2.13%</span>
                  </div>
                </div>

                {/* SVG Candlestick & S&P 500 line Chart */}
                <svg className="w-full h-full pt-8" viewBox="0 0 320 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {/* Grid Lines */}
                  <line x1="0" y1="40" x2="320" y2="40" stroke="#1f2937" strokeWidth="0.5" strokeDasharray="2 4" />
                  <line x1="0" y1="80" x2="320" y2="80" stroke="#1f2937" strokeWidth="0.5" strokeDasharray="2 4" />
                  <line x1="0" y1="120" x2="320" y2="120" stroke="#1f2937" strokeWidth="0.5" strokeDasharray="2 4" />

                  {/* S&P 500 Muted Neutral Gray Line: normalized to % change, starts at 0% (y=80) */}
                  <path
                    d="M 10 80 
                       C 30 75, 45 85, 60 72 
                       C 75 60, 90 95, 110 82 
                       C 130 70, 145 62, 160 55
                       C 175 48, 190 68, 210 52 
                       C 230 36, 245 42, 260 48
                       C 275 54, 290 38, 310 26"
                    stroke="#9ca3af"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity="0.85"
                  />

                  {/* Candlesticks: AAPL. Realistic price moves around $290-$310, mapped to % change. */}
                  {/* C1: Up */}
                  <line x1="20" y1="95" x2="20" y2="70" stroke="#10b981" strokeWidth="1" />
                  <rect x="17" y="75" width="6" height="15" fill="#10b981" rx="0.5" />

                  {/* C2: Down */}
                  <line x1="40" y1="88" x2="40" y2="65" stroke="#ef4444" strokeWidth="1" />
                  <rect x="37" y="70" width="6" height="12" fill="#ef4444" rx="0.5" />

                  {/* C3: Up */}
                  <line x1="60" y1="82" x2="60" y2="52" stroke="#10b981" strokeWidth="1" />
                  <rect x="57" y="58" width="6" height="18" fill="#10b981" rx="0.5" />

                  {/* C4: Up */}
                  <line x1="80" y1="65" x2="80" y2="45" stroke="#10b981" strokeWidth="1" />
                  <rect x="77" y="50" width="6" height="10" fill="#10b981" rx="0.5" />

                  {/* C5: Down */}
                  <line x1="100" y1="72" x2="100" y2="48" stroke="#ef4444" strokeWidth="1" />
                  <rect x="97" y="55" width="6" height="13" fill="#ef4444" rx="0.5" />

                  {/* C6: Down */}
                  <line x1="120" y1="88" x2="120" y2="62" stroke="#ef4444" strokeWidth="1" />
                  <rect x="117" y="68" width="6" height="16" fill="#ef4444" rx="0.5" />

                  {/* C7: Up */}
                  <line x1="140" y1="78" x2="140" y2="55" stroke="#10b981" strokeWidth="1" />
                  <rect x="137" y="62" width="6" height="12" fill="#10b981" rx="0.5" />

                  {/* C8: Up */}
                  <line x1="160" y1="65" x2="160" y2="35" stroke="#10b981" strokeWidth="1" />
                  <rect x="157" y="42" width="6" height="18" fill="#10b981" rx="0.5" />

                  {/* C9: Down */}
                  <line x1="180" y1="62" x2="180" y2="42" stroke="#ef4444" strokeWidth="1" />
                  <rect x="177" y="46" width="6" height="11" fill="#ef4444" rx="0.5" />

                  {/* C10: Up */}
                  <line x1="200" y1="52" x2="200" y2="28" stroke="#10b981" strokeWidth="1" />
                  <rect x="197" y="34" width="6" height="14" fill="#10b981" rx="0.5" />

                  {/* C11: Down */}
                  <line x1="220" y1="58" x2="220" y2="36" stroke="#ef4444" strokeWidth="1" />
                  <rect x="217" y="40" width="6" height="14" fill="#ef4444" rx="0.5" />

                  {/* C12: Up */}
                  <line x1="240" y1="48" x2="240" y2="25" stroke="#10b981" strokeWidth="1" />
                  <rect x="237" y="28" width="6" height="16" fill="#10b981" rx="0.5" />

                  {/* C13: Down */}
                  <line x1="260" y1="56" x2="260" y2="34" stroke="#ef4444" strokeWidth="1" />
                  <rect x="257" y="38" width="6" height="14" fill="#ef4444" rx="0.5" />

                  {/* C14: Up */}
                  <line x1="280" y1="42" x2="280" y2="15" stroke="#10b981" strokeWidth="1" />
                  <rect x="277" y="22" width="6" height="16" fill="#10b981" rx="0.5" />

                  {/* C15: Down */}
                  <line x1="300" y1="50" x2="300" y2="20" stroke="#ef4444" strokeWidth="1" />
                  <rect x="297" y="26" width="6" height="18" fill="#ef4444" rx="0.5" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-5xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {[
            {
              icon: <BarChart3 className="h-8 w-8 text-blue-400" />,
              title: "Real-time Data",
              desc: "Live stock prices, charts, and key financial metrics from NASDAQ & NYSE",
            },
            {
              icon: <TrendingUp className="h-8 w-8 text-emerald-400" />,
              title: "Performance Tracking",
              desc: "Compare any stock against the S&P 500 across multiple time periods",
            },
            {
              icon: <Briefcase className="h-8 w-8 text-blue-400" />,
              title: "Virtual Portfolio",
              desc: "Practice with $1M virtual capital — track P&L, win rate, and total returns",
            },
            {
              icon: <Search className="h-8 w-8 text-amber-400" />,
              title: "Smart Analytics",
              desc: "Income statements, analyst ratings, and consensus price targets",
            },
          ].map((feature, idx) => (
            <div
              key={feature.title}
              className={`p-6 card-interactive ${idx % 2 === 0 ? "card-glass" : "card-flat"}`}
            >
              <div className="mb-4">{feature.icon}</div>
              <h3 className="text-base font-bold mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feature.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="text-center px-6 py-16 border-t border-border/50">
        <h2 className="text-2xl sm:text-3xl font-semibold mb-3">
          Ready to start tracking?
        </h2>
        <p className="text-muted-foreground mb-8 text-lg">
          Join MarkeTracker and level up your investment skills.
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          {mounted && isAuthenticated ? (
            <Link
              href="/dashboard"
              className="px-7 py-3 rounded-lg font-semibold text-white bg-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-blue-500/25 hover:bg-blue-700 transition-all"
            >
              Open the dashboard
            </Link>
          ) : (
            <>
              <Link
                href="/login"
                className="px-7 py-3 rounded-lg font-semibold text-white bg-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-blue-500/25 hover:bg-blue-700 transition-all"
              >
                Login
              </Link>
              <Link
                href="/register"
                className="px-7 py-3 rounded-lg font-medium border border-border hover:border-foreground/30 hover:bg-secondary transition-colors"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
