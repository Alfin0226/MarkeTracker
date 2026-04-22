import Link from "next/link";
import { TrendingUp, BarChart3, Briefcase, Search } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative flex items-center justify-center min-h-[80vh] px-6 py-16 text-center">
        {/* Glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-[radial-gradient(ellipse,rgba(139,92,246,0.15)_0%,transparent_70%)] pointer-events-none" />

        <div className="relative z-10 max-w-3xl">
          <span className="inline-block px-4 py-1.5 mb-6 rounded-full text-sm font-semibold tracking-wide text-violet-400 bg-violet-500/10 border border-violet-500/30">
            📈 Virtual Trading Platform
          </span>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold leading-tight mb-5">
            Track. Analyze.
            <span className="bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent">
              {" "}Outperform.
            </span>
          </h1>

          <p className="text-lg text-muted-foreground leading-relaxed mb-10 max-w-xl mx-auto">
            Real-time stock data, S&P 500 comparisons, and a $1M virtual
            portfolio to practice your trading strategy.
          </p>

          <div className="flex gap-4 justify-center flex-wrap">
            <Link
              href="/register"
              className="px-7 py-3 rounded-lg font-semibold text-white bg-gradient-to-r from-violet-600 to-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-violet-500/25 transition-all"
            >
              Get Started Free
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {[
            {
              icon: <BarChart3 className="h-8 w-8 text-violet-400" />,
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
              desc: "Income statements, analyst ratings, and ML-powered price forecasts",
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="p-6 rounded-xl bg-card/50 border border-border/50 backdrop-blur-sm transition-all hover:border-violet-500/30 hover:-translate-y-1 hover:shadow-lg hover:shadow-violet-500/5"
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

      {/* CTA Section */}
      <section className="text-center px-6 py-16 border-t border-border/50">
        <h2 className="text-2xl sm:text-3xl font-bold mb-3">
          Ready to start tracking?
        </h2>
        <p className="text-muted-foreground mb-8 text-lg">
          Join MarkeTracker and level up your investment skills.
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          <Link
            href="/login"
            className="px-7 py-3 rounded-lg font-semibold text-white bg-gradient-to-r from-violet-600 to-blue-600 shadow-lg hover:-translate-y-0.5 hover:shadow-violet-500/25 transition-all"
          >
            Login
          </Link>
          <Link
            href="/register"
            className="px-7 py-3 rounded-lg font-semibold border border-border/50 hover:border-violet-500/50 hover:text-violet-400 hover:bg-violet-500/5 transition-all"
          >
            Sign Up
          </Link>
        </div>
      </section>
    </div>
  );
}
