"use client";

import { useState, useEffect, useCallback } from "react";
import { searchSymbols, runBacktest } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import BacktestChart from "@/components/backtest-chart";
import BacktestAllocationChart from "@/components/backtest-allocation-chart";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Search,
  Loader2,
  X,
  Play,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Scale,
  Activity,
  DollarSign,
  Trophy,
  Target,
} from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
}

interface TickerEntry {
  symbol: string;
  name: string;
  weight: number;
}

interface BacktestResult {
  portfolio_series: { date: string; value: number }[];
  sp500_series: { date: string; value: number }[];
  summary: {
    initial_investment: number;
    final_value: number;
    total_return_pct: number;
    annualized_return_pct: number;
    max_drawdown_pct: number;
    volatility_pct: number;
    sharpe_ratio: number;
    sp500_return_pct: number;
    trading_days: number;
    eval_interval: number;
  };
  ticker_breakdown: {
    symbol: string;
    weight: number;
    return_pct: number;
    contribution: number;
    start_price: number;
    end_price: number;
  }[];
}

// Default date range: 1 year ago to today
function getDefaultDates() {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 1);
  return {
    start: start.toISOString().split("T")[0],
    end: end.toISOString().split("T")[0],
  };
}

function BacktestContent() {
  const defaults = getDefaultDates();

  // Ticker management
  const [tickers, setTickers] = useState<TickerEntry[]>([]);
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  // Configuration
  const [startDate, setStartDate] = useState(defaults.start);
  const [endDate, setEndDate] = useState(defaults.end);
  const [initialInvestment, setInitialInvestment] = useState("10000");

  // Results
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Total weight calculation
  const totalWeight = tickers.reduce((sum, t) => sum + t.weight, 0);
  const weightsValid = Math.abs(totalWeight - 100) <= 0.5;

  // Search suggestions (debounced)
  const fetchSuggestions = useCallback(async (query: string) => {
    if (query.length > 1) {
      setSearchLoading(true);
      try {
        const data = await searchSymbols(query);
        setSuggestions(data);
      } catch {
        setSuggestions([]);
      } finally {
        setSearchLoading(false);
      }
    } else {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => fetchSuggestions(search), 300);
    return () => clearTimeout(timer);
  }, [search, fetchSuggestions]);

  const addTicker = (symbol: string, name: string) => {
    if (tickers.length >= 10) return;
    if (tickers.some((t) => t.symbol === symbol)) return;
    setTickers((prev) => [...prev, { symbol, name, weight: 0 }]);
    setSearch("");
    setSuggestions([]);
  };

  const removeTicker = (symbol: string) => {
    setTickers((prev) => prev.filter((t) => t.symbol !== symbol));
  };

  const updateWeight = (symbol: string, weight: number) => {
    setTickers((prev) =>
      prev.map((t) => (t.symbol === symbol ? { ...t, weight } : t))
    );
  };

  const equalWeight = () => {
    if (tickers.length === 0) return;
    const w = parseFloat((100 / tickers.length).toFixed(2));
    const remainder = parseFloat((100 - w * tickers.length).toFixed(2));
    setTickers((prev) =>
      prev.map((t, i) => ({
        ...t,
        weight: i === 0 ? w + remainder : w,
      }))
    );
  };

  const handleRunBacktest = async () => {
    if (tickers.length === 0) {
      setError("Add at least one ticker");
      return;
    }
    if (!weightsValid) {
      setError(`Weights must sum to 100% (currently ${totalWeight.toFixed(1)}%)`);
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);

    try {
      const data = await runBacktest({
        tickers: tickers.map((t) => t.symbol),
        weights: tickers.map((t) => t.weight),
        start_date: startDate,
        end_date: endDate,
        initial_investment: parseFloat(initialInvestment) || 10000,
      });
      setResult(data);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { error?: string } }; message?: string };
      setError(e.response?.data?.error || e.message || "Backtest failed");
    } finally {
      setRunning(false);
    }
  };

  const formatCurrency = (v: number) =>
    v.toLocaleString("en-US", { style: "currency", currency: "USD" });

  const formatPct = (v: number) => {
    const sign = v >= 0 ? "+" : "";
    return `${sign}${v.toFixed(2)}%`;
  };

  const pctColor = (v: number) => (v >= 0 ? "text-green-400" : "text-red-400");

  return (
    <div className="min-h-[calc(100vh-3.5rem)] p-6 md:p-10 max-w-[1400px] mx-auto space-y-8">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-blue-400">
          Portfolio Backtester
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Test how a portfolio of stocks would have performed historically.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <Card className="card-glass">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Search className="h-5 w-5 text-blue-400" />
                Add Tickers
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative">
                <Input
                  type="text"
                  placeholder="Search by name or symbol (e.g. AAPL, VTI)"
                  value={search}
                  onChange={(e) => setSearch(e.target.value.toUpperCase())}
                  autoComplete="off"
                  id="backtest-ticker-search"
                  disabled={tickers.length >= 10}
                />
                {searchLoading && (
                  <div className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg z-50 p-3 flex items-center justify-center">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    <span className="text-sm">Searching...</span>
                  </div>
                )}
                {!searchLoading && suggestions.length > 0 && (
                  <ul className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-y-auto z-[100] max-h-[280px]">
                    {suggestions
                      .filter((s) => !tickers.some((t) => t.symbol === s.symbol))
                      .map((s) => (
                        <li
                          key={s.symbol}
                          onClick={() => addTicker(s.symbol, s.name)}
                          className="px-4 py-3 cursor-pointer text-sm hover:bg-accent transition-colors border-b border-border/30 last:border-0"
                        >
                          <strong className="text-blue-400">{s.symbol}</strong>{" "}
                          — {s.name}
                        </li>
                      ))}
                  </ul>
                )}
              </div>

              {tickers.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">
                      {tickers.length}/10 tickers
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={equalWeight}
                      className="text-xs h-7 gap-1"
                    >
                      <Scale className="h-3 w-3" />
                      Equal Weight
                    </Button>
                  </div>

                  {tickers.map((t) => (
                    <div
                      key={t.symbol}
                      className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 p-3 card-flat bg-background/50 border-border/30"
                    >
                      <div className="flex-1 min-w-0">
                        <span className="font-semibold text-sm text-blue-400">
                          {t.symbol}
                        </span>
                        <span className="text-xs text-muted-foreground ml-2 truncate">
                          {t.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 w-full sm:w-auto">
                        <input
                          type="range"
                          min="0"
                          max="100"
                          step="0.5"
                          value={t.weight}
                          onChange={(e) =>
                            updateWeight(t.symbol, parseFloat(e.target.value))
                          }
                          className="w-24 h-1.5 accent-blue-500 cursor-pointer"
                        />
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          step="0.5"
                          value={t.weight}
                          onChange={(e) =>
                            updateWeight(
                              t.symbol,
                              parseFloat(e.target.value) || 0
                            )
                          }
                          className="w-20 text-center text-sm h-8"
                        />
                        <span className="text-xs text-muted-foreground w-4">%</span>
                        <button
                          onClick={() => removeTicker(t.symbol)}
                          className="p-1 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition-colors"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}

                  <div
                    className={`text-sm font-medium text-right pr-2 transition-colors ${
                      weightsValid
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    Total: {totalWeight.toFixed(1)}%{" "}
                    {weightsValid ? "✓" : `(need 100%)`}
                  </div>
                </div>
              )}

              {tickers.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-6">
                  Search and add up to 10 tickers to build your portfolio.
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card className="card-flat bg-card/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-400" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  Start Date
                </label>
                <Input
                  id="backtest-start-date"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  End Date
                </label>
                <Input
                  id="backtest-end-date"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  Initial Investment ($)
                </label>
                <Input
                  id="backtest-initial-investment"
                  type="number"
                  min="1"
                  step="100"
                  value={initialInvestment}
                  onChange={(e) => setInitialInvestment(e.target.value)}
                  placeholder="10000"
                />
              </div>

              {/* Quick date range buttons */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">
                  Quick Range
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    { label: "6M", months: 6 },
                    { label: "1Y", months: 12 },
                    { label: "2Y", months: 24 },
                    { label: "5Y", months: 60 },
                  ].map((r) => (
                    <Button
                      key={r.label}
                      variant="outline"
                      size="sm"
                      className="text-xs h-7 flex-1"
                      onClick={() => {
                        const end = new Date();
                        const start = new Date();
                        start.setMonth(end.getMonth() - r.months);
                        setStartDate(start.toISOString().split("T")[0]);
                        setEndDate(end.toISOString().split("T")[0]);
                      }}
                    >
                      {r.label}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Run Button */}
          <Button
            id="run-backtest-btn"
            onClick={handleRunBacktest}
            disabled={running || tickers.length === 0 || !weightsValid}
            className="w-full h-12 gap-2 text-base font-semibold bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-900/20 transition-all duration-200"
          >
            {running ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Running Backtest...
              </>
            ) : (
              <>
                <Play className="h-5 w-5" />
                Run Backtest
              </>
            )}
          </Button>

          {error && (
            <div className="alert-error text-center">
              {error}
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-in fade-in-0 slide-in-from-bottom-4 duration-500">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            {[
              {
                label: "Final Value",
                value: formatCurrency(result.summary.final_value),
                icon: DollarSign,
                color: result.summary.total_return_pct >= 0 ? "text-green-400" : "text-red-400",
              },
              {
                label: "Total Return",
                value: formatPct(result.summary.total_return_pct),
                icon: result.summary.total_return_pct >= 0 ? TrendingUp : TrendingDown,
                color: pctColor(result.summary.total_return_pct),
              },
              {
                label: "Annualized Return",
                value: formatPct(result.summary.annualized_return_pct),
                icon: BarChart3,
                color: pctColor(result.summary.annualized_return_pct),
              },
              {
                label: "Sharpe Ratio",
                value: result.summary.sharpe_ratio.toFixed(2),
                icon: Trophy,
                color:
                  result.summary.sharpe_ratio >= 1
                    ? "text-green-400"
                    : result.summary.sharpe_ratio >= 0
                    ? "text-yellow-400"
                    : "text-red-400",
              },
              {
                label: "Max Drawdown",
                value: formatPct(result.summary.max_drawdown_pct),
                icon: TrendingDown,
                color: "text-red-400",
              },
              {
                label: "Volatility",
                value: formatPct(result.summary.volatility_pct).replace("+", ""),
                icon: Activity,
                color: "text-yellow-400",
              },
              {
                label: "S&P 500 Return",
                value: formatPct(result.summary.sp500_return_pct),
                icon: Target,
                color: pctColor(result.summary.sp500_return_pct),
              },
              {
                label: "vs. Benchmark",
                value: formatPct(
                  result.summary.total_return_pct - result.summary.sp500_return_pct
                ),
                icon:
                  result.summary.total_return_pct >=
                  result.summary.sp500_return_pct
                    ? TrendingUp
                    : TrendingDown,
                color: pctColor(
                  result.summary.total_return_pct - result.summary.sp500_return_pct
                ),
              },
            ].map((metric, i) => (
              <Card
                key={i}
                className={i % 2 === 0 ? "card-glass shadow-xs" : "card-flat shadow-xs"}
              >
                <CardContent className="pt-5 pb-4 px-4">
                  <div className="flex items-center gap-2 mb-2">
                    <metric.icon className={`h-4 w-4 ${metric.color}`} />
                    <span className="text-xs text-muted-foreground font-medium">
                      {metric.label}
                    </span>
                  </div>
                  <div className={`text-lg md:text-xl font-bold ${metric.color}`}>
                    {metric.value}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2 card-glass">
              <CardHeader>
                <CardTitle className="text-lg">
                  Portfolio Performance vs S&P 500
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {formatCurrency(result.summary.initial_investment)} invested from{" "}
                  {startDate} to {endDate} • Evaluated every{" "}
                  {result.summary.eval_interval === 2
                    ? "2 days"
                    : result.summary.eval_interval === 5
                    ? "week"
                    : "2 weeks"}
                </p>
              </CardHeader>
              <CardContent>
                <BacktestChart
                  portfolioSeries={result.portfolio_series}
                  sp500Series={result.sp500_series}
                  initialInvestment={result.summary.initial_investment}
                />
              </CardContent>
            </Card>

            <Card className="card-flat">
              <CardHeader>
                <CardTitle className="text-lg">Allocation</CardTitle>
              </CardHeader>
              <CardContent>
                <BacktestAllocationChart
                  allocations={tickers.map((t) => ({
                    symbol: t.symbol,
                    weight: t.weight,
                  }))}
                />
              </CardContent>
            </Card>
          </div>

          <Card className="card-glass overflow-hidden shadow-xs">
            <CardHeader>
              <CardTitle className="text-lg">Ticker Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground border-b border-border/50">
                  <tr>
                    <th className="px-6 py-4 font-normal">Symbol</th>
                    <th className="px-6 py-4 font-normal text-right">Weight</th>
                    <th className="px-6 py-4 font-normal text-right">
                      Start Price
                    </th>
                    <th className="px-6 py-4 font-normal text-right">
                      End Price
                    </th>
                    <th className="px-6 py-4 font-normal text-right">Return</th>
                    <th className="px-6 py-4 font-normal text-right">
                      Contribution
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.ticker_breakdown.map((t) => (
                    <tr
                      key={t.symbol}
                      className="border-b border-border/50 last:border-0 hover:bg-white/5 transition-colors"
                    >
                      <td className="px-6 py-4 font-semibold text-blue-400">
                        {t.symbol}
                      </td>
                      <td className="px-6 py-4 text-right">{t.weight}%</td>
                      <td className="px-6 py-4 text-right">
                        ${t.start_price.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 text-right">
                        ${t.end_price.toFixed(2)}
                      </td>
                      <td
                        className={`px-6 py-4 text-right font-medium ${pctColor(
                          t.return_pct
                        )}`}
                      >
                        {formatPct(t.return_pct)}
                      </td>
                      <td
                        className={`px-6 py-4 text-right font-medium ${pctColor(
                          t.contribution
                        )}`}
                      >
                        {formatPct(t.contribution)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default function BacktestPage() {
  return (
    <ProtectedRoute>
      <BacktestContent />
    </ProtectedRoute>
  );
}
