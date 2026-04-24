/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  createChart,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
} from "lightweight-charts";
import {
  fetchDashboardData as apiFetchDashboardData,
  fetchComparisonData as apiFetchComparisonData,
  searchSymbols,
  addToWatchlist,
  removeFromWatchlist,
  fetchWatchlist,
} from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, Star, TrendingUp, DollarSign, Activity, Briefcase } from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
}

function StockDashboardContent() {
  const params = useParams();
  const symbol = typeof params?.symbol === "string" ? params.symbol.toUpperCase() : "";
  const router = useRouter();

  const [dashboardData, setDashboardData] = useState<any>(null);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [period, setPeriod] = useState("1y");
  const [showSP500, setShowSP500] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const searchTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const [isPolling, setIsPolling] = useState(false);
  const [isWatchlisted, setIsWatchlisted] = useState(false);

  const fetchAllData = useCallback(async (sym: string, per: string) => {
    setError(null);
    setDashboardData(null);
    setComparisonData(null);
    try {
      const [dashData, compData] = await Promise.all([
        apiFetchDashboardData(sym),
        apiFetchComparisonData(sym, per),
      ]);
      setDashboardData(dashData);
      setComparisonData(compData);
    } catch (err) {
      setError("Failed to load dashboard data. Please try again.");
    }
  }, []);

  useEffect(() => {
    if (symbol) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      fetchAllData(symbol, period);
    }
  }, [symbol, period, fetchAllData]);

  // Real-time polling: update comparison data every 30s
  useEffect(() => {
    if (!symbol) return;

    const pollData = async () => {
      try {
        const compData = await apiFetchComparisonData(symbol, period);
        setComparisonData(compData);
      } catch {
        // Silent fail on poll
      }
    };

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsPolling(true);
    pollingRef.current = setInterval(pollData, 30000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      setIsPolling(false);
    };
  }, [symbol, period]);

  // Check if symbol is in watchlist
  useEffect(() => {
    const checkWatchlist = async () => {
      try {
        const data = await fetchWatchlist();
        const symbols = data.watchlist.map((w: any) => w.symbol);
        setIsWatchlisted(symbols.includes(symbol));
      } catch {
        // Ignore
      }
    };
    if (symbol) checkWatchlist();
  }, [symbol]);

  const handleWatchlistToggle = async () => {
    try {
      if (isWatchlisted) {
        await removeFromWatchlist(symbol);
        setIsWatchlisted(false);
      } else {
        await addToWatchlist(symbol);
        setIsWatchlisted(true);
      }
    } catch {
      // Ignore
    }
  };

  useEffect(() => {
    if (!comparisonData || !chartContainerRef.current) return;

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.05)" },
        horzLines: { color: "rgba(255, 255, 255, 0.05)" },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      leftPriceScale: {
        visible: showSP500,
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    if (comparisonData.ohlc_data) {
      candlestickSeries.setData(comparisonData.ohlc_data);
    }

    if (comparisonData.volume_data) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: "#26a69a",
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      volumeSeries.setData(comparisonData.volume_data);
    }

    if (showSP500 && comparisonData.sp500_line_data) {
      const sp500Series = chart.addSeries(LineSeries, {
        color: "#f43f5e",
        lineWidth: 2,
        priceScaleId: "left",
      });
      sp500Series.setData(comparisonData.sp500_line_data);
    }

    chart.timeScale().fitContent();

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [comparisonData, showSP500]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);

    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }

    if (val.length > 1) {
      searchTimerRef.current = setTimeout(async () => {
        try {
          const data = await searchSymbols(val);
          setSuggestions(data);
        } catch {
          setSuggestions([]);
        }
      }, 300);
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (sym: string) => {
    router.push(`/dashboard/${sym}`);
    setSearch("");
    setSuggestions([]);
  };

  const renderPeriodPriceChange = () => {
    if (!comparisonData) return null;
    const isPositive = comparisonData.price_change >= 0;
    return (
      <div
        className={`inline-flex items-center px-3 py-1 rounded-md font-semibold text-sm ${
          isPositive
            ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30"
            : "bg-red-500/10 text-red-500 border border-red-500/30"
        }`}
      >
        {isPositive ? "+" : ""}
        {comparisonData.price_change.toFixed(2)} (
        {isPositive ? "+" : ""}
        {comparisonData.price_change_percent}%)
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Search Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h2 className="text-2xl font-extrabold bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent">
            Market Dashboard
          </h2>
        </div>
        <div className="relative w-full md:w-80 z-50">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search symbol..."
              value={search}
              onChange={handleSearchChange}
              className="pl-9 bg-card/50 backdrop-blur-sm border-border/50 focus:border-violet-500"
            />
          </div>
      {suggestions.length > 0 && (
        <ul className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-xl overflow-hidden max-h-72 overflow-y-auto">
          {suggestions.map((s) => (
            <li
              key={s.symbol}
              onClick={() => handleSuggestionClick(s.symbol)}
              className="px-4 py-2.5 cursor-pointer text-sm hover:bg-accent transition-colors border-b border-border/30 last:border-0"
            >
              <strong className="text-violet-400">{s.symbol}</strong> - {s.name}
            </li>
          ))}
        </ul>
      )}
        </div>
      </div>

      {error && (
        <div className="p-4 mb-6 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-center">
          {error}
        </div>
      )}

      {!dashboardData && !error && (
        <div className="flex flex-col space-y-6">
          <div className="h-32 rounded-xl bg-card/50 border border-border/50 animate-pulse" />
          <div className="h-96 rounded-xl bg-card/50 border border-border/50 animate-pulse" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="h-48 rounded-xl bg-card/50 border border-border/50 animate-pulse" />
            <div className="h-48 rounded-xl bg-card/50 border border-border/50 animate-pulse" />
            <div className="h-48 rounded-xl bg-card/50 border border-border/50 animate-pulse" />
          </div>
        </div>
      )}

      {dashboardData && (
        <div className="space-y-6 animate-in fade-in duration-500">
          {/* Stock Header Card */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 to-blue-500" />
            <CardContent className="p-6 md:p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
                    {dashboardData.longName}
                  </h1>
                  <span className="px-3 py-1 rounded-md bg-secondary text-secondary-foreground font-semibold text-lg">
                    {symbol}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-4">
                  <Button
                    variant={isWatchlisted ? "secondary" : "outline"}
                    size="sm"
                    onClick={handleWatchlistToggle}
                    className={`gap-2 ${
                      isWatchlisted
                        ? "bg-amber-500/20 text-amber-500 hover:bg-amber-500/30 border-amber-500/30"
                        : ""
                    }`}
                  >
                    <Star
                      className={`h-4 w-4 ${isWatchlisted ? "fill-current" : ""}`}
                    />
                    {isWatchlisted ? "Watchlisted" : "Add to Watchlist"}
                  </Button>
                  {isPolling && (
                    <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-500 border border-emerald-500/30 text-xs font-bold tracking-wider">
                      <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                      LIVE
                    </span>
                  )}
                </div>
              </div>

              <div className="text-left md:text-right">
                <div className="text-4xl md:text-5xl font-extrabold mb-2 text-foreground">
                  {comparisonData && typeof comparisonData.end_price === "number"
                    ? `$${comparisonData.end_price.toFixed(2)}`
                    : "---"}
                </div>
                {renderPeriodPriceChange()}
              </div>
            </CardContent>
          </Card>

          {/* Chart Section */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-lg">
            <CardHeader className="pb-2 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <CardTitle className="text-xl flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-violet-400" />
                Performance Overview
              </CardTitle>
              <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
                <div className="flex flex-wrap items-center gap-1 bg-secondary/50 p-1 rounded-lg">
                  {["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "max"].map((p) => (
                    <button
                      key={p}
                      onClick={() => setPeriod(p)}
                      className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                        period === p
                          ? "bg-violet-500 text-white shadow-sm"
                          : "text-muted-foreground hover:text-foreground hover:bg-background/50"
                      }`}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSP500(!showSP500)}
                  className={`w-full sm:w-auto text-xs ${
                    showSP500 ? "border-violet-500/50 text-violet-400" : ""
                  }`}
                >
                  {showSP500 ? "Hide S&P 500" : "Compare S&P 500"}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full mt-4" ref={chartContainerRef}></div>
            </CardContent>
          </Card>

          {/* Metrics Grids */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Briefcase className="h-5 w-5 text-blue-400" />
                  Key Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Sector</p>
                    <p className="font-semibold">{dashboardData.sector || "N/A"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Industry</p>
                    <p className="font-semibold">{dashboardData.industry || "N/A"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Market Cap</p>
                    <p className="font-semibold">{dashboardData.marketCap || "N/A"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Website</p>
                    <a
                      href={dashboardData.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-semibold text-violet-400 hover:text-violet-300 hover:underline"
                    >
                      Visit Official
                    </a>
                  </div>
                  <div className="space-y-1 col-span-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Day Range</p>
                    <p className="font-semibold">
                      ${dashboardData.regularMarketDayLow?.toFixed(2) || "N/A"} - ${dashboardData.regularMarketDayHigh?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                  <div className="space-y-1 col-span-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">52-Week Range</p>
                    <p className="font-semibold">
                      ${dashboardData.fiftyTwoWeekLow?.toFixed(2) || "N/A"} - ${dashboardData.fiftyTwoWeekHigh?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <DollarSign className="h-5 w-5 text-emerald-400" />
                  Valuation & Rating
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">P/E Ratio (TTM)</p>
                    <p className="font-semibold">{dashboardData.trailingPE?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">EPS (TTM)</p>
                    <p className="font-semibold">{dashboardData.trailingEps?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Div Yield</p>
                    <p className="font-semibold">
                      {dashboardData.dividendYield ? `${dashboardData.dividendYield.toFixed(2)}%` : "N/A"}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Analyst Rating</p>
                    <p className="font-semibold capitalize">{dashboardData.averageAnalystRating || "N/A"}</p>
                  </div>
                  <div className="space-y-1 col-span-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Mean Target Price</p>
                    <p className="font-semibold">${dashboardData.targetMeanPrice?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div className="col-span-2 p-3 mt-2 rounded-lg bg-violet-500/10 border border-violet-500/20">
                    <p className="text-xs text-violet-400 uppercase tracking-wider font-bold mb-1">
                      ML Price Forecast (Educational)
                    </p>
                    <p className="font-bold text-lg text-foreground">
                      {dashboardData.forecast_price ? `$${dashboardData.forecast_price.toFixed(2)}` : "Not Available"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Business Summary */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardContent className="pt-6">
              <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3">
                Business Summary
              </h3>
              <p className="text-sm leading-relaxed text-foreground/80">
                {dashboardData.longBusinessSummary || "No summary available."}
              </p>
            </CardContent>
          </Card>

          {/* Income Statement */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="h-5 w-5 text-amber-400" />
                Quarterly Income Statement
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {dashboardData.income_grid_items && Array.isArray(dashboardData.income_grid_items) && dashboardData.income_grid_items.length > 0 ? (
                  dashboardData.income_grid_items.map((item: any, idx: number) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-xl border border-border/50 bg-background/50 ${
                        item.css_class === "positive"
                          ? "border-l-4 border-l-emerald-500"
                          : item.css_class === "negative"
                          ? "border-l-4 border-l-red-500"
                          : "border-l-4 border-l-blue-500"
                      }`}
                    >
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                        {item.label}
                      </p>
                      <p
                        className={`text-lg font-bold ${
                          item.css_class === "positive"
                            ? "text-emerald-500"
                            : item.css_class === "negative"
                            ? "text-red-500"
                            : "text-foreground"
                        }`}
                      >
                        {item.value}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-muted-foreground text-sm col-span-full py-4">
                    Income statement data is not available.
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default function StockDashboardPage() {
  return (
    <ProtectedRoute>
      <StockDashboardContent />
    </ProtectedRoute>
  );
}
