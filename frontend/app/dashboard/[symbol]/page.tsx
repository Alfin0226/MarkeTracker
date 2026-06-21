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
import { Input } from "@/components/ui/input";
import PositionSummary from "@/components/position-summary";
import TradeModal from "@/components/trade-modal";
import DetailsAccordion from "@/components/details-accordion";
import {
  Search,
  Star,
  Maximize2,
  Settings,
} from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
}

interface IncomeGridItem {
  label: string;
  value: string;
  css_class: string;
}

interface StockData {
  longName?: string;
  sector?: string;
  industry?: string;
  marketCap?: string;
  website?: string;
  regularMarketDayLow?: number;
  regularMarketDayHigh?: number;
  fiftyTwoWeekLow?: number;
  fiftyTwoWeekHigh?: number;
  trailingPE?: number;
  trailingEps?: number;
  dividendYield?: number;
  averageAnalystRating?: string;
  targetMeanPrice?: number;
  forecast_price?: number;
  longBusinessSummary?: string;
  income_grid_items?: IncomeGridItem[];
  exchange?: string;
}

interface OhlcItem {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface VolumeItem {
  time: string;
  value: number;
  color: string;
}

interface LineItem {
  time: string;
  value: number;
}

interface ComparisonData {
  ohlc_data?: OhlcItem[];
  volume_data?: VolumeItem[];
  sp500_line_data?: LineItem[];
  price_change: number;
  price_change_percent: number | string;
  end_price: number;
}

interface WatchlistItem {
  symbol: string;
  current_price: number | null;
  added_at: string;
}

/* ─── Period map: label → API value ─── */
const PERIODS: { label: string; value: string }[] = [
  { label: "1D", value: "1d" },
  { label: "1W", value: "5d" },
  { label: "1M", value: "1mo" },
  { label: "3M", value: "3mo" },
  { label: "6M", value: "6mo" },
  { label: "1Y", value: "1y" },
  { label: "2Y", value: "2y" },
  { label: "MAX", value: "max" },
];

function StockDashboardContent() {
  const params = useParams();
  const symbol =
    typeof params?.symbol === "string" ? params.symbol.toUpperCase() : "";
  const router = useRouter();

  const [dashboardData, setDashboardData] = useState<StockData | null>(null);
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(
    null
  );
  const [period, setPeriod] = useState("1y");
  const [showSP500, setShowSP500] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoverData, setHoverData] = useState<{
    aapl: number;
    sp500: number | null;
  } | null>(null);

  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const searchTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const [isPolling, setIsPolling] = useState(false);
  const [isWatchlisted, setIsWatchlisted] = useState(false);

  // Trade modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalDefaultAction, setModalDefaultAction] = useState<"buy" | "sell">("buy");
  const [positionRefreshKey, setPositionRefreshKey] = useState(0);

  const bumpRefreshKey = useCallback(() => {
    setPositionRefreshKey((k) => k + 1);
  }, []);

  // ─── Data fetching ───
  const fetchAllData = useCallback(
    async (sym: string, per: string) => {
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
      } catch {
        setError("Failed to load dashboard data/Server is booting. Please try again.");
      }
    },
    []
  );

  useEffect(() => {
    if (symbol) {
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
        const symbols = data.watchlist.map((w: WatchlistItem) => w.symbol);
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

  // ─── Chart setup ───
  useEffect(() => {
    if (!comparisonData || !chartContainerRef.current) return;
    if (!comparisonData.ohlc_data || comparisonData.ohlc_data.length === 0) return;

    const container = chartContainerRef.current;

    // Baseline values: use the opening price of the first visible bar as the baseline (0%)
    const aaplBaseline = comparisonData.ohlc_data[0].open;

    const normalizedOhlc = comparisonData.ohlc_data.map((d) => ({
      time: d.time,
      open: ((d.open - aaplBaseline) / aaplBaseline) * 100,
      high: ((d.high - aaplBaseline) / aaplBaseline) * 100,
      low: ((d.low - aaplBaseline) / aaplBaseline) * 100,
      close: ((d.close - aaplBaseline) / aaplBaseline) * 100,
    }));

    let normalizedSp500: { time: string | number; value: number }[] = [];
    if (comparisonData.sp500_line_data && comparisonData.sp500_line_data.length > 0) {
      const sp500Baseline = comparisonData.sp500_line_data[0].value;
      normalizedSp500 = comparisonData.sp500_line_data.map((d) => ({
        time: d.time,
        value: ((d.value - sp500Baseline) / sp500Baseline) * 100,
      }));
    }

    // Toggle logic: when showSP500 is enabled, display percent. Otherwise display absolute prices.
    const aaplDataToUse = showSP500 ? normalizedOhlc : comparisonData.ohlc_data;

    const latestValues = {
      aapl: aaplDataToUse[aaplDataToUse.length - 1].close,
      sp500: showSP500 && normalizedSp500.length > 0 ? normalizedSp500[normalizedSp500.length - 1].value : null,
    };

    setHoverData(latestValues);

    const handleResize = () => {
      chart.applyOptions({ width: container.clientWidth });
    };

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.04)" },
        horzLines: { color: "rgba(255, 255, 255, 0.04)" },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      leftPriceScale: {
        visible: false, // Drop left scale entirely
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      width: container.clientWidth,
      height: 460,
    });

    const priceFormatOptions = showSP500
      ? {
          type: "custom" as const,
          formatter: (price: number) => {
            const sign = price > 0 ? "+" : "";
            return `${sign}${price.toFixed(1)}%`;
          },
        }
      : {
          type: "price" as const,
          precision: 2,
          minMove: 0.01,
        };

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
      priceFormat: priceFormatOptions,
    });

    candlestickSeries.setData(aaplDataToUse);

    if (comparisonData.volume_data) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: "rgba(38, 166, 154, 0.4)",
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      volumeSeries.setData(comparisonData.volume_data);
    }

    let sp500Series: any = null;
    if (showSP500 && normalizedSp500.length > 0) {
      sp500Series = chart.addSeries(LineSeries, {
        color: "#9ca3af", // Muted neutral gray color
        lineWidth: 2,
        priceFormat: {
          type: "custom",
          formatter: (price: number) => {
            const sign = price > 0 ? "+" : "";
            return `${sign}${price.toFixed(1)}%`;
          },
        },
      });
      sp500Series.setData(normalizedSp500);
    }

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) {
        setHoverData(latestValues);
        return;
      }

      const aaplData = param.seriesData.get(candlestickSeries) as any;
      const sp500Data = showSP500 && sp500Series ? (param.seriesData.get(sp500Series) as any) : null;

      if (aaplData) {
        setHoverData({
          aapl: aaplData.close,
          sp500: sp500Data ? sp500Data.value : null,
        });
      }
    });

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

  // ─── Fullscreen toggle ───
  const handleFullscreen = () => {
    const el = chartContainerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen?.();
    }
  };

  // ─── Price change display ───
  const priceChange = comparisonData?.price_change ?? 0;
  const priceChangePct = comparisonData?.price_change_percent ?? 0;
  const isPositive = priceChange >= 0;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      {/* ──── 1. Search bar ──── */}
      <div className="flex justify-center mb-8">
        <div className="relative w-full max-w-md z-50">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="symbol-search"
            type="text"
            placeholder="Search symbol…"
            value={search}
            onChange={handleSearchChange}
            className="pl-9 bg-card/30 ring-1 ring-foreground/5 border-0 focus:ring-ring"
          />
          {suggestions.length > 0 && (
            <ul className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-xl overflow-hidden max-h-72 overflow-y-auto">
              {suggestions.map((s) => (
                <li
                  key={s.symbol}
                  onClick={() => handleSuggestionClick(s.symbol)}
                  className="px-4 py-2.5 cursor-pointer text-sm hover:bg-accent transition-colors border-b border-border/30 last:border-0"
                >
                  <strong className="text-foreground">{s.symbol}</strong>
                  <span className="text-muted-foreground"> — {s.name}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ──── Error state ──── */}
      {error && (
        <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm text-center">
          {error}
        </div>
      )}

      {/* ──── Loading skeleton ──── */}
      {!dashboardData && !error && (
        <div className="flex flex-col space-y-6 animate-pulse">
          <div className="h-36 rounded-xl bg-card/20" />
          <div className="h-[460px] rounded-xl bg-card/20" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="h-20 rounded-xl bg-card/20" />
            <div className="h-20 rounded-xl bg-card/20" />
            <div className="h-20 rounded-xl bg-card/20" />
            <div className="h-20 rounded-xl bg-card/20" />
          </div>
        </div>
      )}

      {dashboardData && (
        <div className="animate-in fade-in duration-500">
          {/* ──── 2. Flat header ──── */}
          <div className="flex flex-col sm:flex-row justify-between items-start gap-4 mb-6">
            <div className="min-w-0">
              {/* Symbol · Exchange · Live */}
              <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                <span className="font-semibold">{symbol}</span>
                {dashboardData.exchange && (
                  <>
                    <span className="opacity-40">·</span>
                    <span>{dashboardData.exchange}</span>
                  </>
                )}
                {isPolling && (
                  <>
                    <span className="opacity-40">·</span>
                    <span className="inline-flex items-center gap-1 text-emerald-500 text-xs font-semibold">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      Live
                    </span>
                  </>
                )}
              </div>

              {/* Company name */}
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground mb-2 truncate">
                {dashboardData.longName || symbol}
              </h1>

              {/* Massive price */}
              <p className="text-5xl sm:text-6xl font-mono tabular-nums font-extrabold text-foreground mb-2">
                {comparisonData &&
                typeof comparisonData.end_price === "number"
                  ? `$${comparisonData.end_price.toFixed(2)}`
                  : "—"}
              </p>

              {/* Change line */}
              {comparisonData && (
                <p
                  className={`text-sm font-semibold font-mono tabular-nums ${
                    isPositive ? "text-emerald-500" : "text-red-500"
                  }`}
                >
                  {isPositive ? "▲" : "▼"}{" "}
                  {Math.abs(priceChange).toFixed(2)} (
                  {isPositive ? "+" : ""}
                  {priceChangePct}%) today
                </p>
              )}
            </div>

            {/* Right: star + Buy */}
            <div className="flex items-center gap-2 shrink-0 mt-2 sm:mt-6">
              <button
                onClick={handleWatchlistToggle}
                className={`p-2 rounded-lg border transition-colors ${
                  isWatchlisted
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-500"
                    : "border-border text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
                aria-label={
                  isWatchlisted
                    ? "Remove from watchlist"
                    : "Add to watchlist"
                }
              >
                <Star
                  className={`h-5 w-5 ${
                    isWatchlisted ? "fill-current" : ""
                  }`}
                />
              </button>
            </div>
          </div>

          {/* ──── 3. Candlestick chart ──── */}
          <div className="relative rounded-xl ring-1 ring-foreground/5 bg-card/20 overflow-hidden">
            {hoverData && (
              <div className="absolute top-4 left-4 z-10 bg-zinc-950/80 backdrop-blur-md border border-zinc-800 p-3 rounded-lg text-xs font-mono space-y-1.5 shadow-lg pointer-events-none select-none min-w-[140px]">
                <div className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider mb-1">
                  {showSP500 ? "Change (Visible Range)" : "Price"}
                </div>
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full" />
                    <span className="font-bold text-zinc-200">{symbol}</span>
                  </div>
                  <span className={`font-semibold tabular-nums ${
                    showSP500 
                      ? (hoverData.aapl >= 0 ? "text-emerald-500" : "text-red-500") 
                      : "text-zinc-200"
                  }`}>
                    {showSP500 
                      ? `${hoverData.aapl >= 0 ? "+" : ""}${hoverData.aapl.toFixed(2)}%` 
                      : `$${hoverData.aapl.toFixed(2)}`
                    }
                  </span>
                </div>
                {showSP500 && hoverData.sp500 !== null && (
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 bg-zinc-400 rounded-full" />
                      <span className="font-bold text-zinc-200">S&P 500</span>
                    </div>
                    <span className={`font-semibold tabular-nums ${hoverData.sp500 >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                      {hoverData.sp500 >= 0 ? "+" : ""}{hoverData.sp500.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            )}
            <div
              className="h-[320px] sm:h-[460px] w-full"
              ref={chartContainerRef}
            />
          </div>

          {/* ──── 4. Period pills row ──── */}
          <div className="flex flex-wrap items-center justify-between gap-3 mt-3 mb-2">
            <div className="flex flex-wrap items-center gap-1 bg-muted/30 p-1 rounded-lg">
              {PERIODS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => setPeriod(p.value)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                    period === p.value
                      ? "bg-foreground text-background shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-background/50"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowSP500(!showSP500)}
                className={`text-xs font-medium transition-colors ${
                  showSP500
                    ? "text-rose-500 hover:text-rose-400"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {showSP500 ? "Hide S&P 500" : "Compare to S&P 500"}
              </button>
              <button
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                aria-label="Chart settings"
              >
                <Settings className="h-4 w-4" />
              </button>
              <button
                onClick={handleFullscreen}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                aria-label="Fullscreen"
              >
                <Maximize2 className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* ──── 5. Position summary ──── */}
          <PositionSummary
            symbol={symbol}
            onBuyClick={() => {
              setModalDefaultAction("buy");
              setModalOpen(true);
            }}
            onSellClick={() => {
              setModalDefaultAction("sell");
              setModalOpen(true);
            }}
            refreshKey={positionRefreshKey}
          />

          {/* ──── 6. Details accordions ──── */}
          <DetailsAccordion
            dashboardData={dashboardData}
            symbol={symbol}
          />

          {/* ──── 7. Trade modal ──── */}
          <TradeModal
            open={modalOpen}
            onClose={() => setModalOpen(false)}
            symbol={symbol}
            currentPrice={comparisonData?.end_price}
            defaultAction={modalDefaultAction}
            onComplete={bumpRefreshKey}
          />
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
