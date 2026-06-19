"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { searchSymbols, fetchIndices, fetchWatchlist } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import LightweightSparkline from "@/components/lightweight-sparkline";
import { Card } from "@/components/ui/card";
import { Search, Loader2 } from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
}

interface IndexData {
  name: string;
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  history?: {
    time: string | number;
    open: number;
    high: number;
    low: number;
    close: number;
  }[];
}

interface WatchlistData {
  symbol: string;
  name: string;
  current_price: number;
  change: number;
  change_percent: number;
}

const QUICK_PICKS = [
  { symbol: "AAPL", name: "Apple" },
  { symbol: "TSLA", name: "Tesla" },
  { symbol: "NVDA", name: "Nvidia" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "GOOGL", name: "Alphabet" },
  { symbol: "AMZN", name: "Amazon" },
];

function DashboardSearchContent() {
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [indices, setIndices] = useState<IndexData[] | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistData[]>([]);
  
  const router = useRouter();
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        const [indicesData, watchlistData] = await Promise.all([
          fetchIndices().catch(() => null),
          fetchWatchlist().catch(() => ({ watchlist: [] }))
        ]);
        
        if (indicesData) setIndices(indicesData);
        if (watchlistData?.watchlist) setWatchlist(watchlistData.watchlist);
      } catch (err) {
        console.error("Failed to load dashboard data:", err);
      }
    };
    
    loadDashboardData();
  }, []);

  const handleSubmit = (e: React.FormEvent | React.KeyboardEvent) => {
    e.preventDefault();
    if (search.trim()) {
      router.push(`/dashboard/${search.trim().toUpperCase()}`);
    }
  };

  const fetchSuggestions = useCallback(async (query: string) => {
    if (query.length > 1) {
      setIsLoading(true);
      setError(null);
      try {
        const data = await searchSymbols(query);
        setSuggestions(data);
      } catch (err) {
        setError("Failed to fetch suggestions");
        setSuggestions([]);
      } finally {
        setIsLoading(false);
      }
    } else {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      fetchSuggestions(search);
    }, 300);
    return () => clearTimeout(debounceTimer);
  }, [search, fetchSuggestions]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value.toUpperCase());
  };

  const handleSuggestionClick = (sym: string) => {
    router.push(`/dashboard/${sym}`);
    setSearch("");
    setSuggestions([]);
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] p-6 md:p-10 max-w-[1400px] mx-auto space-y-10">
      {/* 1. Search hero */}
      <section>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">Find a stock to start tracking</h1>
        <p className="text-sm text-muted-foreground mt-1 mb-5">Search any company or ticker — autocomplete as you type.</p>
        <div className="relative max-w-2xl">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search e.g. 'Apple' or 'AAPL'"
            value={search}
            onChange={handleSearchChange}
            onKeyDown={(e) => { if (e.key === "Enter") handleSubmit(e); }}
            autoComplete="off"
            className="w-full pl-12 pr-4 py-4 text-base rounded-xl bg-card/50 border border-border/50 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 transition-all"
          />
          {isLoading && (
            <div className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50 max-h-64 overflow-y-auto flex items-center justify-center p-3">
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              <span className="text-sm">Searching...</span>
            </div>
          )}
          {error && (
            <div className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50 max-h-64 overflow-y-auto flex items-center justify-center p-3">
              <span className="text-sm text-red-500">{error}</span>
            </div>
          )}
          {!isLoading && !error && suggestions.length > 0 && (
            <ul className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-x-hidden overflow-y-auto z-[100] max-h-[320px]">
              {suggestions.map((s) => (
                <li
                  key={s.symbol}
                  onClick={() => handleSuggestionClick(s.symbol)}
                  className="px-4 py-3 cursor-pointer text-sm hover:bg-accent transition-colors border-b border-border/30 last:border-0"
                >
                  <strong className="text-emerald-400 font-mono">{s.symbol}</strong> - {s.name}
                </li>
              ))}
            </ul>
          )}
          {!isLoading && !error && suggestions.length === 0 && search.length > 1 && (
            <div className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50 max-h-64 overflow-y-auto flex items-center justify-center p-3">
              <span className="text-sm text-muted-foreground">No results found</span>
            </div>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground mr-1">Or try:</span>
          {QUICK_PICKS.map(p => (
            <button
              key={p.symbol}
              onClick={() => router.push(`/dashboard/${p.symbol}`)}
              className="px-3 py-1.5 rounded-full bg-card/40 border border-border/50 hover:border-emerald-500/50 hover:text-emerald-400 text-xs font-mono transition-colors"
            >
              {p.symbol} <span className="text-muted-foreground">· {p.name}</span>
            </button>
          ))}
        </div>
      </section>

      {/* 2. Market Indices */}
      <section>
        <h2 className="text-xl font-bold mb-4">Market Indices</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {indices ? (
            indices.map((idx) => {
              const isPositive = idx.change >= 0;
              const sign = isPositive ? "+" : "";
              const colorClass = isPositive ? "text-green-500" : "text-red-500";

              return (
                <Card key={idx.symbol} className="bg-[#0b0c0e] border border-zinc-800/85 card-interactive p-6 flex flex-col justify-between h-[200px]">
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-bold text-muted-foreground text-sm tracking-wide">{idx.name}</span>
                      <span className="text-[10px] font-mono bg-zinc-800/60 border border-zinc-700/50 text-zinc-400 px-2 py-0.5 rounded font-semibold tracking-wider">
                        {idx.symbol.replace("^", "")}
                      </span>
                    </div>
                    <div className="text-3xl font-extrabold tracking-tight text-foreground font-sans">
                      {idx.price ? idx.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-"}
                    </div>
                    <div className={`text-sm font-semibold mt-1.5 ${colorClass}`}>
                      {sign}{idx.change ? Math.abs(idx.change).toFixed(2) : "0.00"} ({sign}{idx.changePercent ? Math.abs(idx.changePercent).toFixed(2) : "0.00"}%)
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4 mt-auto pt-4 h-12">
                    <div className="flex-shrink-0 h-10 w-10 rounded-lg border border-zinc-800 bg-zinc-900/40 flex items-center justify-center text-zinc-400">
                      <svg role="img" viewBox="0 0 24 24" className="h-5 w-5 fill-current">
                        <title>TradingView</title>
                        <path d="M15.8654 8.2789c0 1.3541-1.0978 2.4519-2.452 2.4519-1.354 0-2.4519-1.0978-2.4519-2.452 0-1.354 1.0978-2.4518 2.452-2.4518 1.3541 0 2.4519 1.0977 2.4519 2.4519zM9.75 6H0v4.9038h4.8462v7.2692H9.75Zm8.5962 0H24l-5.1058 12.173h-5.6538z"/>
                      </svg>
                    </div>
                    <div className="flex-1 h-12 min-w-0">
                      <LightweightSparkline data={idx.history || []} isPositive={isPositive} />
                    </div>
                  </div>
                </Card>
              );
            })
          ) : (
            <>
              <Card className="bg-card/30 border border-border/50 p-6 animate-pulse">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
              <Card className="bg-card/30 border border-border/50 p-6 animate-pulse hidden md:block">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
              <Card className="bg-card/30 border border-border/50 p-6 animate-pulse hidden md:block">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
            </>
          )}
        </div>
      </section>

      {/* 3. Watchlist */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Watchlist ({watchlist.length})</h2>
          {watchlist.length > 0 && (
            <a href="/watchlist" className="text-sm text-emerald-400 hover:text-emerald-300">
              View all
            </a>
          )}
        </div>
        {watchlist.length > 0 ? (
          <Card className="bg-card/30 border border-border/50 p-0 overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-muted-foreground border-b border-border/50">
                <tr>
                  <th className="px-6 py-4 font-normal">Symbol</th>
                  <th className="px-6 py-4 font-normal hidden sm:table-cell">Company</th>
                  <th className="px-6 py-4 font-normal text-right">Price</th>
                  <th className="px-6 py-4 font-normal text-right">Change</th>
                  <th className="px-6 py-4 font-normal text-right hidden sm:table-cell">Change %</th>
                </tr>
              </thead>
              <tbody>
                {watchlist.slice(0, 6).map((item) => {
                  const isPositive = item.change >= 0;
                  const sign = isPositive ? "+" : "";
                  const colorClass = isPositive ? "text-green-500" : "text-red-500";
                  
                  return (
                    <tr
                      key={item.symbol}
                      className="border-b border-border/50 last:border-0 hover:bg-white/5 transition-colors cursor-pointer"
                      onClick={() => router.push(`/dashboard/${item.symbol}`)}
                    >
                      <td className="px-6 py-4 font-semibold text-emerald-400 font-mono">{item.symbol}</td>
                      <td className="px-6 py-4 truncate max-w-[150px] hidden sm:table-cell">
                        {item.name || "Loading..."}
                      </td>
                      <td className="px-6 py-4 text-right font-medium">
                        {item.current_price
                          ? item.current_price.toLocaleString("en-US", {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2,
                            })
                          : "-"}
                      </td>
                      <td className={`px-6 py-4 text-right ${colorClass}`}>
                        {sign}{item.change ? Math.abs(item.change).toFixed(2) : "0.00"}
                      </td>
                      <td className={`px-6 py-4 text-right hidden sm:table-cell ${colorClass}`}>
                        {sign}{item.change_percent ? Math.abs(item.change_percent).toFixed(2) : "0.00"}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
        ) : (
          <div className="rounded-xl border border-border/50 bg-card/30 p-10 text-center">
            <p className="text-sm font-semibold text-foreground mb-1">No instruments in your watchlist yet</p>
            <p className="text-xs text-muted-foreground">Search for a stock above to start tracking</p>
          </div>
        )}
      </section>
    </div>
  );
}

export default function DashboardSearchPage() {
  return (
    <ProtectedRoute>
      <DashboardSearchContent />
    </ProtectedRoute>
  );
}
