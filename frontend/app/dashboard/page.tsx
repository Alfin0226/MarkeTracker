
"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { searchSymbols, fetchIndices, fetchWatchlist } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import LightweightSparkline from "@/components/lightweight-sparkline";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  history?: number[];
}

interface WatchlistData {
  symbol: string;
  name: string;
  current_price: number;
  change: number;
  change_percent: number;
}

function DashboardSearchContent() {
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [indices, setIndices] = useState<IndexData[] | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistData[]>([]);
  
  const router = useRouter();

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

  const handleSubmit = (e: React.FormEvent) => {
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
    <div className="min-h-[calc(100vh-3.5rem)] p-6 md:p-10 max-w-[1400px] mx-auto space-y-8">
      <div>
        <h2 className="text-xl font-bold mb-4">Market Indices</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {indices ? (
            indices.map((idx) => {
              const isPositive = idx.change >= 0;
              const sign = isPositive ? "+" : "";
              const colorClass = isPositive ? "text-green-500" : "text-red-500";

              return (
                <Card key={idx.symbol} className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl p-6">
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-semibold text-sm">{idx.name}</span>
                    <span className="text-xs bg-muted px-2 py-1 rounded text-muted-foreground">
                      {idx.symbol.replace("^", "")}
                    </span>
                  </div>
                  <div className="text-3xl font-bold mb-1 mt-2">
                    {idx.price ? idx.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-"}
                  </div>
                  <div className={`text-sm font-medium mb-4 ${colorClass}`}>
                    {sign}{idx.change ? Math.abs(idx.change).toFixed(2) : "0.00"} ({sign}{idx.changePercent ? Math.abs(idx.changePercent).toFixed(2) : "0.00"}%)
                  </div>
                  <div className="h-12 w-full mt-4">
                    <LightweightSparkline data={idx.history || []} isPositive={isPositive} />
                  </div>
                </Card>
              );
            })
          ) : (
            <>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl p-6 animate-pulse hidden md:block">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl p-6 animate-pulse hidden md:block">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl p-6 animate-pulse hidden md:block">
                <div className="h-4 bg-muted rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
                <div className="h-12 bg-muted rounded w-full"></div>
              </Card>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl font-bold mb-4">Search Companies</h2>
          <Card className="w-full border-border/50 bg-card/50 backdrop-blur-sm shadow-xl text-center overflow-visible">
            <CardHeader>
              <CardTitle className="text-xl font-bold">
                US Equities & Global ETF Data
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Search by name or stock symbol.
              </p>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="relative mx-auto max-w-sm">
                <Input
                  type="text"
                  placeholder="e.g., 'Apple' or 'AAPL'"
                  value={search}
                  onChange={handleSearchChange}
                  autoComplete="off"
                  className="text-center"
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
                        <strong className="text-violet-400">{s.symbol}</strong> - {s.name}
                      </li>
                    ))}
                  </ul>
                )}
                {!isLoading && !error && suggestions.length === 0 && search.length > 1 && (
                  <div className="absolute left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50 max-h-64 overflow-y-auto flex items-center justify-center p-3">
                    <span className="text-sm text-muted-foreground">No results found</span>
                  </div>
                )}
                <Button
                  type="submit"
                  className="w-32 mt-6 gap-2 bg-gradient-to-r from-violet-600 to-blue-600 text-white hover:from-violet-700 hover:to-blue-700 mx-auto flex"
                  disabled={isLoading}
                >
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  Search
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        <div>
           <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Watchlist ({watchlist.length})</h2>
            <a href="/watchlist" className="text-sm text-violet-400 hover:text-violet-300 transition-colors">View All</a>
          </div>
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl p-0 overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-muted-foreground border-b border-border/50">
                <tr>
                  <th className="px-6 py-4 font-normal">Symbol</th>
                  <th className="px-6 py-4 font-normal">Company</th>
                  <th className="px-6 py-4 font-normal text-right">Price</th>
                  <th className="px-6 py-4 font-normal text-right">Change</th>
                  <th className="px-6 py-4 font-normal text-right">Change %</th>
                </tr>
              </thead>
              <tbody>
                {watchlist.slice(0, 6).map((item) => {
                  const isPositive = item.change >= 0;
                  const sign = isPositive ? "+" : "";
                  const colorClass = isPositive ? "text-green-500" : "text-red-500";
                  
                  return (
                    <tr key={item.symbol} className="border-b border-border/50 last:border-0 hover:bg-white/5 transition-colors cursor-pointer" onClick={() => router.push(`/dashboard/${item.symbol}`)}>
                      <td className="px-6 py-4 font-semibold text-violet-400">{item.symbol}</td>
                      <td className="px-6 py-4 truncate max-w-[150px]">{item.name || "Loading..."}</td>
                      <td className="px-6 py-4 text-right font-medium">
                        {item.current_price ? item.current_price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "-"}
                      </td>
                      <td className={`px-6 py-4 text-right ${colorClass}`}>
                        {sign}{item.change ? Math.abs(item.change).toFixed(2) : "0.00"}
                      </td>
                      <td className={`px-6 py-4 text-right ${colorClass}`}>
                        {sign}{item.change_percent ? Math.abs(item.change_percent).toFixed(2) : "0.00"}%
                      </td>
                    </tr>
                  );
                })}
                {watchlist.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                      No instruments in your watchlist yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </Card>
        </div>
      </div>
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

