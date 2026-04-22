"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { fetchWatchlist, removeFromWatchlist } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { RefreshCw, X } from "lucide-react";

interface WatchlistItem {
  symbol: string;
  current_price: number | null;
  added_at: string;
}

function WatchlistContent() {
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const loadWatchlist = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWatchlist();
      setWatchlist(data.watchlist);
    } catch {
      setError("Failed to load watchlist");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWatchlist();
  }, []);

  const handleRemove = async (symbol: string) => {
    try {
      await removeFromWatchlist(symbol);
      setWatchlist((prev) => prev.filter((item) => item.symbol !== symbol));
    } catch {
      setError("Failed to remove from watchlist");
    }
  };

  const formatDate = (isoString: string) => {
    if (!isoString) return "N/A";
    return new Date(isoString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      <h2 className="text-2xl font-extrabold mb-6 bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent">
        Watchlist
      </h2>

      {error && (
        <div className="p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="border-border/50 bg-card/50">
              <CardContent className="pt-6">
                <div className="h-20 rounded-lg bg-gradient-to-r from-muted/50 to-muted/30 animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : watchlist.length === 0 ? (
        <Card className="border-border/50 bg-card/50">
          <CardContent className="flex flex-col items-center py-12 text-muted-foreground">
            <p className="text-4xl mb-3">👀</p>
            <p className="text-lg mb-1">Your watchlist is empty</p>
            <p className="text-sm">
              Add stocks from the Dashboard to keep an eye on them.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {watchlist.map((item) => (
            <Card
              key={item.symbol}
              className="relative border-border/50 bg-card/50 backdrop-blur-sm cursor-pointer transition-all hover:border-violet-500/30 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-violet-500/5"
            >
              <CardContent
                className="pt-6"
                onClick={() => router.push(`/dashboard/${item.symbol}`)}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-lg font-bold text-violet-400">
                    {item.symbol}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Added {formatDate(item.added_at)}
                  </span>
                </div>
                <div className="text-2xl font-bold">
                  {item.current_price !== null
                    ? `$${item.current_price.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}`
                    : "Price unavailable"}
                </div>
              </CardContent>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(item.symbol);
                }}
                className="absolute top-3 right-3 p-1 rounded-md text-muted-foreground hover:text-red-500 hover:bg-red-500/10 transition-colors"
                title="Remove from watchlist"
              >
                <X className="h-4 w-4" />
              </button>
            </Card>
          ))}
        </div>
      )}

      <Button
        variant="outline"
        className="mt-6 gap-2"
        onClick={loadWatchlist}
        disabled={loading}
      >
        <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        {loading ? "Loading..." : "Refresh"}
      </Button>
    </div>
  );
}

export default function WatchlistPage() {
  return (
    <ProtectedRoute>
      <WatchlistContent />
    </ProtectedRoute>
  );
}
