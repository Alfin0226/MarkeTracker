"use client";

import { useState, useEffect } from "react";
import { fetchPortfolio } from "@/lib/api";
import PortfolioSummary from "@/components/portfolio-summary";
import PortfolioTable from "@/components/portfolio-table";
import TradeForm from "@/components/trade-form";
import ProtectedRoute from "@/components/protected-route";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RefreshCw } from "lucide-react";

function PortfolioContent() {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const initialInvestment = 1000000;

  const loadPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPortfolio();
      setPortfolio(data);
    } catch (err: unknown) {
      const error = err as { message?: string };
      setError(error.message || "Failed to load portfolio");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPortfolio();
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      <h2 className="text-2xl font-extrabold mb-6 bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent">
        Portfolio
      </h2>

      {error && (
        <div className="p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="border-border/50 bg-card/50">
              <CardContent className="pt-6">
                <div className="h-20 rounded-lg bg-gradient-to-r from-muted/50 to-muted/30 animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <>
          <div className="mb-6">
            <PortfolioSummary
              portfolio={portfolio}
              initialInvestment={initialInvestment}
            />
          </div>

          <div className="mb-6">
            <TradeForm onTradeComplete={loadPortfolio} />
          </div>

          <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg">Holdings</CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              <PortfolioTable portfolio={portfolio} />
            </CardContent>
          </Card>
        </>
      )}

      <Button
        variant="outline"
        className="mt-6 gap-2"
        onClick={loadPortfolio}
        disabled={loading}
      >
        <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        {loading ? "Loading..." : "Refresh Portfolio"}
      </Button>
    </div>
  );
}

export default function PortfolioPage() {
  return (
    <ProtectedRoute>
      <PortfolioContent />
    </ProtectedRoute>
  );
}
