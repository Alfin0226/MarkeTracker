"use client";

import { useState, useEffect } from "react";
import { fetchPortfolio, fetchPortfolioHistory } from "@/lib/api";
import PortfolioSummary from "@/components/portfolio-summary";
import PortfolioTable from "@/components/portfolio-table";
import PortfolioChart from "@/components/portfolio-chart";
import PortfolioHistoryTable from "@/components/portfolio-history-table";
import TradeForm from "@/components/trade-form";
import ProtectedRoute from "@/components/protected-route";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RefreshCw, Search } from "lucide-react";

function PortfolioContent() {
  const [portfolio, setPortfolio] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Set default dates: end_date = today, start_date = 3 months ago
  const defaultEndDate = new Date().toISOString().split('T')[0];
  const defaultStartDate = new Date(new Date().setDate(new Date().getDate() - 90)).toISOString().split('T')[0];

  const [startDate, setStartDate] = useState(defaultStartDate);
  const [endDate, setEndDate] = useState(defaultEndDate);
  const [activeTimeframe, setActiveTimeframe] = useState("3mo");

  const handleTimeframeChange = (range: string) => {
    setActiveTimeframe(range);
    const end = new Date();
    let start = new Date();

    switch(range) {
      case '5d':
        start.setDate(end.getDate() - 5);
        break;
      case '1mo':
        start.setMonth(end.getMonth() - 1);
        break;
      case '3mo':
        start.setMonth(end.getMonth() - 3);
        break;
      case '6mo':
        start.setMonth(end.getMonth() - 6);
        break;
      case '1y':
        start.setFullYear(end.getFullYear() - 1);
        break;
      case '2y':
        start.setFullYear(end.getFullYear() - 2);
        break;
      case 'max':
        start = (portfolio as any)?.created_at ? new Date((portfolio as any).created_at) : new Date('2020-01-01');
        break;
      default:
        start.setMonth(end.getMonth() - 3);
    }
    setEndDate(end.toISOString().split('T')[0]);
    setStartDate(start.toISOString().split('T')[0]);
  };

  const initialInvestment = 1000000;

  const loadPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const [portfolioData, historyResponse] = await Promise.all([
        fetchPortfolio(),
        fetchPortfolioHistory(startDate, endDate)
      ]);
      setPortfolio(portfolioData);
      setHistory(historyResponse?.history || []);
    } catch (err: unknown) {
      const error = err as { message?: string };
      setError(error.message || "Failed to load portfolio data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPortfolio();
  }, [startDate, endDate]);

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

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4 overflow-x-auto">
              {['5d', '1mo', '3mo', '6mo', '1y', '2y', 'max'].map((tf) => (
                <Button
                  key={tf}
                  variant={activeTimeframe === tf ? "default" : "outline"}
                  size="sm"
                  className={activeTimeframe === tf ? "bg-violet-600 hover:bg-violet-700 text-white" : ""}
                  onClick={() => handleTimeframeChange(tf)}
                >
                  {tf}
                </Button>
              ))}
            </div>
            <PortfolioChart history={history} />
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

          <div className="mt-8 border-t border-border/50 pt-8">
            <div className="flex flex-col sm:flex-row items-center justify-between mb-4 gap-4">
              <h3 className="text-lg font-semibold">Weekly Equity History</h3>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground font-medium">From:</span>
                <Input
                  type="date"
                  value={startDate}
                  onChange={(e) => {
                    setStartDate(e.target.value);
                    setActiveTimeframe("custom");
                  }}
                  className="w-[140px]"
                />
                <span className="text-sm text-muted-foreground font-medium">To:</span>
                <Input
                  type="date"
                  value={endDate}
                  onChange={(e) => {
                    setEndDate(e.target.value);
                    setActiveTimeframe("custom");
                  }}
                  className="w-[140px]"
                />
              </div>
            </div>
            <PortfolioHistoryTable history={history} />
          </div>
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
