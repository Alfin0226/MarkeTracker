"use client";

import { useState, useEffect } from "react";
import { fetchPortfolio, fetchPortfolioHistory } from "@/lib/api";
import PortfolioSummary from "@/components/portfolio-summary";
import PortfolioTable from "@/components/portfolio-table";
import PortfolioChart from "@/components/portfolio-chart";
import ProtectedRoute from "@/components/protected-route";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RefreshCw, Search } from "lucide-react";

interface Position {
  symbol: string;
  shares: number;
  avg_price: number;
  current_price: number;
  value: number;
  gain_loss: number;
}

interface PortfolioData {
  total_value: number;
  cash_balance: number;
  created_at?: string;
  portfolio: Position[];
}

interface HistoryDataPoint {
  date: string;
  total_value: number;
  cash: number;
  stock_value: number;
}

function PortfolioContent() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [history, setHistory] = useState<HistoryDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Set default dates: end = today, start = 5 days ago (default performance graph is 5d)
  const defaultEndDate = new Date().toISOString().split('T')[0];
  const defaultStartDate = (() => {
    const d = new Date();
    d.setDate(d.getDate() - 5);
    return d.toISOString().split('T')[0];
  })();

  const [startDate, setStartDate] = useState(defaultStartDate);
  const [endDate, setEndDate] = useState(defaultEndDate);
  const [activeTimeframe, setActiveTimeframe] = useState("5d");

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
        start = portfolio?.created_at ? new Date(portfolio.created_at) : new Date('2020-01-01');
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
      <h2 className="text-3xl font-extrabold tracking-tight text-foreground mb-8">
        Portfolio
      </h2>

      {error && (
        <div className="alert-error text-center">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="card-flat animate-pulse">
              <CardContent className="pt-6">
                <div className="h-20 rounded-lg bg-gradient-to-r from-muted/50 to-muted/30" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <>
          <div className="mb-10">
            <PortfolioSummary
              portfolio={portfolio}
              initialInvestment={initialInvestment}
            />
          </div>

          <div className="mb-8">
            <PortfolioChart 
              history={history} 
              activeTimeframe={activeTimeframe}
              onTimeframeChange={handleTimeframeChange}
            />
          </div>



          <Card className="card-glass overflow-hidden">
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
