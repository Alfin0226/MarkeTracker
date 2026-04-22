"use client";

import { Card, CardContent } from "@/components/ui/card";

interface PortfolioData {
  total_value: number;
  cash_balance: number;
}

interface PortfolioSummaryProps {
  portfolio: PortfolioData | null;
  initialInvestment: number;
}

export default function PortfolioSummary({ portfolio, initialInvestment }: PortfolioSummaryProps) {
  if (!portfolio) return null;

  const gainLoss = portfolio.total_value - initialInvestment;
  const gainLossPercent = ((gainLoss / initialInvestment) * 100).toFixed(2);
  const isPositive = gainLoss >= 0;

  const formatCurrency = (value: number) =>
    value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm transition-all hover:border-violet-500/20 hover:-translate-y-0.5">
        <CardContent className="pt-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Total Portfolio Value
          </p>
          <p className="text-2xl font-bold">${formatCurrency(portfolio.total_value)}</p>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur-sm transition-all hover:border-violet-500/20 hover:-translate-y-0.5">
        <CardContent className="pt-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Cash Balance
          </p>
          <p className="text-2xl font-bold">${formatCurrency(portfolio.cash_balance)}</p>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur-sm transition-all hover:border-violet-500/20 hover:-translate-y-0.5">
        <CardContent className="pt-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Total Gain/Loss
          </p>
          <p className={`text-2xl font-bold ${isPositive ? "text-emerald-500" : "text-red-500"}`}>
            {isPositive ? "+" : ""}${formatCurrency(gainLoss)}
            <span className="ml-2 text-sm opacity-80">
              ({isPositive ? "+" : ""}{gainLossPercent}%)
            </span>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
