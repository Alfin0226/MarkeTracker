"use client";



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

interface PortfolioSummaryProps {
  portfolio: PortfolioData | null;
  initialInvestment: number;
}

export default function PortfolioSummary({ portfolio, initialInvestment }: PortfolioSummaryProps) {
  if (!portfolio) return null;

  const gainLoss = portfolio.total_value - initialInvestment;
  const gainLossPercent = ((gainLoss / initialInvestment) * 100).toFixed(2);
  const isPositive = gainLoss >= 0;

  const isLowBalance = portfolio.cash_balance < portfolio.total_value * 0.05;

  const formatCurrency = (value: number) =>
    value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  return (
    <div className="space-y-2 mb-10">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Portfolio Value
        </p>
        <p className="text-5xl sm:text-6xl font-mono font-extrabold tabular-nums tracking-tight text-foreground mt-1">
          ${formatCurrency(portfolio.total_value)}
        </p>
      </div>

      <div className={`flex items-center gap-1.5 text-sm font-mono font-semibold ${isPositive ? "text-emerald-500" : "text-red-500"}`}>
        <span>{isPositive ? "▲" : "▼"}</span>
        <span>
          {isPositive ? "+" : ""}${formatCurrency(Math.abs(gainLoss))} ({isPositive ? "+" : ""}{gainLossPercent}%)
        </span>
        <span className="text-muted-foreground font-sans font-medium text-xs ml-1">all-time</span>
      </div>

      <div className="text-xs text-muted-foreground flex items-center flex-wrap gap-1.5 mt-2">
        <span>Cash:</span>
        <span className="font-mono font-medium text-foreground">${formatCurrency(portfolio.cash_balance)}</span>
        {isLowBalance && (
          <span 
            title="Your cash balance is below 5% of your portfolio. Consider selling a position to free up cash for new trades."
            className="ml-2 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/30 cursor-help"
          >
            Low balance
          </span>
        )}
      </div>
    </div>
  );
}
