"use client";

import { useRouter } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

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

interface PortfolioTableProps {
  portfolio: PortfolioData | null;
}

function Sparkline({ data, positive }: { data: number[]; positive: boolean }) {
  if (!data || data.length === 0) return <div className="w-16 h-6" />;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * 64;
    const y = 24 - ((v - min) / range) * 24;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width="64" height="24" className="text-muted-foreground inline-block">
      <polyline
        points={points}
        fill="none"
        stroke={positive ? "#10b981" : "#ef4444"}
        strokeWidth="1.5"
      />
    </svg>
  );
}

const generateMockSparklineData = (symbol: string, isPositive: boolean) => {
  const data = [];
  const charsSum = symbol.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
  let seed = charsSum;
  let currentVal = 100;
  for (let i = 0; i < 10; i++) {
    seed = (seed * 9301 + 49297) % 233280;
    const rnd = seed / 233280;
    currentVal += (rnd - 0.47) * 15;
    data.push(currentVal);
  }
  if (isPositive && data[data.length - 1] < data[0]) {
    data[data.length - 1] = data[0] + 15;
  } else if (!isPositive && data[data.length - 1] > data[0]) {
    data[data.length - 1] = data[0] - 15;
  }
  return data;
};

export default function PortfolioTable({ portfolio }: PortfolioTableProps) {
  const router = useRouter();
  const totalVal = portfolio?.total_value || 1;

  return (
    <Table>
      <TableHeader>
        <TableRow className="border-border/50">
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground pl-6 text-left">
            Symbol
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right">
            Shares
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right">
            Avg Cost
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right">
            Value
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right">
            % Port
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-center">
            30D
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right">
            Gain/Loss
          </TableHead>
          <TableHead className="px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground text-right pr-6 w-10">
            {/* Action cell */}
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {portfolio?.portfolio?.length ? (
          portfolio.portfolio.map((position) => {
            const totalCost = position.avg_price * position.shares;
            const glPercent =
              typeof position.gain_loss === "number" && totalCost > 0
                ? ((position.gain_loss / totalCost) * 100).toFixed(2)
                : null;
            
            const isPositive = position.gain_loss > 0;
            const isNegative = position.gain_loss < 0;
            
            const rowTint = isPositive
              ? "bg-emerald-500/[0.03] hover:bg-emerald-500/[0.06]"
              : isNegative
                ? "bg-red-500/[0.03] hover:bg-red-500/[0.06]"
                : "hover:bg-card/50";

            const allocation = typeof position.value === "number"
              ? ((position.value / totalVal) * 100).toFixed(1) + "%"
              : "0.0%";

            const sparklineData = generateMockSparklineData(position.symbol, isPositive);

            return (
              <TableRow
                key={position.symbol}
                onClick={() => router.push(`/dashboard/${position.symbol}`)}
                className={`border-border/30 ${rowTint} cursor-pointer transition-colors group`}
              >
                <TableCell className="px-3 py-2.5 font-mono font-semibold text-foreground pl-6 text-left">
                  {position.symbol}
                </TableCell>
                <TableCell className="px-3 py-2.5 font-mono tabular-nums text-right text-muted-foreground">
                  {position.shares}
                </TableCell>
                <TableCell className="px-3 py-2.5 font-mono tabular-nums text-right">
                  {typeof position.avg_price === "number"
                    ? `$${position.avg_price.toFixed(2)}`
                    : "N/A"}
                </TableCell>
                <TableCell className="px-3 py-2.5 font-mono tabular-nums text-right">
                  {typeof position.value === "number"
                    ? `$${position.value.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}`
                    : "N/A"}
                </TableCell>
                <TableCell className="px-3 py-2.5 font-mono tabular-nums text-right text-muted-foreground/80">
                  {allocation}
                </TableCell>
                <TableCell className="px-3 py-2.5 text-center">
                  <Sparkline data={sparklineData} positive={isPositive} />
                </TableCell>
                <TableCell className="px-3 py-2.5 font-mono tabular-nums text-right">
                  {typeof position.gain_loss === "number" ? (
                    <span className={isPositive ? "text-emerald-500" : isNegative ? "text-red-500" : "text-foreground"}>
                      {position.gain_loss >= 0 ? "+" : "−"}${Math.abs(position.gain_loss).toFixed(2)}
                      {glPercent !== null && (
                        <span className="ml-1 text-[10px] opacity-70">
                          ({position.gain_loss >= 0 ? "+" : "−"}{Math.abs(Number(glPercent)).toFixed(2)}%)
                        </span>
                      )}
                    </span>
                  ) : (
                    "N/A"
                  )}
                </TableCell>
                <TableCell className="px-3 py-2.5 text-right pr-6">
                  <span className="text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                    →
                  </span>
                </TableCell>
              </TableRow>
            );
          })
        ) : (
          <TableRow>
            <TableCell
              colSpan={8}
              className="text-center text-muted-foreground py-12"
            >
              No holdings yet — buy stocks from the dashboard to get started
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
