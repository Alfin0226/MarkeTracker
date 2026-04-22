"use client";

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
  portfolio: Position[];
}

interface PortfolioTableProps {
  portfolio: PortfolioData | null;
}

export default function PortfolioTable({ portfolio }: PortfolioTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow className="border-border/50">
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Symbol</TableHead>
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Shares</TableHead>
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Avg Buy Price</TableHead>
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Current Price</TableHead>
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Value</TableHead>
          <TableHead className="text-xs font-semibold uppercase tracking-wider">Gain/Loss</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {portfolio?.portfolio?.length ? (
          portfolio.portfolio.map((position) => {
            const glPercent =
              typeof position.gain_loss === "number"
                ? ((position.gain_loss / (position.avg_price * position.shares)) * 100).toFixed(2)
                : null;

            return (
              <TableRow key={position.symbol} className="border-border/30 transition-colors">
                <TableCell className="font-semibold text-violet-400">
                  {position.symbol}
                </TableCell>
                <TableCell>{position.shares}</TableCell>
                <TableCell>
                  {typeof position.avg_price === "number"
                    ? `$${position.avg_price.toFixed(2)}`
                    : "N/A"}
                </TableCell>
                <TableCell>
                  {typeof position.current_price === "number"
                    ? `$${position.current_price.toFixed(2)}`
                    : "N/A"}
                </TableCell>
                <TableCell>
                  {typeof position.value === "number"
                    ? `$${position.value.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}`
                    : "N/A"}
                </TableCell>
                <TableCell>
                  {typeof position.gain_loss === "number" ? (
                    <span
                      className={
                        position.gain_loss >= 0
                          ? "text-emerald-500"
                          : "text-red-500"
                      }
                    >
                      {position.gain_loss >= 0 ? "+" : ""}$
                      {position.gain_loss.toFixed(2)}
                      <span className="ml-1 text-xs opacity-70">
                        ({position.gain_loss >= 0 ? "+" : ""}
                        {glPercent}%)
                      </span>
                    </span>
                  ) : (
                    "N/A"
                  )}
                </TableCell>
              </TableRow>
            );
          })
        ) : (
          <TableRow>
            <TableCell
              colSpan={6}
              className="text-center text-muted-foreground py-8"
            >
              No holdings yet — execute a trade above to get started
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
