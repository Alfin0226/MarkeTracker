"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface HistoryDataPoint {
  date: string;
  total_value: number;
  cash: number;
  stock_value: number;
}

interface PortfolioHistoryTableProps {
  history: HistoryDataPoint[];
}

export default function PortfolioHistoryTable({ history }: PortfolioHistoryTableProps) {
  const tableData = useMemo(() => {
    if (!history || history.length === 0) return [];

    const sortedHistory = [...history].sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const initialCash = 1000000;

    return sortedHistory.map((point, index) => {
      const prevValue = index === 0 ? initialCash : sortedHistory[index - 1].total_value;
      const weeklyChangeDollar = point.total_value - prevValue;
      const weeklyChangePercent = prevValue === 0 ? 0 : (weeklyChangeDollar / prevValue) * 100;
      const cumulativeReturnPercent = ((point.total_value - initialCash) / initialCash) * 100;

      return {
        date: point.date,
        totalValue: point.total_value,
        changeDollar: weeklyChangeDollar,
        changePercent: weeklyChangePercent,
        cumulativeReturn: cumulativeReturnPercent,
      };
    }).reverse();
  }, [history]);

  if (!history || history.length === 0) {
    return null;
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm mt-6">
      <CardContent className="pt-6">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Date (End of Week)</TableHead>
              <TableHead className="text-right">Total Value</TableHead>
              <TableHead className="text-right">Weekly PnL ($)</TableHead>
              <TableHead className="text-right">Weekly PnL (%)</TableHead>
              <TableHead className="text-right">Cumulative Return</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {tableData.map((row) => (
              <TableRow key={row.date}>
                <TableCell>{new Date(row.date).toLocaleDateString()}</TableCell>
                <TableCell className="text-right">
                  ${row.totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </TableCell>
                <TableCell className={`text-right ${row.changeDollar >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {row.changeDollar >= 0 ? "+" : "-"}${Math.abs(row.changeDollar).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </TableCell>
                <TableCell className={`text-right ${row.changePercent >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {row.changePercent >= 0 ? "+" : ""}{row.changePercent.toFixed(2)}%
                </TableCell>
                <TableCell className={`text-right ${row.cumulativeReturn >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {row.cumulativeReturn >= 0 ? "+" : ""}{row.cumulativeReturn.toFixed(2)}%
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
