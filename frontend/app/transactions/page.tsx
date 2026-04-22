"use client";

import { useState, useEffect } from "react";
import { fetchTransactions } from "@/lib/api";
import ProtectedRoute from "@/components/protected-route";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Transaction {
  id: number;
  timestamp: string;
  action: string;
  symbol: string;
  shares: number;
  price: number;
  total: number;
}

function TransactionContent() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const loadTransactions = async (pg = 1) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchTransactions(pg, 25);
      setTransactions(data.transactions);
      setTotalPages(data.pages);
      setPage(data.current_page);
    } catch {
      setError("Failed to load transactions");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTransactions();
  }, []);

  const formatDate = (isoString: string) => {
    if (!isoString) return "N/A";
    const date = new Date(isoString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      <h2 className="text-2xl font-extrabold mb-6 bg-gradient-to-r from-violet-500 to-blue-500 bg-clip-text text-transparent">
        Transaction History
      </h2>

      {error && (
        <div className="p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <Card className="border-border/50 bg-card/50">
          <CardContent className="pt-6">
            <div className="h-72 rounded-lg bg-gradient-to-r from-muted/50 to-muted/30 animate-pulse" />
          </CardContent>
        </Card>
      ) : transactions.length === 0 ? (
        <Card className="border-border/50 bg-card/50">
          <CardContent className="flex flex-col items-center py-12 text-muted-foreground">
            <p className="text-lg mb-1">No transactions yet</p>
            <p className="text-sm">
              Execute a trade from your Portfolio page to see history here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
            <CardContent className="p-0 overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-border/50">
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Date</TableHead>
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Action</TableHead>
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Symbol</TableHead>
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Shares</TableHead>
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Price</TableHead>
                    <TableHead className="text-xs font-semibold uppercase tracking-wider">Total</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {transactions.map((t) => (
                    <TableRow key={t.id} className="border-border/30">
                      <TableCell className="text-sm text-muted-foreground">
                        {formatDate(t.timestamp)}
                      </TableCell>
                      <TableCell>
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider ${
                            t.action === "buy"
                              ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30"
                              : "bg-red-500/10 text-red-500 border border-red-500/30"
                          }`}
                        >
                          {t.action}
                        </span>
                      </TableCell>
                      <TableCell className="font-semibold text-violet-400">
                        {t.symbol}
                      </TableCell>
                      <TableCell>{t.shares}</TableCell>
                      <TableCell>${t.price.toFixed(2)}</TableCell>
                      <TableCell className="font-semibold">
                        $
                        {t.total.toLocaleString(undefined, {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        })}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3 mt-6">
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadTransactions(page - 1)}
                disabled={page <= 1}
                className="gap-1"
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </Button>
              <span className="text-sm text-muted-foreground px-3">
                Page {page} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadTransactions(page + 1)}
                disabled={page >= totalPages}
                className="gap-1"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function TransactionHistoryPage() {
  return (
    <ProtectedRoute>
      <TransactionContent />
    </ProtectedRoute>
  );
}
