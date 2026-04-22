"use client";

import { useState, type FormEvent } from "react";
import { executeTrade } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, CheckCircle2 } from "lucide-react";

interface TradeFormProps {
  onTradeComplete?: () => void;
}

export default function TradeForm({ onTradeComplete }: TradeFormProps) {
  const [tradeSymbol, setTradeSymbol] = useState("");
  const [tradeShares, setTradeShares] = useState("");
  const [tradeAction, setTradeAction] = useState("buy");
  const [tradeError, setTradeError] = useState("");
  const [tradeSuccess, setTradeSuccess] = useState("");

  const handleTrade = async (e: FormEvent) => {
    e.preventDefault();
    setTradeError("");
    setTradeSuccess("");

    try {
      if (!tradeSymbol || !tradeShares || Number(tradeShares) <= 0) {
        setTradeError("Please enter valid symbol and number of shares");
        return;
      }

      const response = await executeTrade({
        symbol: tradeSymbol.toUpperCase(),
        shares: parseInt(tradeShares),
        action: tradeAction,
      });

      if (response.transaction) {
        const action = response.transaction.action || tradeAction;
        setTradeSuccess(
          `${action.toUpperCase()}: ${response.transaction.shares} shares of ${response.transaction.symbol} at $${response.transaction.price.toFixed(2)} — New Balance: $${response.transaction.new_balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        );
        setTradeSymbol("");
        setTradeShares("");

        if (onTradeComplete) {
          onTradeComplete();
        }
      } else {
        setTradeError("Invalid response from server");
      }
    } catch (error: unknown) {
      const err = error as { response?: { data?: { error?: string; details?: string } }; message?: string };
      const errorMessage =
        err.response?.data?.error ||
        err.response?.data?.details ||
        err.message ||
        "An unexpected error occurred while executing the trade";
      setTradeError(errorMessage);
    }
  };

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg">Execute Trade</CardTitle>
      </CardHeader>
      <CardContent>
        {tradeError && (
          <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {tradeError}
          </div>
        )}
        {tradeSuccess && (
          <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 text-sm">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            {tradeSuccess}
          </div>
        )}
        <form onSubmit={handleTrade}>
          <div className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_1fr_auto] gap-3 items-end">
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                Symbol
              </label>
              <Input
                type="text"
                placeholder="e.g., AAPL"
                value={tradeSymbol}
                onChange={(e) => setTradeSymbol(e.target.value.toUpperCase())}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                Shares
              </label>
              <Input
                type="number"
                placeholder="Qty"
                value={tradeShares}
                onChange={(e) => setTradeShares(e.target.value)}
                min="1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                Action
              </label>
              <select
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                value={tradeAction}
                onChange={(e) => setTradeAction(e.target.value)}
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <Button
                type="submit"
                className="whitespace-nowrap bg-gradient-to-r from-violet-600 to-blue-600 text-white hover:from-violet-700 hover:to-blue-700"
              >
                Execute Trade
              </Button>
            </div>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
