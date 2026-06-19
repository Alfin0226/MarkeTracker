"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { executeTrade, fetchPortfolio } from "@/lib/api";
import { X } from "lucide-react";

interface PortfolioItem {
  symbol: string;
  shares: number;
  avg_price: number;
  current_price: number;
  value: number;
  gain_loss: number;
}

interface TradeModalProps {
  open: boolean;
  onClose: () => void;
  symbol: string;
  currentPrice?: number;
  defaultAction?: "buy" | "sell";
  onComplete?: () => void;
}

export default function TradeModal({
  open,
  onClose,
  symbol,
  currentPrice = 0,
  defaultAction = "buy",
  onComplete,
}: TradeModalProps) {
  const [action, setAction] = useState<"buy" | "sell">(defaultAction);
  const [shares, setShares] = useState("");
  const [cashBalance, setCashBalance] = useState(0);
  const [ownedShares, setOwnedShares] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const successTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Reset state when modal opens
  useEffect(() => {
    if (open) {
      setAction(defaultAction);
      setShares("");
      setError("");
      setSuccess("");
      setLoading(false);

      // Fetch portfolio for cash balance & owned shares
      fetchPortfolio()
        .then((data) => {
          setCashBalance(data.cash_balance || 0);
          const match = data.portfolio?.find(
            (p: PortfolioItem) =>
              p.symbol.toUpperCase() === symbol.toUpperCase()
          );
          setOwnedShares(match?.shares || 0);
        })
        .catch(() => {
          setCashBalance(0);
          setOwnedShares(0);
        });

      // Autofocus input after mount
      setTimeout(() => inputRef.current?.focus(), 100);
    }

    return () => {
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
        successTimerRef.current = null;
      }
    };
  }, [open, symbol, defaultAction]);

  const sharesNum = parseInt(shares) || 0;
  const estimatedTotal = sharesNum * currentPrice;
  const isBuy = action === "buy";
  const overspend = isBuy && estimatedTotal > cashBalance;
  const insufficientShares = !isBuy && sharesNum > ownedShares;
  const canSubmit =
    sharesNum > 0 && !overspend && !insufficientShares && !loading && !success;

  const maxShares = isBuy
    ? Math.floor(cashBalance / currentPrice)
    : ownedShares;

  const quickPicks = [1, 5, 10, 25];

  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;
    setError("");
    setLoading(true);

    try {
      const response = await executeTrade({
        symbol: symbol.toUpperCase(),
        shares: sharesNum,
        action,
      });

      if (response.transaction) {
        const price = response.transaction.price;
        setSuccess(
          `${isBuy ? "Bought" : "Sold"} ${sharesNum} @ $${price.toFixed(2)}`
        );
        onComplete?.();

        successTimerRef.current = setTimeout(() => {
          onClose();
        }, 1400);
      } else {
        setError("Invalid response from server");
      }
    } catch (err: unknown) {
      const e = err as {
        response?: { data?: { error?: string; details?: string } };
        message?: string;
      };
      setError(
        e.response?.data?.error ||
          e.response?.data?.details ||
          e.message ||
          "An unexpected error occurred"
      );
    } finally {
      setLoading(false);
    }
  }, [canSubmit, symbol, sharesNum, action, isBuy, onComplete, onClose]);

  if (!open) return null;

  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop className="fixed inset-0 z-50 bg-black/40 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <DialogPrimitive.Popup className="fixed z-50 outline-none w-full max-w-md
          bottom-0 left-0 right-0 rounded-t-2xl
          sm:bottom-auto sm:top-1/2 sm:left-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-2xl
          bg-popover text-popover-foreground ring-1 ring-foreground/10
          data-open:animate-in data-open:slide-in-from-bottom-full sm:data-open:slide-in-from-bottom-0 sm:data-open:fade-in-0 sm:data-open:zoom-in-95
          data-closed:animate-out data-closed:slide-out-to-bottom-full sm:data-closed:slide-out-to-bottom-0 sm:data-closed:fade-out-0 sm:data-closed:zoom-out-95
          duration-200"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 pb-2">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider">
                {symbol}
              </p>
              <p className="text-2xl font-mono tabular-nums font-bold">
                ${currentPrice.toFixed(2)}
              </p>
            </div>
            <DialogPrimitive.Close className="p-2 rounded-lg hover:bg-muted transition-colors">
              <X className="h-5 w-5 text-muted-foreground" />
              <span className="sr-only">Close</span>
            </DialogPrimitive.Close>
          </div>

          <div className="px-4 pb-4 space-y-4">
            {/* Buy / Sell toggle */}
            <div className="flex rounded-lg bg-muted/50 p-1">
              <button
                onClick={() => {
                  setAction("buy");
                  setShares("");
                  setError("");
                }}
                className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${
                  isBuy
                    ? "bg-emerald-500 text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Buy
              </button>
              <button
                onClick={() => {
                  setAction("sell");
                  setShares("");
                  setError("");
                }}
                className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${
                  !isBuy
                    ? "bg-red-500 text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Sell
              </button>
            </div>

            {/* Shares input */}
            <div>
              <label className="block text-xs text-muted-foreground uppercase tracking-wider mb-1.5">
                Shares
              </label>
              <input
                ref={inputRef}
                type="number"
                min="0"
                step="1"
                value={shares}
                onChange={(e) => {
                  setShares(e.target.value);
                  setError("");
                }}
                placeholder="0"
                className="w-full h-14 text-3xl font-mono tabular-nums font-bold text-center bg-transparent border border-input rounded-xl outline-none focus:border-ring focus:ring-2 focus:ring-ring/50 transition-colors [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>

            {/* Quick picks */}
            <div className="flex items-center gap-2">
              {quickPicks.map((n) => (
                <button
                  key={n}
                  onClick={() => setShares(n.toString())}
                  className="flex-1 py-1.5 text-sm font-mono rounded-lg border border-border hover:bg-muted transition-colors"
                >
                  {n}
                </button>
              ))}
              <button
                onClick={() => setShares(maxShares > 0 ? maxShares.toString() : "0")}
                className="flex-1 py-1.5 text-sm font-semibold rounded-lg border border-border hover:bg-muted transition-colors"
              >
                Max
              </button>
            </div>

            {/* Summary */}
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Estimated total</span>
                <span className="font-mono tabular-nums font-semibold">
                  ${estimatedTotal.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">
                  {isBuy ? "Available cash" : "Cash after sale"}
                </span>
                <span
                  className={`font-mono tabular-nums font-semibold ${
                    overspend ? "text-red-500" : ""
                  }`}
                >
                  $
                  {(isBuy
                    ? cashBalance
                    : cashBalance + estimatedTotal
                  ).toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </span>
              </div>
              {overspend && (
                <p className="text-red-500 text-xs font-medium">
                  Insufficient funds — reduce shares or add cash.
                </p>
              )}
              {insufficientShares && (
                <p className="text-red-500 text-xs font-medium">
                  You only own {ownedShares} share
                  {ownedShares !== 1 ? "s" : ""} of {symbol}.
                </p>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
                {error}
              </div>
            )}

            {/* Success */}
            {success && (
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 text-sm font-semibold text-center">
                {success}
              </div>
            )}

            {/* Submit */}
            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={`w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                isBuy
                  ? "bg-emerald-500 hover:bg-emerald-600"
                  : "bg-red-500 hover:bg-red-600"
              }`}
            >
              {loading
                ? "Processing…"
                : success
                ? "✓ Done"
                : `${isBuy ? "Buy" : "Sell"} ${symbol}`}
            </button>
          </div>
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
