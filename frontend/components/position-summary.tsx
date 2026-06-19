"use client";

import { useEffect, useState } from "react";
import { fetchPortfolio } from "@/lib/api";

interface PositionItem {
  symbol: string;
  shares: number;
  avg_price: number;
  current_price: number;
  value: number;
  gain_loss: number;
}

interface PositionSummaryProps {
  symbol: string;
  onBuyClick?: () => void;
  onSellClick?: () => void;
  refreshKey?: number;
}

export default function PositionSummary({
  symbol,
  onBuyClick,
  onSellClick,
  refreshKey,
}: PositionSummaryProps) {
  const [position, setPosition] = useState<PositionItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [hasPosition, setHasPosition] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchPortfolio();
        if (cancelled) return;

        const match = data.portfolio?.find(
          (p: PositionItem) =>
            p.symbol.toUpperCase() === symbol.toUpperCase()
        );

        if (match) {
          setPosition(match);
          setHasPosition(true);
        } else {
          setPosition(null);
          setHasPosition(false);
        }
      } catch {
        if (!cancelled) {
          setPosition(null);
          setHasPosition(false);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [symbol, refreshKey]);

  if (loading) {
    return (
      <div className="mt-6">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Your Investment
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 animate-pulse">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="rounded-xl bg-card/30 ring-1 ring-foreground/5 p-4"
            >
              <div className="h-3 w-16 bg-muted-foreground/20 rounded mb-2" />
              <div className="h-6 w-24 bg-muted-foreground/20 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!hasPosition || !position) {
    return (
      <div className="mt-6">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Your Investment
        </h3>
        <div className="rounded-xl bg-card/30 ring-1 ring-foreground/5 p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-muted-foreground text-sm">
            You don&apos;t own{" "}
            <span className="font-semibold text-foreground">{symbol}</span> yet
          </p>
          <button
            onClick={onBuyClick}
            className="px-5 py-2 rounded-lg bg-emerald-500 text-white font-semibold text-sm hover:bg-emerald-600 transition-colors"
          >
            Buy {symbol}
          </button>
        </div>
      </div>
    );
  }

  const returnPct =
    position.avg_price * position.shares !== 0
      ? (position.gain_loss / (position.avg_price * position.shares)) * 100
      : 0;
  const isPositive = position.gain_loss >= 0;

  const items = [
    {
      label: "Value",
      value: `$${position.value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      color: "text-foreground",
    },
    {
      label: "Return",
      value: `${isPositive ? "+" : ""}$${position.gain_loss.toLocaleString(
        undefined,
        { minimumFractionDigits: 2, maximumFractionDigits: 2 }
      )} (${isPositive ? "+" : ""}${returnPct.toFixed(2)}%)`,
      color: isPositive ? "text-emerald-500" : "text-red-500",
    },
    {
      label: "Shares",
      value: position.shares.toString(),
      color: "text-foreground",
    },
    {
      label: "Avg Price",
      value: `$${position.avg_price.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`,
      color: "text-foreground",
    },
  ];

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Your Investment
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={onBuyClick}
            className="px-4 py-1.5 rounded-lg bg-emerald-500 text-white font-semibold text-xs hover:bg-emerald-600 transition-colors"
          >
            Buy
          </button>
          <button
            onClick={onSellClick}
            className="px-4 py-1.5 rounded-lg bg-red-500 text-white font-semibold text-xs hover:bg-red-600 transition-colors"
          >
            Sell
          </button>
        </div>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((item) => (
          <div
            key={item.label}
            className="rounded-xl bg-card/30 ring-1 ring-foreground/5 p-4"
          >
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
              {item.label}
            </p>
            <p
              className={`text-lg font-semibold font-mono tabular-nums ${item.color}`}
            >
              {item.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
