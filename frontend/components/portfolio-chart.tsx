"use client";

import { useEffect, useRef, useMemo } from "react";
import { createChart, ColorType, CrosshairMode, AreaSeries, LineType } from "lightweight-charts";

interface HistoryDataPoint {
  date: string;
  total_value: number;
  cash: number;
  stock_value: number;
}

interface PortfolioChartProps {
  history: HistoryDataPoint[];
  activeTimeframe: string;
  onTimeframeChange: (tf: string) => void;
}

const PERIODS = [
  { value: "5d", label: "5D" },
  { value: "1mo", label: "1M" },
  { value: "3mo", label: "3M" },
  { value: "6mo", label: "6M" },
  { value: "1y", label: "1Y" },
  { value: "2y", label: "2Y" },
  { value: "max", label: "MAX" },
];

export default function PortfolioChart({
  history,
  activeTimeframe,
  onTimeframeChange,
}: PortfolioChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const chartData = useMemo(() => {
    if (!history || history.length === 0) return [];
    
    return history
      .map((point) => ({
        time: point.date,
        value: point.total_value,
      }))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  }, [history]);

  useEffect(() => {
    if (!containerRef.current || chartData.length === 0) return;

    const chart = createChart(containerRef.current, {
      handleScroll: false,
      handleScale: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "transparent" },
        horzLines: { color: "rgba(55, 65, 81, 0.5)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
      },
    });

    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: "#10b981", // Emerald brand color
      topColor: "rgba(16, 185, 129, 0.4)",
      bottomColor: "rgba(16, 185, 129, 0.0)",
      lineWidth: 2,
      lineType: LineType.Curved, // Adds geometric curve smoothing
    });

    areaSeries.setData(chartData as any);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [chartData]);

  return (
    <div className="rounded-xl border border-border/50 bg-card/30 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/50">
        <h3 className="text-sm font-semibold text-foreground">Performance</h3>
        <div className="inline-flex rounded-md p-0.5 bg-background/50">
          {PERIODS.map((p) => (
            <button
              key={p.value}
              onClick={() => onTimeframeChange(p.value)}
              className={`px-2.5 py-1 text-[11px] font-semibold rounded transition-all cursor-pointer ${
                activeTimeframe === p.value
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
      
      {chartData.length === 0 ? (
        <div className="h-[400px] flex items-center justify-center p-6 text-center">
          <p className="text-muted-foreground">
            No historical data available yet. Start trading to see your portfolio's history!
          </p>
        </div>
      ) : (
        <div ref={containerRef} className="h-[400px] w-full" />
      )}
    </div>
  );
}
