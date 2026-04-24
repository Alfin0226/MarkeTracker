"use client";

import { useEffect, useRef, useMemo } from "react";
import { createChart, ColorType, CrosshairMode, AreaSeries } from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface HistoryDataPoint {
  date: string;
  total_value: number;
  cash: number;
  stock_value: number;
}

interface PortfolioChartProps {
  history: HistoryDataPoint[];
}

export default function PortfolioChart({ history }: PortfolioChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const chartData = useMemo(() => {
    if (!history || history.length === 0) return [];
    
    return history.map((point) => ({
      time: point.date,
      value: point.total_value,
    })).sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  }, [history]);

  useEffect(() => {
    if (!containerRef.current || chartData.length === 0) return;

    // Use dark theme matching the aesthetics
    const chart = createChart(containerRef.current, {
      handleScroll: false,
      handleScale: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "transparent" },
        horzLines: { color: "rgba(55, 65, 81, 0.5)" }, // #374151
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
      lineColor: "#8b5cf6", // violet-500
      topColor: "rgba(139, 92, 246, 0.4)",
      bottomColor: "rgba(139, 92, 246, 0.0)",
      lineWidth: 2,
    });

    areaSeries.setData(chartData as any);
    
    // Fit the content
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

  if (!chartData.length) {
    return (
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm h-[400px] flex items-center justify-center">
        <p className="text-muted-foreground">No historical data available yet. Start trading to see your portfolio's history!</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-lg">Performance Over Time</CardTitle>
          <CardDescription>Historical trend of your portfolio value</CardDescription>
        </CardHeader>
        <CardContent>
          <div ref={containerRef} className="h-[350px] w-full mt-4" />
        </CardContent>
      </Card>
    </div>
  );
}
