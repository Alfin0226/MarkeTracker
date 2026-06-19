"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, CandlestickSeries } from "lightweight-charts";

interface OHLCData {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface LightweightSparklineProps {
  data: OHLCData[];
  isPositive: boolean;
}

export default function LightweightSparkline({ data, isPositive }: LightweightSparklineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    const chartData = data
      .map((d) => {
        let timeVal: number;
        if (typeof d.time === "number") {
          timeVal = d.time > 100000000000 ? Math.floor(d.time / 1000) : d.time;
        } else {
          timeVal = Math.floor(new Date(d.time).getTime() / 1000);
        }
        return {
          time: timeVal,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        };
      })
      .sort((a, b) => a.time - b.time);

    // Filter out duplicates
    const uniqueChartData: typeof chartData = [];
    const seenTimes = new Set<number>();
    for (const point of chartData) {
      if (!seenTimes.has(point.time)) {
        seenTimes.add(point.time);
        uniqueChartData.push(point);
      }
    }

    if (uniqueChartData.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 48,
      handleScroll: false,
      handleScale: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "transparent",
        attributionLogo: false,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      crosshair: {
        mode: 0,
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
      leftPriceScale: {
        visible: false,
      },
      rightPriceScale: {
        visible: false,
        borderVisible: false,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        visible: false,
        borderVisible: false,
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });

    candlestickSeries.setData(uniqueChartData as any);
    
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight || 48
        });
      }
    };
    
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data]);

  if (!data || !data.length) {
    return <div className="h-12 w-full flex items-center text-xs text-muted-foreground">No data</div>;
  }

  return (
    <div className="relative w-full h-12">
      <style dangerouslySetInnerHTML={{ __html: `
        #tv-attr-logo, [class*="tv-attr-logo"], a[href*="tradingview.com"] {
          display: none !important;
        }
      `}} />
      <div ref={containerRef} className="h-12 w-full" />
    </div>
  );
}
