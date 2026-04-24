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

    const chart = createChart(containerRef.current, {
      handleScroll: false,
      handleScale: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "transparent", // Hide text
      },
      grid: {
        vertLines: { color: "transparent", visible: false },
        horzLines: { color: "transparent", visible: false },
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
          top: 0.35,
          bottom: 0.35,
        },
      },
      timeScale: {
        visible: false,
        borderVisible: false,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      priceLineVisible: false,
      lastValueVisible: false,
    });

    candleSeries.setData(data as any);
    
    // Fit the content initially smoothly
    const margin = data.length * 0.4; // 40% margin on both sides to make candles thinner
    chart.timeScale().setVisibleLogicalRange({
      from: -margin,
      to: data.length - 1 + margin
    });

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
  }, [data, isPositive]);

  if (!data || !data.length) {
    return <div className="h-12 w-full flex items-center text-xs text-muted-foreground">No data</div>;
  }

  return <div ref={containerRef} className="h-12 w-full" />;
}
