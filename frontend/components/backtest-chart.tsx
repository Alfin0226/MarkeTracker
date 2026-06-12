"use client";

import { useRef, useEffect } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  Legend
);

interface DataPoint {
  date: string;
  value: number;
}

interface BacktestChartProps {
  portfolioSeries: DataPoint[];
  sp500Series: DataPoint[];
  initialInvestment: number;
}

export default function BacktestChart({
  portfolioSeries,
  sp500Series,
  initialInvestment,
}: BacktestChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current || portfolioSeries.length === 0) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    // Create gradient for portfolio line
    const portfolioGradient = ctx.createLinearGradient(0, 0, 0, 400);
    const isPositive =
      portfolioSeries[portfolioSeries.length - 1]?.value >= initialInvestment;
    if (isPositive) {
      portfolioGradient.addColorStop(0, "rgba(139, 92, 246, 0.3)");
      portfolioGradient.addColorStop(1, "rgba(139, 92, 246, 0.0)");
    } else {
      portfolioGradient.addColorStop(0, "rgba(239, 68, 68, 0.3)");
      portfolioGradient.addColorStop(1, "rgba(239, 68, 68, 0.0)");
    }

    // Format labels: show shorter dates
    const labels = portfolioSeries.map((p) => {
      const d = new Date(p.date);
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
    });

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Portfolio",
            data: portfolioSeries.map((p) => p.value),
            borderColor: isPositive
              ? "rgba(139, 92, 246, 1)"
              : "rgba(239, 68, 68, 1)",
            backgroundColor: portfolioGradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: isPositive
              ? "rgba(139, 92, 246, 1)"
              : "rgba(239, 68, 68, 1)",
          },
          {
            label: "S&P 500",
            data: sp500Series.map((p) => p.value),
            borderColor: "rgba(100, 116, 139, 0.7)",
            backgroundColor: "transparent",
            borderWidth: 1.5,
            borderDash: [6, 4],
            fill: false,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: "rgba(100, 116, 139, 1)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            align: "end",
            labels: {
              color: "rgba(255, 255, 255, 0.7)",
              usePointStyle: true,
              pointStyle: "line",
              padding: 20,
              font: { size: 12, family: "Inter, sans-serif" },
            },
          },
          tooltip: {
            backgroundColor: "rgba(15, 15, 25, 0.95)",
            titleColor: "rgba(255, 255, 255, 0.9)",
            bodyColor: "rgba(255, 255, 255, 0.8)",
            borderColor: "rgba(139, 92, 246, 0.3)",
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { size: 12, family: "Inter, sans-serif" },
            bodyFont: { size: 13, family: "Inter, sans-serif" },
            callbacks: {
              label: (context) => {
                const val = context.parsed.y;
                if (val == null) return "";
                const formatted = val.toLocaleString("en-US", {
                  style: "currency",
                  currency: "USD",
                });
                return `${context.dataset.label}: ${formatted}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: {
              color: "rgba(255,255,255,0.4)",
              maxTicksLimit: 10,
              font: { size: 11, family: "Inter, sans-serif" },
            },
          },
          y: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: {
              color: "rgba(255,255,255,0.4)",
              font: { size: 11, family: "Inter, sans-serif" },
              callback: (value) =>
                "$" +
                Number(value).toLocaleString("en-US", {
                  maximumFractionDigits: 0,
                }),
            },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [portfolioSeries, sp500Series, initialInvestment]);

  return (
    <div className="w-full h-[400px]">
      <canvas ref={canvasRef} />
    </div>
  );
}
