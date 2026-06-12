"use client";

import { useRef, useEffect } from "react";
import { Chart, DoughnutController, ArcElement, Tooltip, Legend } from "chart.js";

Chart.register(DoughnutController, ArcElement, Tooltip, Legend);

interface AllocationItem {
  symbol: string;
  weight: number;
}

interface BacktestAllocationChartProps {
  allocations: AllocationItem[];
}

const PALETTE = [
  "rgba(139, 92, 246, 0.85)",   // violet
  "rgba(59, 130, 246, 0.85)",   // blue
  "rgba(16, 185, 129, 0.85)",   // emerald
  "rgba(245, 158, 11, 0.85)",   // amber
  "rgba(236, 72, 153, 0.85)",   // pink
  "rgba(6, 182, 212, 0.85)",    // cyan
  "rgba(249, 115, 22, 0.85)",   // orange
  "rgba(168, 85, 247, 0.85)",   // purple
  "rgba(34, 197, 94, 0.85)",    // green
  "rgba(239, 68, 68, 0.85)",    // red
];

const PALETTE_BORDER = [
  "rgba(139, 92, 246, 1)",
  "rgba(59, 130, 246, 1)",
  "rgba(16, 185, 129, 1)",
  "rgba(245, 158, 11, 1)",
  "rgba(236, 72, 153, 1)",
  "rgba(6, 182, 212, 1)",
  "rgba(249, 115, 22, 1)",
  "rgba(168, 85, 247, 1)",
  "rgba(34, 197, 94, 1)",
  "rgba(239, 68, 68, 1)",
];

export default function BacktestAllocationChart({
  allocations,
}: BacktestAllocationChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current || allocations.length === 0) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    chartRef.current = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: allocations.map((a) => a.symbol),
        datasets: [
          {
            data: allocations.map((a) => a.weight),
            backgroundColor: allocations.map((_, i) => PALETTE[i % PALETTE.length]),
            borderColor: allocations.map((_, i) => PALETTE_BORDER[i % PALETTE_BORDER.length]),
            borderWidth: 2,
            hoverOffset: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              color: "rgba(255, 255, 255, 0.7)",
              usePointStyle: true,
              pointStyle: "circle",
              padding: 16,
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
            callbacks: {
              label: (context) => {
                return `${context.label}: ${context.parsed}%`;
              },
            },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [allocations]);

  return (
    <div className="w-full h-[280px]">
      <canvas ref={canvasRef} />
    </div>
  );
}
