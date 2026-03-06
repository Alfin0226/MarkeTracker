import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function StockChart({ stockData, symbol }) {
  const chartData = {
    labels: stockData.dates,
    datasets: [
      {
        label: symbol,
        data: stockData.prices,
        fill: false,
        borderColor: '#6c5ce7',
        backgroundColor: 'rgba(108, 92, 231, 0.1)',
        tension: 0.3,
        pointRadius: 0,
        pointHitRadius: 10,
        borderWidth: 2,
      },
    ],
  };

  const options = {
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#9aa0b0',
          font: { family: 'Inter', weight: 500 },
        },
      },
      tooltip: {
        backgroundColor: '#252838',
        titleColor: '#e8eaed',
        bodyColor: '#e8eaed',
        borderColor: 'rgba(255,255,255,0.06)',
        borderWidth: 1,
        padding: 12,
        titleFont: { family: 'Inter', weight: 600 },
        bodyFont: { family: 'Inter' },
      },
    },
    scales: {
      x: {
        ticks: { color: '#5f6578', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        ticks: { color: '#5f6578', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
    },
  };

  return (
    <div style={{ height: '400px' }}>
      <Line data={chartData} options={options} />
    </div>
  );
}

export default StockChart;