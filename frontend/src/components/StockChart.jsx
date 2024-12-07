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
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <div style={{ height: '400px' }}>
      <Line data={chartData} options={{ maintainAspectRatio: false }} />
    </div>
  );
}

export default StockChart;