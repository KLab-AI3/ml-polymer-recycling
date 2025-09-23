import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { SpectrumData } from "../apiClient";

interface SpectrumChartProps {
  spectrum: SpectrumData;
  processedSpectrum?: SpectrumData;
}

const SpectrumChart: React.FC<SpectrumChartProps> = ({
  spectrum,
  processedSpectrum,
}) => {
  // Convert spectrum data to chart format
  const chartData = spectrum.x_values.map((x, index) => ({
    wavenumber: x,
    intensity: spectrum.y_values[index],
    processed: processedSpectrum
      ? processedSpectrum.y_values[index]
      : undefined,
  }));

  return (
    <div style={{ width: "100%", height: "300px" }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="wavenumber"
            stroke="#666"
            fontSize={12}
            label={{
              value: "Wavenumber (cm⁻¹)",
              position: "insideBottom",
              offset: -10,
            }}
          />
          <YAxis
            stroke="#666"
            fontSize={12}
            label={{ value: "Intensity", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#f8f9fa",
              border: "1px solid #dee2e6",
              borderRadius: "4px",
              fontSize: "12px",
            }}
            formatter={(value: any, name: string) => [
              typeof value === "number" ? value.toFixed(4) : value,
              name === "intensity" ? "Original" : "Processed",
            ]}
            labelFormatter={(label: any) => `Wavenumber: ${label} cm⁻¹`}
          />
          <Line
            type="monotone"
            dataKey="intensity"
            stroke="#ff6b6b"
            strokeWidth={1.5}
            dot={false}
            name="Original"
          />
          {processedSpectrum && (
            <Line
              type="monotone"
              dataKey="processed"
              stroke="#4ecdc4"
              strokeWidth={1.5}
              dot={false}
              name="Processed"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SpectrumChart;
