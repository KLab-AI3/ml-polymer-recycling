import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
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
  if (!spectrum || !spectrum.x_values || spectrum.x_values.length === 0) {
    return null;
  }

  const chartData = spectrum.x_values.map((x, index) => ({
    wavenumber: x,
    intensity: spectrum.y_values[index],
    processed: processedSpectrum
      ? processedSpectrum.y_values[index]
      : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={chartData}
        // FIX 1: Increased left and bottom margin to prevent labels from being cut off.
        margin={{ top: 10, right: 30, left: 20, bottom: 25 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="wavenumber"
          type="number"
          domain={["dataMin", "dataMax"]}
          stroke="var(--color-text-tertiary)"
          fontSize={12}
          // FIX 2: Added a tickFormatter to round the numbers on the axis.
          tickFormatter={(tick) => Math.round(tick).toString()}
          tick={{ fill: "var(--color-text-tertiary)" }}
          label={{
            value: "Wavenumber (cm⁻¹)",
            position: "insideBottom",
            offset: -15,
            fill: "var(--color-text-secondary)",
          }}
        />
        <YAxis
          stroke="var(--color-text-tertiary)"
          fontSize={12}
          tick={{ fill: "var(--color-text-tertiary)" }}
          label={{
            value: "Intensity",
            angle: -90,
            position: "insideLeft",
            // FIX 3: Added an offset to the Y-axis label to position it better.
            offset: -5,
            fill: "var(--color-text-secondary)",
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "var(--color-bg-component)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            boxShadow: "var(--shadow-sm)",
            fontSize: "12px",
          }}
          formatter={(value: any, name: string) => [
            typeof value === "number" ? value.toFixed(4) : value,
            name === "intensity" ? "Original" : "Processed",
          ]}
          labelFormatter={(label: any) =>
            typeof label === "number"
              ? `Wavenumber: ${label.toFixed(2)} cm⁻¹`
              : ""
          }
        />
        <Legend
          verticalAlign="top"
          height={40}
          wrapperStyle={{ fontSize: "14px", paddingTop: "10px" }}
        />
        <Line
          type="monotone"
          dataKey="intensity"
          stroke="var(--color-primary)"
          strokeWidth={2}
          dot={false}
          name="Original"
        />
        {processedSpectrum && (
          <Line
            type="monotone"
            dataKey="processed"
                  stroke="var(--color-success)"
                />
              )}
              </LineChart>
            </ResponsiveContainer>
  );
};

export default SpectrumChart;

