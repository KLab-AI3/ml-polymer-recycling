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
  // Defensive Programming: If there's no data, don't render anything.
  // This prevents errors if the spectrum prop is invalid.
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
    // ResponsiveContainer will now correctly use the height from the CSS
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={chartData}
        margin={{ top: 5, right: 20, left: -10, bottom: 20 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="wavenumber"
          type="number" // Explicitly set type for better scaling
          domain={["dataMin", "dataMax"]} // Ensure the full range is shown
          stroke="var(--color-text-tertiary)"
          fontSize={12}
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
          // Safer label formatter
          labelFormatter={(label: any) =>
            typeof label === "number"
              ? `Wavenumber: ${label.toFixed(2)} cm⁻¹`
              : ""
          }
        />
        <Legend wrapperStyle={{ fontSize: "14px", paddingTop: "10px" }} />
        <Line
          type="monotone"
          dataKey="intensity"
          // CORRECTED: Use --color-primary from our design system
          stroke="var(--color-primary)"
          strokeWidth={2}
          dot={false}
          name="Original"
        />
        {processedSpectrum && (
          <Line
            type="monotone"
            dataKey="processed"
            // CORRECTED: Use a valid color from our system, e.g., --color-success
            stroke="var(--color-success)"
            strokeWidth={2}
            dot={false}
            name="Processed"
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default SpectrumChart;
