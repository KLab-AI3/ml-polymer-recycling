type ReportHandler = (metric: any) => void;

const reportWebVitals = (onPerfEntry?: ReportHandler): void => {
  if (onPerfEntry && typeof onPerfEntry === "function") {
    import("web-vitals")
      .then((wv: any) => {
        const { getCLS, getFID, getFCP, getLCP, getTTFB } = wv as any;
        if (typeof getCLS === "function") getCLS(onPerfEntry);
        if (typeof getFID === "function") getFID(onPerfEntry);
        if (typeof getFCP === "function") getFCP(onPerfEntry);
        if (typeof getLCP === "function") getLCP(onPerfEntry);
        if (typeof getTTFB === "function") getTTFB(onPerfEntry);
      })
      .catch(() => {
        // optional: swallow import errors in environments where web-vitals isn't available
      });
  }
};

export default reportWebVitals;
