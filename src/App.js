import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [ticker, setTicker] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!ticker) {
      setError("Please enter a stock ticker.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/analyze/${ticker}`
      );
      setAnalysis(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Risk Analyzer</h1>
        <input
          type="text"
          placeholder="Enter stock ticker (e.g., RELIANCE.NS)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        {error && <p className="error">{error}</p>}
        {analysis && (
          <div>
            <h2>Risk Analysis for {ticker.toUpperCase()}</h2>
            <p>Risk Category: {analysis.analysis.risk_category}</p>
            <h3>Metrics:</h3>
            <ul>
              {Object.entries(analysis.metrics).map(([key, value]) => (
                <li key={key}>
                  {key}: {value}
                </li>
              ))}
            </ul>
            <h3>Analysis:</h3>
            <pre>{JSON.stringify(analysis.analysis, null, 2)}</pre>
            <h3>Plot:</h3>
            <img
              src={`data:image/png;base64,${analysis.plot}`}
              alt="Risk Analysis Plot"
              style={{ width: "100%", height: "auto" }}
            />
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
