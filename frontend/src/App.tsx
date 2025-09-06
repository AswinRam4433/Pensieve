import React, { useEffect, useState } from "react";
import { fetchFaces, unifiedSearch } from "./api";
import type { SearchQuery } from "./api";
import type { FaceInfo, UnifiedSearchResults } from "./types";
import { SearchForm } from "./components/SearchForm";
import { SearchResults } from "./components/SearchResults";
import "./App.css";

const App: React.FC = () => {
  const [faces, setFaces] = useState<FaceInfo[]>([]);
  const [results, setResults] = useState<UnifiedSearchResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);

  useEffect(() => {
    fetchFaces(true)
      .then(setFaces)
      .catch(() => setFaces([]));
  }, []);

  const handleSearch = async (query: SearchQuery) => {
    setLoading(true);
    setError(undefined);
    setResults(null);
    try {
      const res = await unifiedSearch(query);
      setResults(res);
    } catch (e: any) {
      setError(e.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100%",
          zIndex: 1000,
          boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
          padding: "16px 0",
        }}
      >
        <h1 style={{ margin: "0 0 12px 0", textAlign: "center" }}>
          Image Search
        </h1>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <SearchForm faces={faces} onSearch={handleSearch} loading={loading} />
        </div>
      </div>
      <div style={{ marginTop: "120px" }}>
        <SearchResults results={results} loading={loading} error={error} />
      </div>
    </div>
  );
};

export default App;
