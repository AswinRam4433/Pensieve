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
      <h1>Image Search</h1>
      <SearchForm faces={faces} onSearch={handleSearch} loading={loading} />
      <SearchResults results={results} loading={loading} error={error} />
    </div>
  );
};

export default App;
