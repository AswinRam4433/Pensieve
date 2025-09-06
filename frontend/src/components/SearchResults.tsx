import React from "react";
import type { UnifiedSearchResults } from "../types";
import { ImageResult } from "./ImageResult";

interface SearchResultsProps {
  results: UnifiedSearchResults | null;
  loading: boolean;
  error?: string;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  loading,
  error,
}) => {
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
  if (loading) return <div className="results-loading">Loading results...</div>;
  if (error) return <div className="results-error">{error}</div>;
  if (!results) return <div className="results-empty">No results yet.</div>;

  return (
    <div className="search-results">
      {results.intersection && results.intersection.length > 0 && (
        <div className="results-section">
          <h3>Intersection (All Criteria Matched)</h3>
          <ul>
            {results.intersection.map((path, i) => (
              <li key={i}>{path}</li>
            ))}
          </ul>
        </div>
      )}
      {results.face && results.face.length > 0 && (
        <div className="results-section">
          <h3>Face Search Results</h3>
          {results.face.map((face, i) => (
            <div key={i} className="face-result">
              <div>Query Face #{face.query_face_index}</div>
              <ul>
                {face.similar_faces.map((sim, j) => (
                  <li key={j}>
                    {sim.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${sim.thumbnail}`}
                        alt="face thumbnail"
                        style={{
                          width: 48,
                          height: 48,
                          objectFit: "cover",
                          marginRight: 8,
                        }}
                      />
                    ) : (
                      <img
                        src={
                          sim.photo_path
                            ? `${API_BASE}/image?path=${encodeURIComponent(
                                sim.photo_path
                              )}`
                            : undefined
                        }
                        alt="face"
                        style={{
                          width: 48,
                          height: 48,
                          objectFit: "cover",
                          marginRight: 8,
                        }}
                      />
                    )}
                    {sim.photo_name} (Face #{sim.face_index}) - Similarity:{" "}
                    {(sim.similarity * 100).toFixed(1)}%
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
      {results.text && results.text.length > 0 && (
        <div className="results-section">
          <h3>Text-to-Image Results</h3>
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              flexWrap: "wrap",
              gap: 16,
              alignItems: "flex-start",
            }}
          >
            {results.text.map((res, i) => {
              const imgPath = res.image_path || "";
              const imageUrl = imgPath
                ? `${API_BASE}/image?path=${encodeURIComponent(imgPath)}`
                : "";
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                  }}
                >
                  <ImageResult
                    result={imageUrl}
                    similarity={res.score_percentage}
                    path={imgPath}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}
      {results.image && results.image.length > 0 && (
        <div className="results-section">
          <h3>Image-to-Image Results</h3>
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              flexWrap: "wrap",
              gap: 16,
              alignItems: "flex-start",
            }}
          >
            {results.image.map((res, i) => {
              const imgPath = res.metadata.file_path || res.metadata.url || "";
              const imageUrl = imgPath
                ? `${API_BASE}/image?path=${encodeURIComponent(imgPath)}`
                : "";
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                  }}
                >
                  <ImageResult
                    result={imageUrl}
                    distance={res.distance}
                    path={imgPath}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
