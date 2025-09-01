import React from "react";
import type { UnifiedSearchResults } from "../types";

export const ImageResult: React.FC<{
  result: string;
  distance: number;
  path: string;
}> = ({ result, distance, path }) => {
  return (
    <div className="image-result">
      <img
        src={result}
        alt="search result"
        style={{ maxWidth: "100%", maxHeight: "200px", objectFit: "contain" }}
      />
      {distance != null && (
        <div className="image-distance">Distance: {distance}</div>
      )}
      {path && <div className="image-path">Path: {path}</div>}
    </div>
  );
};
