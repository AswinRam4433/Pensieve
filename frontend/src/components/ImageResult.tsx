import React from "react";
import type { UnifiedSearchResults } from "../types";

export const ImageResult: React.FC<{
  result: string;
  distance?: number;
  path: string;
  similarity?: number;
}> = ({ result, distance, path, similarity }) => {
  return (
    <div
      className="polaroid-image-result"
      style={{
        background: "#fff",
        border: "8px solid #fff",
        boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
        borderRadius: "8px",
        padding: "12px 12px 32px 12px",
        display: "inline-block",
        position: "relative",
        maxWidth: "240px",
        textAlign: "center",
      }}
    >
      <img
        src={result}
        alt="search result"
        style={{
          height: "180px", // fixed height for all images
          width: "auto", // width adjusts to aspect ratio
          maxWidth: "100%",
          objectFit: "contain",
          borderRadius: "4px",
          background: "#eee",
          display: "block",
          margin: "0 auto",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: "8px",
          right: "12px",
          fontSize: "0.35em",
          color: "#333",
          fontWeight: 500,
          opacity: 0.85,
          maxWidth: "90%",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          direction: "rtl",
        }}
        title={path}
      >
        {path}
      </div>
      {distance != null && (
        <div
          className="image-distance"
          style={{ marginTop: "8px", fontSize: "0.25em" }}
        >
          Distance: {distance}
        </div>
      )}
      {similarity != null && (
        <div className="image-similarity" style={{ fontSize: "0.25em" }}>
          Similarity: {similarity.toFixed(2)}%
        </div>
      )}
    </div>
  );
};
