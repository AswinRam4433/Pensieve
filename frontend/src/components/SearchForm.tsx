import React, { useState, useRef } from "react";
import { FaceDropdown } from "./FaceDropdown";
import type { FaceInfo } from "../types";

interface SearchFormProps {
  faces: FaceInfo[];
  onSearch: (form: {
    face_id?: string;
    face_image?: File;
    query_image?: File;
    text?: string;
    max_results?: number;
  }) => void;
  loading: boolean;
}

export const SearchForm: React.FC<SearchFormProps> = ({
  faces,
  onSearch,
  loading,
}) => {
  const [faceId, setFaceId] = useState<string>("");
  const [text, setText] = useState("");
  const [sliderValue, setSliderValue] = useState("10");
  const faceImageRef = useRef<HTMLInputElement>(null);
  const queryImageRef = useRef<HTMLInputElement>(null);
  const numberOfResults = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const face_image = faceImageRef.current?.files?.[0];
    const query_image = queryImageRef.current?.files?.[0];
    const max_results = Number(numberOfResults.current?.value) || 10;
    onSearch({
      face_id: faceId,
      face_image,
      query_image,
      text: text.trim() || undefined,
      max_results,
    });
  };

  return (
    <form
      className="search-form"
      onSubmit={handleSubmit}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        maxWidth: 500,
        margin: "0 auto",
      }}
    >
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <label htmlFor="text" style={{ flex: 1 }}>
          Description:
        </label>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Description (e.g. smiling person, outdoor...)"
          disabled={loading}
          style={{ flex: 2, padding: "0.5rem" }}
        />
        <label htmlFor="queryImage" style={{ flex: 1 }}>
          Similar Image:
        </label>
        <input
          type="file"
          ref={queryImageRef}
          accept="image/*"
          disabled={loading}
          style={{ flex: 1 }}
          title="Similar Image"
        />
      </div>
      <label htmlFor="faceImageRef">Or search by face:</label>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <FaceDropdown
          faces={faces}
          value={faceId}
          onChange={setFaceId}
          loading={loading}
        />
        <input
          type="file"
          ref={faceImageRef}
          accept="image/*"
          disabled={loading}
          style={{ flex: 1 }}
          title="Face Image"
        />
      </div>
      <div>
        <label htmlFor="numberOfResults">Number of results to return: </label>
        <input
          ref={numberOfResults}
          type="range"
          min="5"
          max="20"
          step="1"
          value={numberOfResults.current?.value || "10"}
          onChange={() => {
            // Force re-render to show updated value
            setSliderValue(numberOfResults.current?.value ?? "10");
          }}
        />
        <span style={{ marginLeft: "1rem" }}>
          {numberOfResults.current?.value || "10"}
        </span>
      </div>
      <button
        type="submit"
        disabled={loading}
        style={{ alignSelf: "center", padding: "0.5rem 1.5rem" }}
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
};
