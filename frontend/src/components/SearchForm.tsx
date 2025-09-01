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
    require_all?: boolean;
    natural_query?: string;
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
  const [requireAll, setRequireAll] = useState(false);
  const [naturalQuery, setNaturalQuery] = useState("");
  const faceImageRef = useRef<HTMLInputElement>(null);
  const queryImageRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const face_image = faceImageRef.current?.files?.[0];
    const query_image = queryImageRef.current?.files?.[0];
    onSearch({
      face_id: faceId,
      face_image,
      query_image,
      text: text.trim() || undefined,
      require_all: requireAll,
      natural_query: naturalQuery.trim() || undefined,
    });
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Description (text):</label>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. a person smiling, outdoor scene..."
          disabled={loading}
        />
      </div>
      <div className="form-group">
        <label>Similar Image:</label>
        <input
          type="file"
          ref={queryImageRef}
          accept="image/*"
          disabled={loading}
        />
      </div>
      <div className="form-group">
        <label>Face Search:</label>
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
        />
      </div>
      <div className="form-group">
        <label>
          <input
            type="checkbox"
            checked={requireAll}
            onChange={(e) => setRequireAll(e.target.checked)}
            disabled={loading}
          />
          Require all criteria (intersection)
        </label>
      </div>
      <div className="form-group">
        <label>Natural Language Query:</label>
        <input
          type="text"
          value={naturalQuery}
          onChange={(e) => setNaturalQuery(e.target.value)}
          placeholder="e.g. has to have face A, matches this description"
          disabled={loading}
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
};
