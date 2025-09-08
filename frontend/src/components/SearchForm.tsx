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
  const [queryImagePreview, setQueryImagePreview] = useState<string | null>(
    null
  );

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

  const handleQueryImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setQueryImagePreview(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setQueryImagePreview(null);
    }
  };

  const handleRemoveImage = () => {
    setQueryImagePreview(null);
    if (queryImageRef.current) {
      queryImageRef.current.value = "";
    }
  };

  return (
    <form className="search-form horizontal-form" onSubmit={handleSubmit}>
      <div className="form-sparkle-bg" />
      <div className="form-row">
        <div className="form-group">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Description"
            disabled={loading}
            className="input"
          />
        </div>
        <div className="form-group">
          <label htmlFor="queryImageInput" className="file-input-label">
            Upload Image
          </label>
          <input
            type="file"
            id="queryImageInput"
            ref={queryImageRef}
            accept="image/*"
            disabled={loading}
            className="input file-input hidden-file-input"
            title="Similar Image"
            onChange={handleQueryImageChange}
          />
          {queryImagePreview && (
            <div className="image-preview">
              <img
                src={queryImagePreview}
                alt="Preview"
                className="preview-image"
              />
              <button
                type="button"
                onClick={handleRemoveImage}
                className="remove-image-btn"
                title="Remove image"
              >
                ×
              </button>
            </div>
          )}
        </div>
        {/* <div className="form-group">
          <FaceDropdown
            faces={faces}
            value={faceId}
            onChange={setFaceId}
            loading={loading}
          />
        </div> */}
        {/* <div className="form-group">
          <label htmlFor="faceImageInput" className="file-input-label">
            Upload Face Image
          </label>
          <input
            type="file"
            id="faceImageInput"
            ref={faceImageRef}
            accept="image/*"
            disabled={loading}
            className="input file-input"
            title="Face Image"
          />
        </div> */}
        <div className="form-group slider-group">
          <input
            ref={numberOfResults}
            type="range"
            min="5"
            max="20"
            step="1"
            value={numberOfResults.current?.value || "10"}
            onChange={() => {
              setSliderValue(numberOfResults.current?.value ?? "10");
            }}
            className="slider"
          />
          <span className="slider-value">
            {numberOfResults.current?.value || "10"}
          </span>
        </div>
        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </form>
  );
};
