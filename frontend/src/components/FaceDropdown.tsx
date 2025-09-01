import React from "react";
import type { FaceInfo } from "../types";

interface FaceDropdownProps {
  faces: FaceInfo[];
  value?: string;
  onChange: (faceId: string) => void;
  loading?: boolean;
}

export const FaceDropdown: React.FC<FaceDropdownProps> = ({
  faces,
  value,
  onChange,
  loading,
}) => (
  <select
    className="face-dropdown"
    value={value || ""}
    onChange={(e) => onChange(e.target.value)}
    disabled={loading}
  >
    <option value="">Select a face (optional)</option>
    {faces.map((face) => (
      <option key={face.face_id} value={face.face_id}>
        {face.photo_name} (Face #{face.face_index})
      </option>
    ))}
  </select>
);
