// Strongly typed API types for the image search frontend

export interface FaceInfo {
  face_id: string;
  photo_name: string;
  face_index: number;
  face_location: [number, number, number, number];
  thumbnail?: string;
}

export interface SearchResultFace {
  query_face_index: number;
  query_face_location: [number, number, number, number];
  similar_faces: Array<{
    face_id: string;
    photo_path: string;
    photo_name: string;
    face_index: number;
    distance: number;
    similarity: number;
    face_location: [number, number, number, number];
    thumbnail?: string;
  }>;
}

export interface SearchResultText {
  image_path: string;
  relative_path: string;
  similarity: number;
  score_percentage: number;
  query: string;
  search_type: string;
}

export interface SearchResultImage {
  distance: number;
  index: number;
  metadata: {
    file_path?: string;
    url?: string;
    [key: string]: any;
  };
}

export interface UnifiedSearchResults {
  face?: SearchResultFace[];
  text?: SearchResultText[];
  image?: SearchResultImage[];
  intersection?: string[];
}
