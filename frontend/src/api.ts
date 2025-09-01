import type { FaceInfo, UnifiedSearchResults } from './types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export async function fetchFaces(withThumbnails = false): Promise<FaceInfo[]> {
  const res = await fetch(`${API_BASE}/faces?with_thumbnails=${withThumbnails}`);
  if (!res.ok) throw new Error('Failed to fetch faces');
  const data = await res.json();
  return data.faces;
}

export interface SearchQuery {
  face_id?: string;
  face_image?: File;
  query_image?: File;
  text?: string;
  require_all?: boolean;
  natural_query?: string;
}

export async function unifiedSearch(query: SearchQuery): Promise<UnifiedSearchResults> {
  const form = new FormData();
  if (query.face_image) form.append('face_image', query.face_image);
  if (query.query_image) form.append('query_image', query.query_image);
  if (query.text) form.append('text', query.text);
  if (query.require_all) form.append('require_all', String(query.require_all));
  if (query.natural_query) form.append('natural_query', query.natural_query);
  // If searching by face_id, backend expects a face image upload, so fetch thumbnail and send as file
  // (Or, optionally, add a /face/{id}/image endpoint to backend)

  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Search failed');
  return await res.json();
}
