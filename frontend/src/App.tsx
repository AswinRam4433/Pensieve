import React, { useEffect, useRef, useState } from "react";
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
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchFaces(true)
      .then(setFaces)
      .catch(() => setFaces([]));
  }, []);

  // Sparkle animation effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    // Sparkle properties
    const NUM_SPARKLES = 60;
    const sparkles = Array.from({ length: NUM_SPARKLES }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 1.8 + 1.2,
      alpha: Math.random() * 0.5 + 0.5,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      twinkle: Math.random() * 0.05 + 0.01,
    }));

    type Sparkle = {
      x: number;
      y: number;
      r: number;
      alpha: number;
      dx: number;
      dy: number;
      twinkle: number;
    };

    function drawSparkle(s: Sparkle) {
      if (!ctx) return;
      ctx.save();
      ctx.globalAlpha = s.alpha;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255, 224, 102, 0.85)"; // golden
      ctx.shadowColor = "#ffe066";
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.restore();
    }

    function animate() {
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);
      for (let s of sparkles) {
        drawSparkle(s);
        s.x += s.dx;
        s.y += s.dy;
        s.alpha += (Math.random() - 0.5) * s.twinkle;
        s.alpha = Math.max(0.3, Math.min(0.9, s.alpha));
        // Wrap around edges
        if (s.x < 0) s.x = width;
        if (s.x > width) s.x = 0;
        if (s.y < 0) s.y = height;
        if (s.y > height) s.y = 0;
      }
      requestAnimationFrame(animate);
    }
    animate();

    // Resize handler
    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
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

  const HEADER_HEIGHT = 160;

  // Only show results if there are actual search results
  const hasResults = results && (results.face || results.image || results.text);

  return (
    <div className="app-container">
      {/* Sparkly animated background */}
      <canvas
        ref={canvasRef}
        className="sparkle-bg"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          zIndex: 0,
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100%",
          zIndex: 1000,
          boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
          padding: "16px 0",
          minHeight: HEADER_HEIGHT,
        }}
      >
        <h1
          style={{
            margin: "0 0 12px 0",
            textAlign: "center",
            fontFamily: "'Cinzel Decorative', cursive",
            fontSize: "2.5rem",
            letterSpacing: "2px",
            color: "#ffe066",
            textShadow: "0 2px 8px #2d2d2d, 0 0 10px #7f5af0",
          }}
        >
          Pensive
        </h1>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <SearchForm faces={faces} onSearch={handleSearch} loading={loading} />
        </div>
      </div>
      {hasResults && (
        <div
          className="main-content"
          style={{
            marginTop: HEADER_HEIGHT + 32, // Add extra space below header
            height: `calc(100vh - ${HEADER_HEIGHT + 32}px)`,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            position: "relative",
            zIndex: 1,
          }}
        >
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "16px",
            }}
          >
            <SearchResults results={results} loading={loading} error={error} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
