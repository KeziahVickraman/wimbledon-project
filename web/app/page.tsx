"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type DirEstimate = {
  direction: string;
  p_mean: number;
  p_ci80: [number, number];
};

type TendencyResponse = {
  server: string;
  side: string;
  serve_n: string;
  estimates: DirEstimate[];
};

// Fixed categorical order (never re-sorted by value) — palette.md slots 1/2/3.
const DIR_ORDER = ["wide", "body", "T"] as const;
const DIR_COLOR: Record<string, string> = {
  wide: "var(--series-1)",
  body: "var(--series-2)",
  T: "var(--series-3)",
};

export default function Dashboard() {
  const [allServers, setAllServers] = useState<string[]>([]);
  const [serverQuery, setServerQuery] = useState("");
  const [side, setSide] = useState<"deuce" | "ad">("deuce");
  const [serveN, setServeN] = useState<"first" | "second">("first");
  const [data, setData] = useState<TendencyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/servers`)
      .then((r) => r.json())
      .then((d) => setAllServers(d.servers))
      .catch(() => setError("Could not reach the API — is uvicorn running on :8000?"));
  }, []);

  const suggestions = useMemo(() => {
    const q = serverQuery.trim().toLowerCase();
    if (!q) return allServers.slice(0, 8);
    return allServers.filter((s) => s.toLowerCase().includes(q)).slice(0, 8);
  }, [serverQuery, allServers]);

  useEffect(() => {
    const server = serverQuery.trim();
    if (!allServers.includes(server)) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/tendency`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ server, side, serve_n: serveN }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? "request failed");
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [serverQuery, side, serveN, allServers]);

  const ordered = data
    ? DIR_ORDER.map((d) => data.estimates.find((e) => e.direction === d)).filter(
        (e): e is DirEstimate => !!e
      )
    : [];

  return (
    <main className="viz-root mx-auto max-w-2xl px-6 py-12">
      <style>{`
        .viz-root {
          --surface-1: #fcfcfb;
          --text-primary: #0b0b0b;
          --text-secondary: #52514e;
          --text-muted: #898781;
          --gridline: #e1e0d9;
          --baseline: #c3c2b7;
          --series-1: #2a78d6;
          --series-2: #1baf7a;
          --series-3: #eda100;
        }
        @media (prefers-color-scheme: dark) {
          .viz-root {
            --surface-1: #1a1a19;
            --text-primary: #ffffff;
            --text-secondary: #c3c2b7;
            --text-muted: #898781;
            --gridline: #2c2c2a;
            --baseline: #383835;
            --series-1: #3987e5;
            --series-2: #199e70;
            --series-3: #c98500;
          }
        }
      `}</style>

      <h1 className="text-2xl font-semibold" style={{ color: "var(--text-primary)" }}>
        Serve direction tendency
      </h1>
      <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
        Grass-court posterior — search a server, pick side and serve number.
      </p>

      <div className="mt-6 flex flex-wrap gap-3">
        <div className="relative">
          <input
            value={serverQuery}
            onChange={(e) => setServerQuery(e.target.value)}
            placeholder="Search server…"
            className="w-56 rounded border px-3 py-2 text-sm"
            style={{ borderColor: "var(--baseline)", color: "var(--text-primary)", background: "var(--surface-1)" }}
          />
          {serverQuery && !allServers.includes(serverQuery) && suggestions.length > 0 && (
            <ul
              className="absolute z-10 mt-1 w-56 rounded border shadow-sm"
              style={{ borderColor: "var(--baseline)", background: "var(--surface-1)" }}
            >
              {suggestions.map((s) => (
                <li key={s}>
                  <button
                    className="block w-full px-3 py-1.5 text-left text-sm hover:opacity-70"
                    style={{ color: "var(--text-primary)" }}
                    onClick={() => setServerQuery(s)}
                  >
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <select
          value={side}
          onChange={(e) => setSide(e.target.value as "deuce" | "ad")}
          className="rounded border px-3 py-2 text-sm"
          style={{ borderColor: "var(--baseline)", color: "var(--text-primary)", background: "var(--surface-1)" }}
        >
          <option value="deuce">Deuce side</option>
          <option value="ad">Ad side</option>
        </select>

        <select
          value={serveN}
          onChange={(e) => setServeN(e.target.value as "first" | "second")}
          className="rounded border px-3 py-2 text-sm"
          style={{ borderColor: "var(--baseline)", color: "var(--text-primary)", background: "var(--surface-1)" }}
        >
          <option value="first">1st serve</option>
          <option value="second">2nd serve</option>
        </select>
      </div>

      <div className="mt-8">
        {error && <p style={{ color: "#e34948" }}>{error}</p>}
        {!error && !data && !loading && (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Type a server name to see direction probabilities.
          </p>
        )}
        {loading && (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Loading…
          </p>
        )}
        {data && ordered.length > 0 && (
          <>
            <BarChart estimates={ordered} hovered={hovered} onHover={setHovered} />
            <table className="mt-6 w-full text-sm" style={{ color: "var(--text-secondary)" }}>
              <caption className="sr-only">
                Direction probabilities for {data.server}, {data.side} side, {data.serve_n} serve
              </caption>
              <thead>
                <tr style={{ color: "var(--text-muted)" }}>
                  <th className="text-left font-normal">Direction</th>
                  <th className="text-right font-normal">Mean</th>
                  <th className="text-right font-normal">80% CI</th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((e) => (
                  <tr key={e.direction} style={{ borderTop: "1px solid var(--gridline)" }}>
                    <td className="py-1">{e.direction}</td>
                    <td className="py-1 text-right tabular-nums">{(e.p_mean * 100).toFixed(1)}%</td>
                    <td className="py-1 text-right tabular-nums">
                      {(e.p_ci80[0] * 100).toFixed(1)}–{(e.p_ci80[1] * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </main>
  );
}

function BarChart({
  estimates,
  hovered,
  onHover,
}: {
  estimates: DirEstimate[];
  hovered: string | null;
  onHover: (d: string | null) => void;
}) {
  const width = 480;
  const height = 260;
  const padLeft = 36;
  const padBottom = 28;
  const padTop = 16;
  const plotW = width - padLeft - 16;
  const plotH = height - padTop - padBottom;
  const bandW = plotW / estimates.length;
  const barW = 24;
  const yMax = 1;
  const y = (v: number) => padTop + plotH * (1 - v / yMax);

  return (
    <div className="relative">
      <svg width={width} height={height} role="img" aria-label="Direction probability bar chart">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={padLeft}
              x2={width - 16}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
            <text x={padLeft - 8} y={y(t) + 4} fontSize={11} textAnchor="end" fill="var(--text-muted)">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        <line
          x1={padLeft}
          x2={width - 16}
          y1={y(0)}
          y2={y(0)}
          stroke="var(--baseline)"
          strokeWidth={1}
        />

        {estimates.map((e, i) => {
          const cx = padLeft + bandW * i + bandW / 2;
          const barX = cx - barW / 2;
          const topY = y(e.p_mean);
          const isHovered = hovered === e.direction;
          const [lo, hi] = e.p_ci80;
          return (
            <g
              key={e.direction}
              onMouseEnter={() => onHover(e.direction)}
              onMouseLeave={() => onHover(null)}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={barX}
                y={topY}
                width={barW}
                height={Math.max(y(0) - topY, 0)}
                rx={4}
                fill={DIR_COLOR[e.direction]}
                opacity={isHovered ? 1 : 0.92}
              />
              {/* 80% credible interval whisker */}
              <line x1={cx} x2={cx} y1={y(hi)} y2={y(lo)} stroke="var(--text-secondary)" strokeWidth={2} />
              <line x1={cx - 5} x2={cx + 5} y1={y(hi)} y2={y(hi)} stroke="var(--text-secondary)" strokeWidth={2} />
              <line x1={cx - 5} x2={cx + 5} y1={y(lo)} y2={y(lo)} stroke="var(--text-secondary)" strokeWidth={2} />
              {/* direct label at the tip — required relief for light-mode aqua/yellow contrast */}
              <text
                x={cx}
                y={y(hi) - 8}
                fontSize={12}
                fontWeight={600}
                textAnchor="middle"
                fill="var(--text-primary)"
              >
                {(e.p_mean * 100).toFixed(0)}%
              </text>
              <text
                x={cx}
                y={height - padBottom + 18}
                fontSize={12}
                textAnchor="middle"
                fill="var(--text-secondary)"
              >
                {e.direction}
              </text>
            </g>
          );
        })}
      </svg>
      {hovered && (
        <div
          className="pointer-events-none absolute left-2 top-2 rounded border px-2 py-1 text-xs shadow-sm"
          style={{ borderColor: "var(--baseline)", background: "var(--surface-1)", color: "var(--text-primary)" }}
        >
          {(() => {
            const e = estimates.find((x) => x.direction === hovered)!;
            return (
              <>
                <strong>{e.direction}</strong>: {(e.p_mean * 100).toFixed(1)}% (80% CI{" "}
                {(e.p_ci80[0] * 100).toFixed(1)}–{(e.p_ci80[1] * 100).toFixed(1)}%)
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
