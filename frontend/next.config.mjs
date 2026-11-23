import path from "node:path";
import { fileURLToPath } from "node:url";

const API = process.env.LH_API_URL || "http://127.0.0.1:8000";
const here = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  // 홈 폴더의 다른 package-lock.json을 프로젝트 루트로 착각하지 않게
  outputFileTracingRoot: here,
  turbopack: { root: here },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/outputs/:path*", destination: `${API}/outputs/:path*` },
    ];
  },
};

export default nextConfig;
