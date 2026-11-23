const API = process.env.LH_API_URL || "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/outputs/:path*", destination: `${API}/outputs/:path*` },
    ];
  },
};

export default nextConfig;
