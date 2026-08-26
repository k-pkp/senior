/** @type {import('next').NextConfig} */
const nextConfig = {
  // The samples path is fully static — no backend needed to browse precomputed
  // runs. Uploads talk to a separate FastAPI service when it is reachable.
  transpilePackages: ["three"],
};
export default nextConfig;
