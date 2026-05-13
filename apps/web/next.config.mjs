/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@tractor/engine', '@tractor/protocol'],
};

export default nextConfig;
