/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    // This will completely ignore the onnxruntime-node package
    config.externals = [...config.externals, "onnxruntime-node"];

    return config;
  },
};

export default nextConfig;
