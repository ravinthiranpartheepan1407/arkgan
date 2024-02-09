#!/bin/bash
set -e

# Generate Rust documentation
cargo doc --no-deps

# Create a new directory for Vercel
mkdir -p vercel-docs

# Copy the documentation to the new directory
cp -r target/doc/* vercel-docs/
