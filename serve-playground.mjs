import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.wav': 'audio/wav',
};

export function createPlaygroundServer(rootDir = path.resolve('docs')) {
  return http.createServer(async (req, res) => {
    try {
      const urlPath = new URL(req.url || '/', 'http://localhost').pathname;
      const rel = urlPath === '/' ? '/index.html' : urlPath;
      const safePath = path.resolve(rootDir, '.' + rel);
      if (!safePath.startsWith(rootDir)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }

      const data = await fs.readFile(safePath);
      const ext = path.extname(safePath).toLowerCase();
      res.writeHead(200, {
        'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'credentialless',
        'Cross-Origin-Resource-Policy': 'same-origin',
        'Cache-Control': 'no-cache',
      });
      res.end(data);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
    }
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.PORT || 4173);
  const server = createPlaygroundServer();
  server.listen(port, '127.0.0.1', () => {
    console.log(`wasm-vad COOP/COEP playground running at http://127.0.0.1:${port}`);
  });
}
