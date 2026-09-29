# Free AI Text-to-Speech Website

A minimal static website: **text box → voice selector → Generate button → audio player.**

## How it works
- Uses **Kokoro-82M** (`onnx-community/Kokoro-82M-v1.0-ONNX`), one of the best free,
  open-source TTS models on Hugging Face — MIT/Apache-licensed, high quality, 50+ voices.
- Runs via the **`kokoro-js`** library, which executes the model **entirely in the visitor's
  browser** (WebGPU if available, otherwise WebAssembly) using ONNX Runtime Web.
- No backend, no Hugging Face API key, no per-request cost, no rate limits from HF —
  the model file is downloaded once (~80MB) from the HF CDN and cached by the browser.
- Because there's zero server-side logic, it can be hosted on **any free static host**.

## Files
- `index.html` — page structure (textarea, voice `<select>`, Generate button, `<audio>` player)
- `style.css` — styling
- `script.js` — loads Kokoro via CDN (`kokoro-js` from jsDelivr), populates voices,
  generates audio on click, plays it and offers a WAV download

## Run locally
Just open `index.html` in a modern browser (Chrome/Edge recommended for WebGPU speed).
Some browsers block ES module imports from `file://`; if the model doesn't load, serve
the folder locally instead:

```bash
cd tts-site
python3 -m http.server 8080
# then open http://localhost:8080
```

## Deploy for free (pick any one)

### Option A — GitHub Pages (recommended, easiest)
1. Create a new GitHub repo and push these 3 files (`index.html`, `style.css`, `script.js`).
2. Go to repo **Settings → Pages**.
3. Under "Build and deployment", choose **Deploy from a branch**, branch `main`, folder `/root`.
4. Save — your site will be live at `https://<username>.github.io/<repo>/` within a minute.

### Option B — Cloudflare Pages
1. Sign up at pages.cloudflare.com (free).
2. "Create a project" → "Upload assets" (or connect a GitHub repo).
3. Upload the 3 files, deploy. You get a `*.pages.dev` URL instantly, plus a global CDN.

### Option C — Netlify
1. Sign up at netlify.com (free).
2. Drag-and-drop the `tts-site` folder into the Netlify dashboard ("Deploy manually").
3. Live instantly at a `*.netlify.app` URL.

### Option D — Vercel
1. Sign up at vercel.com (free).
2. `vercel` CLI → `vercel deploy` inside the folder, or import via GitHub.
3. Live at a `*.vercel.app` URL.

All four are free forever for a static site like this (no server, no database).

## Notes / things you can tweak
- **Voice selection**: split into 3 dropdowns — **Accent** (American / British English),
  **Voice** (Male / Female), and **Style** (specific named voice within that combo, e.g.
  Bella, Michael, Emma, George). Only Kokoro voice IDs confirmed to exist in the loaded
  model are shown.
- **Speed**: dropdown with fixed steps 1.0x–2.0x (1.0, 1.2, 1.3, ... 2.0), passed straight
  to `tts.generate(text, { voice, speed })`.
- **Android Chrome compatibility**: the script now ALWAYS uses the **single-threaded WASM**
  backend (never WebGPU), because WebGPU support on Android Chrome is inconsistent across
  devices/drivers and was observed to hang indefinitely on "Loading voice model..." with no
  error. Single-threaded WASM also avoids needing special `Cross-Origin-Opener-Policy` /
  `Cross-Origin-Embedder-Policy` headers, which a plain Netlify drag-and-drop deploy does not
  set. Audio playback is triggered inside the button tap so it counts as a user gesture and
  isn't blocked by Chrome's autoplay policy. Viewport meta tag and responsive CSS are
  included for small screens.
- **Loading feedback**: the status text now shows live download percentage, has a 90-second
  timeout, and displays a clear tappable error (with retry) instead of hanging forever if the
  model fails to load (e.g. slow/unstable mobile connection, VPN/ad-blocker interference).
- **Known bug fixed (2026-09-29)**: a prior revision imported a non-existent `env` export from
  `kokoro-js`, which crashed the module load silently with zero visible error — the page just
  sat on the static "Loading voice model, please wait..." text forever on every host (Netlify,
  GitHub Pages, etc.). Fixed by importing only the real export (`KokoroTTS`) and wrapping the
  library import itself in a try/catch with a timeout, so any future load failure now shows a
  visible, tappable-to-retry error message instead of hanging silently.
- **First load time**: the model (~80MB) downloads once and is cached by the browser;
  subsequent visits load instantly from cache. On mobile data this first load may take
  a bit — consider warning users to load on Wi-Fi.
- If you ever want a hosted/server-side alternative instead of in-browser inference,
  you could swap `script.js` to call a Hugging Face Space's Gradio API (e.g. an
  `hexgrad/Kokoro-TTS` Space) via `@gradio/client`, but that depends on the Space being
  up and has usage quotas — the current in-browser approach avoids that entirely.
