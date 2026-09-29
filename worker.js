// worker.js — all TTS model loading + inference happens HERE, in a dedicated Web
// Worker, off the main thread. This is what actually fixes page/button hangs:
// running WASM inference on the main thread blocks it for the FULL duration of
// each chunk's computation (not just between chunks), so no amount of
// yielding/repainting on the main thread prevents scroll/click jank while a
// chunk is being generated. Moving the model + generate() calls here means the
// main thread is 100% free the whole time - scrolling, clicking, and the
// progress bar animation all stay smooth no matter how long inference takes.
//
// Two selectable engines:
//   "kokoro" - kokoro-js / Kokoro-82M-v1.0-ONNX. Natural, expressive. Slower.
//   "piper"  - @diffusionstudio/vits-web / Piper (VITS). ~15-20x faster on CPU,
//              more robotic/synthetic voice quality. Good for bulk/long text.

const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const KOKORO_CDN_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";
const PIPER_CDN_URL = "https://cdn.jsdelivr.net/npm/@diffusionstudio/vits-web@latest/+esm";

const isMobileUA = typeof navigator !== "undefined" && /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);

let engine = null; // "kokoro" | "piper"
let kokoroTts = null;
let piperMod = null;
let backendUsed = "wasm";
const downloadedPiperVoices = new Set();

function post(msg, transfer) {
  if (transfer) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function loadKokoro() {
  const mod = await withTimeout(
    import(/* @vite-ignore */ KOKORO_CDN_URL),
    30_000,
    "Timed out loading the kokoro-js library from the CDN."
  );
  if (!mod.KokoroTTS) throw new Error("kokoro-js loaded but KokoroTTS export was not found.");
  const KokoroTTS = mod.KokoroTTS;

  const onProgress = (p) => {
    if (p && typeof p.progress === "number") post({ type: "loadProgress", pct: Math.round(p.progress) });
    else if (p && p.status) post({ type: "loadStatus", status: p.status });
  };

  if (!isMobileUA && self.navigator && self.navigator.gpu) {
    try {
      post({ type: "loadStatus", status: "Loading voice model (GPU-accelerated)..." });
      kokoroTts = await withTimeout(
        KokoroTTS.from_pretrained(KOKORO_MODEL_ID, { dtype: "fp32", device: "webgpu", progress_callback: onProgress }),
        25_000,
        "WebGPU load timed out"
      );
      backendUsed = "webgpu";
      return;
    } catch (gpuErr) {
      kokoroTts = null; // fall through to WASM
    }
  }

  post({ type: "loadStatus", status: "Loading voice model (downloading ~80MB, first visit only)..." });
  kokoroTts = await withTimeout(
    KokoroTTS.from_pretrained(KOKORO_MODEL_ID, { dtype: "q8", device: "wasm", progress_callback: onProgress }),
    90_000,
    "Model download timed out. Your connection may be too slow/unstable, or Hugging Face's servers may be temporarily unreachable."
  );
  backendUsed = "wasm";
}

async function loadPiper() {
  piperMod = await withTimeout(
    import(/* @vite-ignore */ PIPER_CDN_URL),
    30_000,
    "Timed out loading the Piper (vits-web) library from the CDN."
  );
  backendUsed = "wasm";
}

async function ensurePiperVoiceDownloaded(voiceId) {
  if (downloadedPiperVoices.has(voiceId)) return;
  post({ type: "loadStatus", status: `Downloading Piper voice "${voiceId}" (~30-60MB, first use only)...` });
  await piperMod.download(voiceId, (p) => {
    if (p && p.total) post({ type: "loadProgress", pct: Math.round((p.loaded / p.total) * 100) });
  });
  downloadedPiperVoices.add(voiceId);
  post({ type: "loadStatus", status: "Voice ready." });
}

// Minimal WAV parser - good enough for vits-web's PCM16/float32 mono output.
function parseWav(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  function str(off, len) {
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(dv.getUint8(off + i));
    return s;
  }
  if (str(0, 4) !== "RIFF" || str(8, 4) !== "WAVE") throw new Error("Unexpected audio format returned by Piper engine.");
  let offset = 12;
  let sampleRate = 22050,
    bitsPerSample = 16,
    numChannels = 1,
    dataOffset = -1,
    dataLen = 0,
    audioFormat = 1;
  while (offset < dv.byteLength - 8) {
    const chunkId = str(offset, 4);
    const chunkSize = dv.getUint32(offset + 4, true);
    if (chunkId === "fmt ") {
      audioFormat = dv.getUint16(offset + 8, true);
      numChannels = dv.getUint16(offset + 10, true);
      sampleRate = dv.getUint32(offset + 12, true);
      bitsPerSample = dv.getUint16(offset + 22, true);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataLen = chunkSize;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  if (dataOffset === -1) throw new Error("No audio data chunk found in Piper output.");
  const numSamples = Math.floor(dataLen / (bitsPerSample / 8) / numChannels);
  const out = new Float32Array(numSamples);
  if (audioFormat === 3 && bitsPerSample === 32) {
    for (let i = 0; i < numSamples; i++) out[i] = dv.getFloat32(dataOffset + i * 4, true);
  } else if (bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) out[i] = dv.getInt16(dataOffset + i * 2, true) / 32768;
  } else {
    throw new Error("Unsupported PCM bit depth from Piper: " + bitsPerSample);
  }
  return { samples: out, sampleRate };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "loadEngine") {
      engine = msg.engine;
      kokoroTts = null;
      piperMod = null;
      post({ type: "loadStatus", status: "Loading TTS engine..." });

      if (engine === "kokoro") await loadKokoro();
      else await loadPiper();

      let voices = null;
      if (engine === "kokoro" && kokoroTts) {
        if (kokoroTts.voices && typeof kokoroTts.voices === "object") voices = Object.keys(kokoroTts.voices);
        else if (typeof kokoroTts.list_voices === "function") voices = kokoroTts.list_voices();
      }
      post({ type: "engineReady", engine, backendUsed, voices });
    } else if (msg.type === "generateChunk") {
      const { chunkId, text, voiceId, speed } = msg;
      let samples, sampleRate;

      if (engine === "kokoro") {
        const audio = await kokoroTts.generate(text, { voice: voiceId, speed });
        const raw = audio.audio || audio.data;
        samples = raw instanceof Float32Array ? Float32Array.from(raw) : new Float32Array(raw);
        sampleRate = audio.sampling_rate || audio.sample_rate || 24000;
      } else {
        await ensurePiperVoiceDownloaded(voiceId);
        // Piper has no direct "speed" multiplier - lengthScale is the inverse
        // (values <1 = faster speech), applied by the model itself (no pitch shift).
        const lengthScale = speed && speed > 0 ? 1 / speed : 1;
        const blob = await piperMod.predict({ text, voiceId, options: { lengthScale } });
        const bytes = new Uint8Array(await blob.arrayBuffer());
        const parsed = parseWav(bytes);
        samples = parsed.samples;
        sampleRate = parsed.sampleRate;
      }

      // Transfer the buffer back - zero-copy, keeps the message passing cheap.
      post({ type: "chunkDone", chunkId, sampleRate, samples }, [samples.buffer]);
    }
  } catch (err) {
    post({ type: "error", context: msg.type, chunkId: msg.chunkId, message: err && err.message ? err.message : String(err) });
  }
};
