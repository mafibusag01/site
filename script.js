// Free, open-source, 100% client-side Text-to-Speech
// Model: onnx-community/Kokoro-82M-v1.0-ONNX (Kokoro-82M, Apache-2.0, hosted on Hugging Face)
// Library: kokoro-js (https://www.npmjs.com/package/kokoro-js) - runs the model fully in the browser
// No API key, no server, no per-request cost.
//
// MOBILE COMPATIBILITY: WebGPU support on Android Chrome is inconsistent across
// devices/drivers and can silently hang instead of failing cleanly (confirmed on real
// devices). So on mobile we ALWAYS use the WASM backend (proven reliable). On desktop
// browsers we try WebGPU first (much faster for long texts) and automatically fall
// back to WASM if it fails or times out.
//
// LONG TEXT HANDLING: text is split into sentence-sized chunks and generated one at a
// time. Audio for each chunk starts playing via the Web Audio API as soon as it's
// ready (instead of waiting for the whole 10k-word input to finish), and the status
// bar shows live % progress + an ETA. All chunks are also stitched into one combined
// WAV file for download once generation completes.

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const KOKORO_CDN_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";
const LOAD_TIMEOUT_MS = 90_000; // 90s - mobile data can be slow for the model download
const WEBGPU_LOAD_TIMEOUT_MS = 25_000; // shorter probe - if WebGPU stalls, bail to WASM fast
const MAX_CHUNK_CHARS = 350; // sentence-aware chunk size fed to the model per call

const isMobile = /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);

// Known Kokoro-82M voice IDs grouped by accent + gender, with friendly labels.
const VOICE_CATALOG = {
  american: {
    female: [
      { id: "af_heart", label: "Heart (default)" },
      { id: "af_bella", label: "Bella" },
      { id: "af_nova", label: "Nova" },
      { id: "af_sky", label: "Sky" },
      { id: "af_sarah", label: "Sarah" },
      { id: "af_nicole", label: "Nicole" },
      { id: "af_alloy", label: "Alloy" },
      { id: "af_aoede", label: "Aoede" },
      { id: "af_jessica", label: "Jessica" },
      { id: "af_kore", label: "Kore" },
      { id: "af_river", label: "River" },
    ],
    male: [
      { id: "am_michael", label: "Michael (default)" },
      { id: "am_adam", label: "Adam" },
      { id: "am_echo", label: "Echo" },
      { id: "am_eric", label: "Eric" },
      { id: "am_fenrir", label: "Fenrir" },
      { id: "am_liam", label: "Liam" },
      { id: "am_onyx", label: "Onyx" },
      { id: "am_puck", label: "Puck" },
      { id: "am_santa", label: "Santa" },
    ],
  },
  british: {
    female: [
      { id: "bf_emma", label: "Emma (default)" },
      { id: "bf_alice", label: "Alice" },
      { id: "bf_isabella", label: "Isabella" },
      { id: "bf_lily", label: "Lily" },
    ],
    male: [
      { id: "bm_george", label: "George (default)" },
      { id: "bm_daniel", label: "Daniel" },
      { id: "bm_fable", label: "Fable" },
      { id: "bm_lewis", label: "Lewis" },
    ],
  },
};

const textInput = document.getElementById("textInput");
const accentSelect = document.getElementById("accentSelect");
const genderSelect = document.getElementById("genderSelect");
const nameSelect = document.getElementById("nameSelect");
const speedSelect = document.getElementById("speedSelect");
const generateBtn = document.getElementById("generateBtn");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
const downloadLink = document.getElementById("downloadLink");
const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const wordCountEl = document.getElementById("wordCount");
const clearBtn = document.getElementById("clearBtn");

let tts = null;
let availableVoiceIds = null;
let currentGenerationId = 0; // guards against overlapping generate() calls
let activeAudioCtx = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setProgress(pct) {
  progressWrap.classList.remove("hidden");
  progressFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
}

function hideProgress() {
  progressWrap.classList.add("hidden");
  progressFill.style.width = "0%";
}

// WASM inference blocks the main JS thread while it computes. Without yielding,
// the browser queues up all our status/progress DOM updates but never actually
// paints them until the whole synchronous work is done - making the UI *look*
// frozen even though it's working (audio keeps playing underneath). Awaiting this
// after each chunk forces a real repaint so the progress bar/status update live.
function yieldToBrowser() {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

function withTimeout(promise, ms, timeoutMessage) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(timeoutMessage)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function formatEta(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `~${m}m ${s}s left` : `~${s}s left`;
}

async function loadKokoroLibrary() {
  try {
    const mod = await withTimeout(
      import(/* @vite-ignore */ KOKORO_CDN_URL),
      30_000,
      "Timed out loading the kokoro-js library itself from the CDN."
    );
    if (!mod.KokoroTTS) throw new Error("kokoro-js loaded but KokoroTTS export was not found.");
    return mod.KokoroTTS;
  } catch (importErr) {
    throw new Error("Could not load the kokoro-js library: " + importErr.message);
  }
}

async function init() {
  try {
    setStatus("Loading TTS engine (kokoro-js)...");
    const KokoroTTS = await loadKokoroLibrary();

    const onProgress = (progress) => {
      if (progress && typeof progress.progress === "number") {
        const pct = Math.round(progress.progress);
        setStatus(`Loading voice model... ${pct}% (downloading, first visit only)`);
      } else if (progress && progress.status) {
        setStatus(`Loading voice model... (${progress.status})`);
      }
    };

    // Desktop: try WebGPU first (much faster inference for long texts), with a
    // fast fallback to WASM if it stalls or errors. Mobile: skip straight to WASM,
    // since WebGPU on Android Chrome has been observed to hang silently.
    let backendUsed = "wasm";
    if (!isMobile && navigator.gpu) {
      try {
        setStatus("Loading voice model (GPU-accelerated)... 0%");
        tts = await withTimeout(
          KokoroTTS.from_pretrained(MODEL_ID, {
            dtype: "fp32",
            device: "webgpu",
            progress_callback: onProgress,
          }),
          WEBGPU_LOAD_TIMEOUT_MS,
          "WebGPU load timed out"
        );
        backendUsed = "webgpu";
      } catch (gpuErr) {
        console.warn("WebGPU load failed, falling back to WASM:", gpuErr);
        tts = null;
      }
    }

    if (!tts) {
      setStatus("Loading voice model... 0% (downloading ~80MB, first visit only)");
      const loadPromise = KokoroTTS.from_pretrained(MODEL_ID, {
        dtype: "q8",
        device: "wasm",
        progress_callback: onProgress,
      });
      tts = await withTimeout(
        loadPromise,
        LOAD_TIMEOUT_MS,
        "Model download timed out. Your connection may be too slow/unstable, or Hugging Face's " +
          "servers may be temporarily unreachable from your network. Try switching to Wi-Fi, " +
          "disabling any VPN/ad-blocker, and reloading the page."
      );
    }

    if (tts.voices && typeof tts.voices === "object") {
      availableVoiceIds = new Set(Object.keys(tts.voices));
    } else if (typeof tts.list_voices === "function") {
      availableVoiceIds = new Set(tts.list_voices());
    }

    populateNameOptions();

    accentSelect.disabled = false;
    genderSelect.disabled = false;
    nameSelect.disabled = false;
    generateBtn.disabled = false;
    setStatus(
      `Model loaded (${backendUsed === "webgpu" ? "GPU-accelerated" : "CPU"} mode). Ready to generate speech!`
    );
  } catch (err) {
    console.error("KokoroTTS load failed:", err);
    const rawMsg = err && err.message ? err.message : "unknown error";
    const friendlyMsg =
      rawMsg === "Failed to fetch"
        ? "Could not reach the model server (huggingface.co). This is usually caused by an " +
          "ad-blocker/privacy extension, VPN, or firewall blocking that domain. Try an " +
          "Incognito window with extensions disabled, or a different network."
        : rawMsg;
    setStatus("⚠ Failed to load the voice model: " + friendlyMsg + " — Tap to retry.");
    statusEl.style.cursor = "pointer";
    statusEl.onclick = () => {
      statusEl.style.cursor = "default";
      statusEl.onclick = null;
      init();
    };
  }
}

function populateNameOptions() {
  const accent = accentSelect.value;
  const gender = genderSelect.value;
  let options = VOICE_CATALOG[accent][gender];

  if (availableVoiceIds) {
    const filtered = options.filter((o) => availableVoiceIds.has(o.id));
    if (filtered.length > 0) options = filtered;
  }

  nameSelect.innerHTML = "";
  for (const { id, label } of options) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    nameSelect.appendChild(opt);
  }
}

accentSelect.addEventListener("change", populateNameOptions);
genderSelect.addEventListener("change", populateNameOptions);

function updateWordCount() {
  const text = textInput.value;
  const trimmed = text.trim();
  const words = trimmed.length > 0 ? trimmed.split(/\s+/).length : 0;
  const chars = text.length;
  wordCountEl.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"} · ${chars.toLocaleString()} character${chars === 1 ? "" : "s"}`;
}

textInput.addEventListener("input", updateWordCount);
updateWordCount(); // reflect the pre-filled placeholder text on load

clearBtn.addEventListener("click", () => {
  textInput.value = "";
  updateWordCount();
  textInput.focus();
});

// Splits long text into sentence-aware chunks so: (a) the model handles each request
// reliably, (b) we can report incremental % progress, and (c) audio can start playing
// before the entire input has been processed.
function splitTextIntoChunks(text, maxLen = MAX_CHUNK_CHARS) {
  const sentences = text.match(/[^.!?\n]+[.!?\n]*(\s+|$)/g) || [text];
  const chunks = [];
  let current = "";

  for (let sentence of sentences) {
    if (sentence.length > maxLen) {
      // Extremely long "sentence" (no punctuation) - hard split on spaces.
      const words = sentence.split(/(\s+)/);
      for (const w of words) {
        if ((current + w).length > maxLen) {
          if (current.trim()) chunks.push(current.trim());
          current = w;
        } else {
          current += w;
        }
      }
      continue;
    }
    if ((current + sentence).length > maxLen) {
      if (current.trim()) chunks.push(current.trim());
      current = sentence;
    } else {
      current += sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks.filter((c) => c.length > 0);
}

function concatFloat32Arrays(arrays) {
  const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

function extractSamples(audio) {
  const samples = audio.audio || audio.data;
  const sampleRate = audio.sampling_rate || audio.sample_rate || 24000;
  return { samples: samples instanceof Float32Array ? samples : new Float32Array(samples), sampleRate };
}

async function generate() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus("Please enter some text first.");
    return;
  }
  if (!tts) {
    setStatus("Model is not ready yet.");
    return;
  }

  // Invalidate any previous in-flight generation and stop its audio.
  const myGenerationId = ++currentGenerationId;
  if (activeAudioCtx) {
    try {
      activeAudioCtx.close();
    } catch (e) {
      /* ignore */
    }
    activeAudioCtx = null;
  }

  const voiceId = nameSelect.value;
  const speed = parseFloat(speedSelect.value) || 1.0;

  const chunks = splitTextIntoChunks(text);
  const totalChars = chunks.reduce((sum, c) => sum + c.length, 0);

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating... 0%";
  downloadLink.classList.add("hidden");
  player.classList.add("hidden");
  setProgress(0);

  // Create the AudioContext synchronously (within this user-gesture click handler)
  // so mobile/desktop autoplay policies allow scheduled playback without an extra tap.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  activeAudioCtx = audioCtx;
  let nextStartTime = audioCtx.currentTime + 0.05;

  const allChunkSamples = [];
  let sampleRateGlobal = 24000;
  let processedChars = 0;
  const startTime = performance.now();

  try {
    for (let i = 0; i < chunks.length; i++) {
      if (myGenerationId !== currentGenerationId) return; // superseded by a newer click

      const audio = await tts.generate(chunks[i], { voice: voiceId, speed });
      if (myGenerationId !== currentGenerationId) return;

      const { samples, sampleRate } = extractSamples(audio);
      sampleRateGlobal = sampleRate;
      allChunkSamples.push(samples);

      // Schedule this chunk to play immediately after the previous one - the user
      // starts hearing audio after just the FIRST chunk, not after all of them.
      const audioBuffer = audioCtx.createBuffer(1, samples.length, sampleRate);
      audioBuffer.copyToChannel(samples, 0);
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      const startAt = Math.max(nextStartTime, audioCtx.currentTime);
      source.start(startAt);
      nextStartTime = startAt + audioBuffer.duration;

      processedChars += chunks[i].length;
      const pct = Math.round((processedChars / totalChars) * 100);
      const elapsedSec = (performance.now() - startTime) / 1000;
      const remainingChars = totalChars - processedChars;
      const etaSec = remainingChars > 0 ? (elapsedSec / processedChars) * remainingChars : 0;
      setProgress(pct);
      generateBtn.textContent = `Generating... ${pct}%`;
      setStatus(
        `Generating speech... ${pct}% (chunk ${i + 1}/${chunks.length})` +
          (chunks.length > 1 ? ` · ${formatEta(etaSec)} · playing as it's ready` : "")
      );

      // Force the browser to actually paint the update above before the next
      // (blocking) chunk of WASM inference starts - otherwise the UI appears
      // frozen even though audio is playing and work is progressing.
      await yieldToBrowser();
      if (myGenerationId !== currentGenerationId) return;
    }

    if (myGenerationId !== currentGenerationId) return;

    // Stitch every chunk together into one downloadable WAV file.
    const combined = concatFloat32Arrays(allChunkSamples);
    const blob = encodeWav(combined, sampleRateGlobal);
    const url = URL.createObjectURL(blob);
    player.src = url;
    player.classList.remove("hidden");
    downloadLink.href = url;
    downloadLink.classList.remove("hidden");

    setProgress(100);
    setStatus(`Done! ${chunks.length > 1 ? "Full audio ready below." : "Playing audio."}`);
  } catch (err) {
    console.error("Generate failed:", err);
    setStatus("⚠ Error generating audio: " + (err && err.message ? err.message : "unknown error"));
  } finally {
    if (myGenerationId === currentGenerationId) {
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate Speech";
      setTimeout(() => {
        if (myGenerationId === currentGenerationId) hideProgress();
      }, 1500);
    }
  }
}

generateBtn.addEventListener("click", generate);

init();
