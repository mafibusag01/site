// Free, open-source, 100% client-side Text-to-Speech
// Two selectable engines, both run fully in-browser via a Web Worker (see worker.js):
//   - "kokoro": onnx-community/Kokoro-82M-v1.0-ONNX (kokoro-js). Natural, expressive
//     voice, but slower (~1-2x realtime on CPU). Good default for shorter texts.
//   - "piper": @diffusionstudio/vits-web (Piper/VITS). ~15-20x realtime on CPU -
//     roughly 5,000 words in about a minute - but a more robotic/synthetic voice.
//     Best for bulk/long-text jobs where speed matters more than natural delivery.
// No API key, no server, no per-request cost.
//
// MAIN-THREAD RESPONSIVENESS: all model loading + inference now happens inside a
// dedicated Web Worker (worker.js), not on the main thread. Previously, WASM
// inference blocked the main thread for the FULL duration of each chunk's
// computation - not just between chunks - so the page (including scrolling and
// the Generate button) could still hang mid-chunk even with inter-chunk yields.
// Running the model in a worker keeps the main thread 100% free for scrolling,
// clicks, and smooth progress-bar animation regardless of how long a chunk takes.
//
// LONG TEXT HANDLING: text is split into sentence-sized chunks and generated one
// at a time via the worker. Audio for each chunk starts playing via the Web Audio
// API as soon as it's ready, and the status bar shows live % progress + an ETA.
// All chunks are stitched into one combined WAV file for download once complete.

const MAX_CHUNK_CHARS = 350; // sentence-aware chunk size fed to the model per call

// Kokoro-82M voice IDs grouped by accent + gender.
const KOKORO_VOICES = {
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

// Piper voice IDs (Rhasspy Piper Voices). Fewer style variations per voice - one
// model = one speaker - but dramatically faster inference.
const PIPER_VOICES = {
  american: {
    female: [
      { id: "en_US-amy-medium", label: "Amy (default)" },
      { id: "en_US-lessac-medium", label: "Lessac" },
      { id: "en_US-hfc_female-medium", label: "HFC Female" },
      { id: "en_US-kristin-medium", label: "Kristin" },
    ],
    male: [
      { id: "en_US-ryan-medium", label: "Ryan (default)" },
      { id: "en_US-joe-medium", label: "Joe" },
      { id: "en_US-john-medium", label: "John" },
      { id: "en_US-hfc_male-medium", label: "HFC Male" },
    ],
  },
  british: {
    female: [
      { id: "en_GB-jenny_dioco-medium", label: "Jenny (default)" },
      { id: "en_GB-alba-medium", label: "Alba" },
      { id: "en_GB-southern_english_female-low", label: "Southern English Female" },
    ],
    male: [
      { id: "en_GB-alan-medium", label: "Alan (default)" },
      { id: "en_GB-northern_english_male-medium", label: "Northern English Male" },
    ],
  },
};

function currentVoiceCatalog() {
  return engineSelect.value === "piper" ? PIPER_VOICES : KOKORO_VOICES;
}

const textInput = document.getElementById("textInput");
const engineSelect = document.getElementById("engineSelect");
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
const engineNote = document.getElementById("engineNote");

let worker = null;
let availableVoiceIds = null; // Set, from engine "engineReady" message (kokoro only)
let currentEngine = null; // engine actually loaded in the worker right now
let currentGenerationId = 0; // guards against overlapping generate() calls
let activeAudioCtx = null;
let pendingChunkResolvers = new Map(); // chunkId -> {resolve, reject}
let chunkIdCounter = 0;

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

function formatEta(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `~${m}m ${s}s left` : `~${s}s left`;
}

function createWorker() {
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "loadProgress") {
      setStatus(`Loading voice model... ${msg.pct}% (first visit only)`);
    } else if (msg.type === "loadStatus") {
      setStatus(msg.status);
    } else if (msg.type === "engineReady") {
      currentEngine = msg.engine;
      availableVoiceIds = msg.voices ? new Set(msg.voices) : null;
      populateNameOptions();
      setControlsEnabled(true);
      setStatus(
        `${msg.engine === "piper" ? "Piper (fast)" : "Kokoro (quality)"} model loaded` +
          (msg.backendUsed === "webgpu" ? " (GPU-accelerated)" : "") +
          ". Ready to generate speech!"
      );
    } else if (msg.type === "chunkDone") {
      const resolver = pendingChunkResolvers.get(msg.chunkId);
      if (resolver) {
        pendingChunkResolvers.delete(msg.chunkId);
        resolver.resolve({ samples: msg.samples, sampleRate: msg.sampleRate });
      }
    } else if (msg.type === "error") {
      if (msg.chunkId != null && pendingChunkResolvers.has(msg.chunkId)) {
        const resolver = pendingChunkResolvers.get(msg.chunkId);
        pendingChunkResolvers.delete(msg.chunkId);
        resolver.reject(new Error(msg.message));
      } else {
        onEngineLoadError(msg.message);
      }
    }
  };
  worker.onerror = (e) => {
    console.error("Worker error:", e);
    onEngineLoadError(e.message || "Unknown worker error");
  };
}

function setControlsEnabled(enabled) {
  engineSelect.disabled = !enabled;
  accentSelect.disabled = !enabled;
  genderSelect.disabled = !enabled;
  nameSelect.disabled = !enabled;
  generateBtn.disabled = !enabled;
}

function onEngineLoadError(rawMsg) {
  const friendlyMsg =
    rawMsg === "Failed to fetch"
      ? "Could not reach the model server. This is usually caused by an ad-blocker/privacy " +
        "extension, VPN, or firewall. Try an Incognito window with extensions disabled, or a different network."
      : rawMsg;
  setStatus("⚠ Failed to load the voice model: " + friendlyMsg + " — Tap to retry.");
  statusEl.style.cursor = "pointer";
  statusEl.onclick = () => {
    statusEl.style.cursor = "default";
    statusEl.onclick = null;
    loadEngine(engineSelect.value);
  };
}

function loadEngine(engine) {
  setControlsEnabled(false);
  nameSelect.innerHTML = "<option>Loading...</option>";
  setStatus(`Loading ${engine === "piper" ? "Piper (fast)" : "Kokoro (quality)"} engine...`);
  setProgress(0);
  progressWrap.classList.remove("hidden");
  if (!worker) createWorker();
  worker.postMessage({ type: "loadEngine", engine });
}

engineSelect.addEventListener("change", () => {
  hideProgress();
  engineNote.textContent =
    engineSelect.value === "piper"
      ? "⚡ Fast mode: ~15-20x realtime, great for long/bulk text. Voice sounds more synthetic."
      : "🎙 Quality mode: natural, expressive voice. Slower on long text.";
  loadEngine(engineSelect.value);
});

function populateNameOptions() {
  const accent = accentSelect.value;
  const gender = genderSelect.value;
  let options = currentVoiceCatalog()[accent][gender];

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

// Sends one chunk to the worker and resolves when that chunk's audio comes back.
// Because inference now runs entirely inside the worker, this await does NOT
// block the main thread - the browser stays fully interactive (scrolling, clicks,
// progress bar animation) while the worker crunches the chunk in the background.
function generateChunkInWorker(text, voiceId, speed) {
  const chunkId = ++chunkIdCounter;
  return new Promise((resolve, reject) => {
    pendingChunkResolvers.set(chunkId, { resolve, reject });
    worker.postMessage({ type: "generateChunk", chunkId, text, voiceId, speed });
  });
}

async function generate() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus("Please enter some text first.");
    return;
  }
  if (!currentEngine) {
    setStatus("Model is not ready yet.");
    return;
  }

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

      // This await yields control back to the browser's event loop immediately -
      // the actual computation happens in the worker thread, not here - so the
      // page remains scrollable/clickable the entire time this is pending.
      const { samples, sampleRate } = await generateChunkInWorker(chunks[i], voiceId, speed);
      if (myGenerationId !== currentGenerationId) return;

      sampleRateGlobal = sampleRate;
      allChunkSamples.push(samples);

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
    }

    if (myGenerationId !== currentGenerationId) return;

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

// Boot: load the default engine (Kokoro) in the worker.
loadEngine(engineSelect.value);
