// Free, open-source, 100% client-side Text-to-Speech
// Model: onnx-community/Kokoro-82M-v1.0-ONNX (Kokoro-82M, Apache-2.0, hosted on Hugging Face)
// Library: kokoro-js (https://www.npmjs.com/package/kokoro-js) - runs the model fully in the browser
// No API key, no server, no per-request cost.
//
// IMPORTANT (mobile compatibility): WebGPU support on Android Chrome is inconsistent
// across devices/drivers and can silently hang instead of failing cleanly. To guarantee
// this works on Android Chrome (and every other browser), we ALWAYS use the WASM backend
// with single-threaded execution (no SharedArrayBuffer / cross-origin-isolation headers
// required - those aren't set by a plain Netlify static drop-deploy anyway).

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const KOKORO_CDN_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";
const LOAD_TIMEOUT_MS = 90_000; // 90s - mobile data can be slow for an ~80MB download

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

let tts = null;
let availableVoiceIds = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function withTimeout(promise, ms, timeoutMessage) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(timeoutMessage)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function init() {
  try {
    setStatus("Loading TTS engine (kokoro-js)...");

    // Dynamic import wrapped in try/catch so a failure here (bad CDN response,
    // network block, syntax/version mismatch, etc.) shows a visible error on the
    // page instead of silently freezing the whole script with no feedback.
    let KokoroTTS;
    try {
      const mod = await withTimeout(
        import(/* @vite-ignore */ KOKORO_CDN_URL),
        30_000,
        "Timed out loading the kokoro-js library itself from the CDN."
      );
      KokoroTTS = mod.KokoroTTS;
      if (!KokoroTTS) throw new Error("kokoro-js loaded but KokoroTTS export was not found.");
    } catch (importErr) {
      throw new Error("Could not load the kokoro-js library: " + importErr.message);
    }

    setStatus("Loading voice model... 0% (downloading ~80MB, first visit only)");

    const loadPromise = KokoroTTS.from_pretrained(MODEL_ID, {
      dtype: "q8",
      device: "wasm",
      progress_callback: (progress) => {
        if (progress && typeof progress.progress === "number") {
          const pct = Math.round(progress.progress);
          setStatus(`Loading voice model... ${pct}% (downloading ~80MB, first visit only)`);
        } else if (progress && progress.status) {
          setStatus(`Loading voice model... (${progress.status})`);
        }
      },
    });

    tts = await withTimeout(
      loadPromise,
      LOAD_TIMEOUT_MS,
      "Model download timed out. Your connection may be too slow/unstable, or Hugging Face's " +
        "servers may be temporarily unreachable from your network. Try switching to Wi-Fi, " +
        "disabling any VPN/ad-blocker, and reloading the page."
    );

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
    setStatus("Model loaded. Ready to generate speech!");
  } catch (err) {
    console.error("KokoroTTS load failed:", err);
    setStatus(
      "⚠ Failed to load the voice model: " +
        (err && err.message ? err.message : "unknown error") +
        " — Tap to retry."
    );
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

async function audioToBlob(audio) {
  if (typeof audio.toBlob === "function") {
    return await audio.toBlob();
  }
  const samples = audio.audio || audio.data;
  const sampleRate = audio.sampling_rate || audio.sample_rate || 24000;
  return encodeWav(samples, sampleRate);
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

  const voiceId = nameSelect.value;
  const speed = parseFloat(speedSelect.value) || 1.0;

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  setStatus("Generating speech...");

  try {
    const audio = await tts.generate(text, { voice: voiceId, speed });
    const blob = await audioToBlob(audio);
    const url = URL.createObjectURL(blob);

    player.src = url;
    player.play().catch(() => {
      // Autoplay may be blocked; user can press play on the visible controls.
    });

    downloadLink.href = url;
    downloadLink.classList.remove("hidden");

    setStatus("Done! Playing audio.");
  } catch (err) {
    console.error("Generate failed:", err);
    setStatus("⚠ Error generating audio: " + (err && err.message ? err.message : "unknown error"));
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Speech";
  }
}

generateBtn.addEventListener("click", generate);

init();
