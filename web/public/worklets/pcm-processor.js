// AudioWorkletProcessor: mic Float32 audio -> Int16 PCM @ 16kHz mono.
// Loaded by src/worklet/pcm.ts via `audioWorklet.addModule(...)`. Runs on the
// audio render thread, so it must stay dependency-free (no ES module imports —
// AudioWorkletGlobalScope does not reliably support them across browsers).
//
// CLAUDE.md invariant #8: Sarvam's streaming endpoint wants raw 16-bit PCM at
// 16kHz mono, not MediaRecorder's WebM/Opus. The AudioContext is *requested*
// at 16kHz (see pcm.ts), which Chrome/Firefox honour, but a processor that
// silently assumes the render rate matches would produce corrupted audio the
// moment it runs on a browser that ignores the request (older Safari) — so
// this resamples defensively whenever the actual render rate differs.

const TARGET_SAMPLE_RATE = 16000;
// ~100ms frames at 16kHz — small enough for low-latency streaming, large
// enough to not spam postMessage.
const FRAME_SIZE = 1600;

class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    /** @type {number[]} */
    this._pending = [];
    this._resampleRatio = TARGET_SAMPLE_RATE / sampleRate;
  }

  _resample(input) {
    if (this._resampleRatio === 1) return input;
    const outLength = Math.round(input.length * this._resampleRatio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcPos = i / this._resampleRatio;
      const i0 = Math.floor(srcPos);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = srcPos - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const resampled = this._resample(channel);

    // RMS on the raw (pre-resample) block — cheap level meter for the
    // waveform ring, independent of the PCM framing below.
    let sumSquares = 0;
    for (let i = 0; i < channel.length; i++) sumSquares += channel[i] * channel[i];
    const rms = Math.sqrt(sumSquares / channel.length);
    this.port.postMessage({ type: "level", rms });

    for (let i = 0; i < resampled.length; i++) this._pending.push(resampled[i]);

    while (this._pending.length >= FRAME_SIZE) {
      const frame = this._pending.splice(0, FRAME_SIZE);
      const pcm16 = new Int16Array(FRAME_SIZE);
      for (let i = 0; i < FRAME_SIZE; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage({ type: "pcm", frame: pcm16.buffer }, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PcmProcessor);
