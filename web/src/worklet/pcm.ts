// Main-thread wrapper around public/worklets/pcm-processor.js.
// CLAUDE.md invariant #8: AudioWorklet -> Int16 PCM @ 16kHz mono, never
// MediaRecorder (that gives WebM/Opus, which Sarvam's streaming endpoint
// rejects).

export interface PcmCaptureCallbacks {
  onFrame: (frame: Int16Array) => void;
  onLevel: (rms: number) => void;
}

export class PcmCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;

  async start(callbacks: PcmCaptureCallbacks): Promise<void> {
    if (this.context) return; // already running

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Requesting 16000 directly avoids resampling on browsers that honour
    // it (Chrome, Firefox); pcm-processor.js resamples defensively for the
    // ones that don't (see its own comment).
    this.context = new AudioContext({ sampleRate: 16000 });
    await this.context.audioWorklet.addModule("/worklets/pcm-processor.js");

    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, "pcm-processor");
    this.node.port.onmessage = (event: MessageEvent) => {
      const msg = event.data as { type: "pcm"; frame: ArrayBuffer } | { type: "level"; rms: number };
      if (msg.type === "pcm") callbacks.onFrame(new Int16Array(msg.frame));
      else callbacks.onLevel(msg.rms);
    };

    this.source.connect(this.node);
    // Not connected to context.destination — we never want to hear our own mic back.
  }

  stop(): void {
    this.node?.port.close();
    this.node?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.context?.close();
    this.context = null;
    this.stream = null;
    this.source = null;
    this.node = null;
  }

  get isRunning(): boolean {
    return this.context !== null;
  }
}
