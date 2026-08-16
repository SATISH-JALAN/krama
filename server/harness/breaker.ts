/**
 * Circuit breaker per provider (ARCHITECTURE.md §7): 5 failures in 30s ->
 * open for 60s -> half-open probe. One instance per external provider
 * (Sarvam, Groq, Cerebras) -- keeps a flaky provider from being hammered
 * with retries while it's down, and from blowing the latency budget on
 * every request while it recovers.
 */
export type BreakerState = "closed" | "open" | "half-open";

export interface BreakerOptions {
  failureThreshold: number;
  windowMs: number;
  openMs: number;
}

const DEFAULT_OPTIONS: BreakerOptions = {
  failureThreshold: 5,
  windowMs: 30_000,
  openMs: 60_000,
};

export class CircuitBreaker {
  private state: BreakerState = "closed";
  private failureTimestamps: number[] = [];
  private openedAt: number | null = null;
  private readonly opts: BreakerOptions;

  constructor(opts: Partial<BreakerOptions> = {}) {
    this.opts = { ...DEFAULT_OPTIONS, ...opts };
  }

  /** Call before attempting the guarded operation. */
  canAttempt(now: number = Date.now()): boolean {
    if (this.state === "closed") return true;
    if (this.state === "half-open") return true; // allow exactly one probe through
    // open
    if (this.openedAt !== null && now - this.openedAt >= this.opts.openMs) {
      this.state = "half-open";
      return true;
    }
    return false;
  }

  onSuccess(): void {
    this.state = "closed";
    this.failureTimestamps = [];
    this.openedAt = null;
  }

  onFailure(now: number = Date.now()): void {
    if (this.state === "half-open") {
      // probe failed -- back to open immediately, don't wait for threshold again
      this.state = "open";
      this.openedAt = now;
      return;
    }

    this.failureTimestamps.push(now);
    this.failureTimestamps = this.failureTimestamps.filter(
      (t) => now - t <= this.opts.windowMs,
    );

    if (this.failureTimestamps.length >= this.opts.failureThreshold) {
      this.state = "open";
      this.openedAt = now;
    }
  }

  getState(): BreakerState {
    return this.state;
  }
}
