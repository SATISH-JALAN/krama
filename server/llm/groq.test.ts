import { describe, expect, test } from "bun:test";
import { createGroqProvider, DEFAULT_GROQ_MODEL } from "./groq";

describe("createGroqProvider", () => {
  test("returns the completion text on success", async () => {
    const provider = createGroqProvider("fake-key", {
      client: {
        chat: {
          completions: {
            create: (async () => ({
              choices: [{ message: { content: "the answer" } }],
            })) as any,
          },
        } as any,
      },
    });
    expect(await provider.generate("a prompt")).toBe("the answer");
    expect(provider.name).toBe("groq");
  });

  test("sends the expected model and message shape", async () => {
    let capturedArgs: any = null;
    const provider = createGroqProvider("fake-key", {
      model: "custom-model",
      client: {
        chat: {
          completions: {
            create: (async (args: any) => {
              capturedArgs = args;
              return { choices: [{ message: { content: "ok" } }] };
            }) as any,
          },
        } as any,
      },
    });
    await provider.generate("what is a corporation?");
    expect(capturedArgs.model).toBe("custom-model");
    expect(capturedArgs.messages).toEqual([{ role: "user", content: "what is a corporation?" }]);
  });

  test("defaults to DEFAULT_GROQ_MODEL when no model given", async () => {
    let capturedModel: string | null = null;
    const provider = createGroqProvider("fake-key", {
      client: {
        chat: {
          completions: {
            create: (async (args: any) => {
              capturedModel = args.model;
              return { choices: [{ message: { content: "ok" } }] };
            }) as any,
          },
        } as any,
      },
    });
    await provider.generate("x");
    expect(capturedModel).toBe(DEFAULT_GROQ_MODEL);
  });

  test("throws on an empty completion instead of returning an empty string silently", async () => {
    const provider = createGroqProvider("fake-key", {
      client: {
        chat: { completions: { create: (async () => ({ choices: [{ message: {} }] })) as any } } as any,
      },
    });
    await expect(provider.generate("x")).rejects.toThrow(/empty completion/);
  });
});
