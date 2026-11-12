# AGENTS.md

A character-level GPT trained in the browser on a list of first names. Its purpose is **learning**: the owner reads the code to understand how an LLM works. Clear code matters more than speed or features.

## Files

- `tiny-llm.ts`: the whole model in one file, with no dependencies. It runs top to bottom through data, tokenizer, autograd (`V`), the transformer (`gpt`), Adam (`trainStep`) and inference (`sample`, `nextTokenProbs`). Importing it has no side effects; nothing trains until `trainStep()` is called.
- `ui.ts`: connects the model to the DOM. It must not contain model logic.
- `index.html`: the layout and styles, using CSS variables for light and dark mode.

## Commands

- `npm run dev`: starts Vite and opens the UI. Opening `index.html` via `file://` does not work.
- `npm run typecheck`: runs `tsc --noEmit`. Vite does not type-check, so run this after every change.
- `node tiny-llm.ts`: imports the model in Node (22.18+ strips types). Use it for quick checks without the UI.

## Rules

- **Keep `tiny-llm.ts` readable top to bottom.** Don't wrap it in a factory or classes, and don't split it into modules. Keep the `// ---- section ----` headers and the short comments that explain *why*.
- **No dependencies in `tiny-llm.ts`.** Vite and TypeScript are dev-only tooling.
- **Don't change the model's maths without being asked.** If you do change it, check that the loss still behaves as described below.
- Keep the UI framework-free (plain DOM and canvas).

## Gotchas

- **Positions are capped at `blockSize` (16).** `wpe[pos]` is undefined beyond that. Pass user text through `cleanPrefix()`, which lowercases it, drops unknown characters and caps it at 15 characters. Unknown characters would otherwise encode to `-1`.
- **The learning-rate schedule assumes a fixed `steps` (1000).** `trainStep()` throws once `step >= steps`; call `reset()` to train again. Don't add a "train more" option without reworking the schedule, because `lr * (1 - step/steps)` goes negative.
- **`reset()` must clear all state:** weights (re-randomised in place), grads, Adam `m`/`v`, and `step`. Adam's bias correction depends on `step`.
- **Training is time-sliced in `ui.ts`** (about 12 ms per animation frame) so the page stays responsive. Don't run a synchronous 1000-step loop on the main thread.
- Inference builds autograd graphs it never uses. This is wasteful but intentional, because it keeps one code path.

## Verifying a change

1. Run `npm run typecheck`.
2. In the browser (`npm run dev`), train for 1000 steps. This takes about 3 s, at roughly 3 ms per step.
3. The moving-average loss should start near ln(27) ≈ 3.3 and finish near **2.0**.
4. Generated names should look name-like (e.g. "javan", "lian").
5. Reset, then train again: it should work a second time, with no console errors.
6. There is no random seed, so check numbers against these ranges rather than exact values.
