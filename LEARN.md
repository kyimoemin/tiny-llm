# How an LLM works, explained with `tiny-llm.ts`

This guide teaches the core ideas behind large language models (LLMs) such as GPT, using [tiny-llm.ts](tiny-llm.ts) as the worked example. That file is a complete GPT in about 280 lines with no libraries. Read it side by side with this guide; each section below points to the code it explains.

The model has 4,192 parameters. GPT-3 has 175 billion. The ideas are the same; only the scale differs.

---

## 0. The one idea everything rests on

**An LLM is a function that, given some text, outputs a probability for every possible next token.**

That's all. Everything else is detail about *how* that function is built and *how* its numbers are tuned.

- **Generating text** = ask for the next-token probabilities, pick one, append it, repeat.
- **Training** = show it real text, measure how surprised it was by each actual next token, and nudge its numbers so it is less surprised next time.

Our model works on first names, one character at a time. Given `"em"`, a trained model says roughly:

| next char | `a` | `i` | `e` | `l` | `y` | … |
|---|---|---|---|---|---|---|
| probability | 0.38 | 0.25 | 0.15 | 0.04 | 0.03 | … |

("emma", "emily", "emery"…). The file is laid out in the same order you need to learn it:

```text
data → tokenizer → autograd → model (gpt) → training (loss + Adam) → inference (sample)
```

---

## 1. Data ([tiny-llm.ts:7](tiny-llm.ts#L7))

```ts
export const docs = `emma olivia ava ...`.split(/\s+/);
```

254 first names. Each name is one "document". A real LLM trains on trillions of tokens of web pages, books and code, but the job is the same: learn the statistical patterns of the text.

**Key point:** the model is never told any rules ("names often end in *a*", "*q* is followed by *u*"). It has to discover them from examples.

---

## 2. Tokenizer: text → numbers ([tiny-llm.ts:30](tiny-llm.ts#L30))

Neural networks work on numbers, not letters. A **tokenizer** maps text to a list of integer ids called **tokens**.

```ts
export const chars = [...new Set(docs.join(""))].sort(); // "abc...z"
export const BOS = chars.length;                         // 26
export const vocab = chars.length + 1;                   // 27
const encode = (s) => [BOS, ...[...s].map((ch) => chars.indexOf(ch)), BOS];
```

- Each letter gets an id: `a`=0, `b`=1, … `z`=25.
- `BOS` (id 26) is a special token meaning "beginning of sequence". It is reused as "end of sequence", which is how the model learns when a name should stop.
- `encode("emma")` → `[26, 4, 12, 12, 0, 26]`.

The **vocabulary size** here is 27. Real LLMs use **subword** tokenizers (such as BPE) with 50k–200k tokens, so `"unbelievable"` might be `["un", "believ", "able"]`. Character-level tokenizing is the simplest choice and makes every step easy to see.

### Turning a name into training examples

One name gives one prediction task per position:

| model sees | must predict |
|---|---|
| `<BOS>` | `e` |
| `<BOS> e` | `m` |
| `<BOS> e m` | `m` |
| `<BOS> e m m` | `a` |
| `<BOS> e m m a` | `<BOS>` (end) |

This is **next-token prediction**, the whole training objective of a GPT.

---

## 3. Autograd: how the model learns which way to move ([tiny-llm.ts:40](tiny-llm.ts#L40))

Training means: "change each parameter a little in the direction that lowers the loss." To do that we need the **gradient**: for every parameter, how much the loss would change if that parameter increased slightly. Computing it by hand for thousands of parameters is impossible, so we automate it. This is **automatic differentiation**, or **autograd**.

### The `V` class

Every number in the model is wrapped in a `V` (for "value"):

```ts
class V {
  data: number;   // the actual number
  grad = 0;       // d(loss)/d(this), filled in by backward()
  kids: V[];      // the inputs that produced this value
  local: number[];// d(this)/d(kid) for each kid
}
```

Every operation creates a new `V` that **remembers its inputs and its local derivative** with respect to each input:

```ts
add(o) { return new V(this.data + o.data, [this, o], [1, 1]); }
mul(o) { return new V(this.data * o.data, [this, o], [o.data, this.data]); }
```

- For `c = a + b`: dc/da = 1, dc/db = 1.
- For `c = a * b`: dc/da = b, dc/db = a.

As the model runs, these links build a **computation graph**: a tree from the final loss back to every parameter.

### `backward()`: the chain rule, automated

```ts
backward() {
  // 1. sort nodes so each one comes after all of its inputs (topological order)
  // 2. walk from the output back to the inputs:
  for (const n of topo.reverse())
    n.kids.forEach((k, i) => (k.grad += n.local[i] * n.grad));
}
```

This is the **chain rule**: if the loss depends on `n`, and `n` depends on `k`, then

```text
d(loss)/dk  +=  d(n)/dk  ×  d(loss)/dn
                (local)      (n.grad)
```

The `+=` matters: when one value feeds several later values (a parameter is used at every position, for example), its gradient is the **sum** of contributions from each path.

### Worked example

```text
a = 2, b = 3
c = a * b     → 6
L = c + a     → 8
```

Backward, starting with `L.grad = 1`:

- `L = c + a`: `c.grad += 1 × 1 = 1`, `a.grad += 1 × 1 = 1`
- `c = a * b`: `a.grad += b × c.grad = 3`, `b.grad += a × c.grad = 2`

Result: `a.grad = 4`, `b.grad = 2`. Check: L = ab + a, so dL/da = b + 1 = 4 and dL/db = a = 2. ✓

PyTorch and JAX do exactly this, but on whole tensors (big arrays) on GPUs instead of one scalar at a time. That is why this file is slow (about 3 ms per step for a tiny model) yet easy to follow.

`sum` and `dot` ([tiny-llm.ts:88](tiny-llm.ts#L88)) are shortcuts: one node with many inputs instead of a long chain of `add` nodes. They give the same gradients with a much smaller graph.

---

## 4. Parameters: the model's "knowledge" ([tiny-llm.ts:102](tiny-llm.ts#L102))

```ts
const nEmbd = 16, nHead = 4, nLayer = 1, blockSize = 16;
```

| name | meaning | here | GPT-3 |
|---|---|---|---|
| `nEmbd` | size of the vector representing each token | 16 | 12,288 |
| `nHead` | number of attention heads | 4 | 96 |
| `nLayer` | number of transformer blocks stacked | 1 | 96 |
| `blockSize` | maximum context length in tokens | 16 | 2,048 |

All learnable numbers are created by `matrix()` and collected in `params`. They start as small random numbers (`gauss() * 0.08`). Small, because big random weights would make early outputs extreme and training unstable. Random, because if every weight started equal, every neuron would compute the same thing and they could never become different.

The weight matrices:

| matrix | shape | job |
|---|---|---|
| `wte` | 27 × 16 | token embedding: one 16-number vector per character |
| `wpe` | 16 × 16 | position embedding: one vector per position 0–15 |
| `wq`, `wk`, `wv`, `wo` | 16 × 16 each | attention (query, key, value, output) |
| `fc1`, `fc2` | 64 × 16, 16 × 64 | the MLP (feed-forward) |
| `lmHead` | 27 × 16 | turns the final vector back into 27 scores |

Total: 432 + 256 + 1024 + 2048 + 432 = **4,192 parameters**. When people say "a 7B model", they mean 7 billion of these numbers.

---

## 5. The forward pass: `gpt()` ([tiny-llm.ts:156](tiny-llm.ts#L156))

`gpt(tok, pos, keys, vals)` takes **one** token and its position, and returns 27 **logits** (raw scores, one per possible next token). Let's walk through it.

### 5.1 Embeddings: a token becomes a vector

```ts
let x = rmsnorm(addVec(wte[tok], wpe[pos]));
```

- `wte[tok]` looks up the token's row: the id becomes a 16-number vector. During training, similar characters (vowels, say) tend to end up with similar vectors.
- `wpe[pos]` adds a vector for the position. Without it, the model couldn't tell "first letter" from "fifth letter", because attention by itself ignores order.
- Adding the two lets one vector carry both "what" and "where".

This vector `x` is called the **residual stream**. Each layer reads from it and adds its result back to it.

### 5.2 RMSNorm: keep the numbers a sensible size ([tiny-llm.ts:137](tiny-llm.ts#L137))

```ts
const rmsnorm = (x) => {
  const inv = (mean(x²) + 1e-5) ^ -0.5;
  return x.map((xi) => xi * inv);
};
```

It divides the vector by its root-mean-square, so its typical value is about 1. Without normalization, values can grow or shrink layer after layer and training breaks down. `1e-5` avoids dividing by zero. GPT-2 uses LayerNorm; modern models such as Llama use RMSNorm, which is simpler.

The model is **pre-norm**: each sub-block normalizes its *input* (`rmsnorm(x)`), while the residual stream `x` itself is left un-normalized.

### 5.3 Self-attention: tokens look at earlier tokens ([tiny-llm.ts:159-179](tiny-llm.ts#L159-L179))

This is the heart of the transformer. Every token produces three vectors from its normalized input `xn`:

```ts
const q = linear(L.wq, xn);           // Query: "what am I looking for?"
keys[l].push(linear(L.wk, xn));       // Key:   "what do I contain?"
vals[l].push(linear(L.wv, xn));       // Value: "what do I pass on if chosen?"
```

`linear(w, x)` is a matrix–vector multiply: each output is a `dot` of one weight row with `x`.

Then the current token compares its query to the key of **every token so far** (itself included):

```ts
const w = softmax(keys[l].map((k) => dot(qh, k_h) * (1/√headDim)));
```

- `dot(q, k)` is large when the query and key point the same way, meaning "this earlier token is relevant to me".
- `softmax` turns these scores into **attention weights** that are positive and sum to 1.
- Dividing by √headDim stops dot products from growing with vector size, which would push softmax towards picking one token with weight 1 and leave almost no gradient to learn from.

The output is a weighted average of the earlier tokens' value vectors:

```ts
attn.push(dot(w, vals[l].map((v) => v[j])));  // for each channel j
```

**Intuition:** predicting what follows `"emm"`, the model might learn a head that attends strongly to the earlier `m` and the start of the name, because "double m after e" is a strong hint that `a` is next.

#### Multi-head attention

The 16 channels are split into 4 **heads** of 4 channels each (`headDim = 4`). Each head runs its own attention independently on its slice, so different heads can learn different patterns (one tracks the previous letter, another tracks the first letter, for example). Their outputs are concatenated back into 16 channels and mixed by `wo`.

#### Causality and the KV cache

A language model must not peek at future tokens; otherwise predicting the next token would be trivial. Big implementations process a whole sequence at once and apply a **causal mask** to hide future positions.

This file does it more simply: it feeds tokens **one at a time** and keeps every past key and value in `keys`/`vals` (the **KV cache**). When token 3 runs, only keys 0–3 exist, so it *cannot* see the future. Causality comes for free.

Real LLMs use a KV cache too, at inference time, so they don't recompute past tokens for every new one. Here the same mechanism serves both training and inference.

### 5.4 The residual connection

```ts
x = addVec(x, linear(L.wo, attn));
```

The attention output is **added** to `x` rather than replacing it. This **residual connection** means each block only needs to learn a *correction*, and it gives gradients a direct path back to early layers. Deep networks can't be trained well without it.

### 5.5 The MLP: thinking about each token on its own ([tiny-llm.ts:180-186](tiny-llm.ts#L180-L186))

```ts
x = addVec(x, linear(L.fc2, linear(L.fc1, rmsnorm(x)).map((v) => v.relu())));
```

- `fc1` expands 16 → 64 dimensions.
- `relu` (`max(0, v)`) is the **non-linearity**. Without it, stacking linear layers would collapse into one linear layer and the model could only learn very simple patterns.
- `fc2` projects 64 → 16, and the result is added back to the residual stream.

A common way to put it: **attention moves information between tokens; the MLP processes it within each token.** One attention plus one MLP is one **transformer block**. Here `nLayer = 1`; GPT-3 stacks 96.

### 5.6 The output head: vector → scores

```ts
return linear(lmHead, x);   // 27 logits
```

Each of the 27 rows of `lmHead` scores how well `x` matches one possible next token. These raw scores are called **logits**.

---

## 6. Softmax: scores → probabilities ([tiny-llm.ts:144](tiny-llm.ts#L144))

```ts
const softmax = (z) => {
  const max = Math.max(...z);
  const e = z.map((x) => exp(x - max));
  return e.map((x) => x / sum(e));
};
```

`exp` makes every score positive, and dividing by the sum makes them add to 1. Subtracting `max` first doesn't change the result (it cancels out) but prevents `exp` from overflowing on large scores. Every real implementation does this.

---

## 7. The loss: measuring surprise ([tiny-llm.ts:209-211](tiny-llm.ts#L209-L211))

```ts
for (let t = 0; t < n; t++)
  logProbs.push(softmax(gpt(toks[t], t, keys, vals))[toks[t + 1]].log());
const loss = sum(logProbs).scale(-1 / n);
```

At every position, look up the probability the model gave to the **correct** next token, and take `-log` of it. This is **cross-entropy loss**:

| model's probability for the right answer | loss = −ln(p) |
|---|---|
| 1.0 (certain and right) | 0 |
| 0.5 | 0.69 |
| 1/27 (pure guessing) | **3.30** |
| 0.01 (confidently wrong) | 4.6 |

The loss is averaged over the positions in the name.

**Why the loss starts near 3.3:** an untrained model with small random weights gives every character about 1/27 probability, and −ln(1/27) = ln 27 ≈ 3.30. A first step that shows a loss near 3.3 tells you the setup is correct.

**Why it ends near 2.0 and not 0:** names are genuinely unpredictable. After `"ma"`, the next letter could reasonably be `d`, `r`, `x`, `y`… No model can be certain. A loss of 2.0 means the model gives the right character about e^−2 ≈ 14% probability on average, against 3.7% for random guessing.

---

## 8. Training: gradient descent with Adam ([tiny-llm.ts:202](tiny-llm.ts#L202))

One call to `trainStep()`:

1. Pick a random name and encode it.
2. Run the forward pass over every position and compute the loss (section 7).
3. `loss.backward()` fills in `.grad` on every parameter (section 3).
4. Update every parameter with **Adam**.
5. Reset the gradients to 0 for the next step.

### Plain gradient descent first

The simplest update is:

```text
p = p − learning_rate × p.grad
```

The gradient points uphill (towards more loss), so we step the other way.

### Adam: smarter steps

```ts
m[i] = b1 * m[i] + (1 - b1) * p.grad;        // running average of gradients
v[i] = b2 * v[i] + (1 - b2) * p.grad ** 2;   // running average of gradient²
const mHat = m[i] / (1 - b1 ** (step + 1));  // bias correction
const vHat = v[i] / (1 - b2 ** (step + 1));
p.data -= (lrT * mHat) / (Math.sqrt(vHat) + 1e-8);
```

- **`m` (momentum)** smooths out noise. Each step sees only one name, so its gradient is noisy; averaging over recent steps gives a steadier direction.
- **`v`** tracks how large a parameter's gradients usually are. Dividing by √v gives each parameter its **own step size**: parameters with consistently large gradients take smaller steps, and rarely updated ones take relatively larger ones.
- **Bias correction**: `m` and `v` start at 0, so early averages are too small. Dividing by `(1 − b^(step+1))` corrects for this. That's why `reset()` must also reset `step`.

### Learning-rate schedule

```ts
const lrT = lr * (1 - step / steps);   // 0.01 → 0 over 1000 steps
```

Big steps early to make fast progress, small steps late to settle into a good spot. Real LLMs use a similar idea (usually a short warm-up, then cosine decay).

### What training looks like

From one real run in Node (exact numbers vary because there is no random seed):

| step | moving-average loss |
|---|---|
| 0 | 3.29 |
| 100 | 2.68 |
| 1000 | 2.07 |

About 3 ms per step, so around 3 seconds in total.

---

## 9. Inference: generating new names ([tiny-llm.ts:265](tiny-llm.ts#L265))

```ts
export function sample(temperature = 0.6, prefix = "") {
  // feed BOS + prefix to fill the KV cache, then loop:
  const probs = softmax(logits.map((z) => z / temperature));
  let r = Math.random();
  let tok = probs.findIndex((p) => (r -= p) < 0);   // pick by probability
  if (tok === BOS) break;                           // model chose "end"
  out += chars[tok];
  logits = gpt(tok, pos, keys, vals);               // feed it back in
}
```

This is **autoregressive generation**: each chosen token becomes the input for the next step. ChatGPT writes its replies the same way, one token at a time.

### Sampling

`findIndex` with `r -= p` is a **weighted random pick**. Think of the probabilities as slices of a line from 0 to 1; `r` lands in one slice. A character with probability 0.38 is chosen 38% of the time. If we always took the most likely character (**greedy decoding**), every run would produce the same name.

### Temperature

The logits are divided by `temperature` before softmax:

| temperature | effect | typical output |
|---|---|---|
| → 0 | almost always picks the top choice | safe, repetitive |
| 0.6 (default) | slightly sharpened | name-like |
| 1.0 | the model's true distribution | more varied |
| > 1.5 | flattened towards random | gibberish |

Sample output from a trained model at 0.6: `kamiin auian elaren misele elavan adery averon deya`. None of these are in the training list. The model learned the *patterns* of names, not just the list.

### Prompting

The `prefix` argument is the tiny version of a **prompt**. `sample(0.6, "ma")` primes the KV cache with `<BOS> m a`, and generation continues from there. A chat prompt to a real LLM works the same way, just with far more tokens.

### Context length

```ts
if (pos === blockSize) break; // no position embedding left
```

`wpe` has only 16 rows, so the model can't handle a 17th position. This is the tiny version of a **context window**. `cleanPrefix()` caps prompts at 15 characters for the same reason.

---

## 10. From tiny-llm to ChatGPT

| | tiny-llm | modern LLM |
|---|---|---|
| tokens | characters (27) | subwords (~100k) |
| parameters | 4,192 | billions to trillions |
| layers | 1 | dozens to 100+ |
| context | 16 | 100k+ tokens |
| data | 254 names | trillions of tokens |
| compute | one CPU thread, scalars | thousands of GPUs, tensors |
| positions | learned `wpe` | usually RoPE (rotary embeddings) |
| activation | ReLU | GELU / SwiGLU |
| after training | done | fine-tuning on instructions, then RLHF, to become an assistant |

Everything in the left column has a direct counterpart on the right. A *pre-trained* LLM is exactly what this file builds: a next-token predictor. The "assistant" behaviour comes from further training on conversations, with the same loss and the same optimizer.

---

## 11. Experiments to try

Make a copy or a branch first, since these change the model.

1. **Watch the untrained model.** Call `nextTokenProbs("em")` before training: the values are all roughly 1/27 ≈ 0.037 (one run gave between 0.02 and 0.09). Train, then call it again and see a few letters take most of the probability.
2. **Temperature sweep.** Generate 10 names each at 0.2, 0.6, 1.0 and 2.0. When do they stop looking like names?
3. **Remove position embeddings.** Replace `addVec(wte[tok], wpe[pos])` with `wte[tok]`. Does the loss get worse? Why?
4. **Remove attention.** Skip the `x = addVec(x, linear(L.wo, attn))` line. Now the model only sees the current character and its position, so it is close to a **bigram model** (one that predicts from the previous character only). Compare the final loss.
5. **Make it bigger.** Try `nLayer = 2` or `nEmbd = 32` (keep `nEmbd` divisible by `nHead`). The loss should drop a little, and each step gets slower.
6. **Change the data.** Replace `docs` with city names, Pokémon-style made-up words, or your own list. Nothing else needs to change.
7. **Check the gradients.** For one parameter, change `p.data` by `+1e-4`, rerun the forward pass and compute `(newLoss − oldLoss) / 1e-4`. It should match `p.grad`. This is how autograd code is tested.

---

## Glossary

- **Token**: the unit of text the model works with (a character here).
- **Embedding**: a learned vector that represents a token or a position.
- **Logits**: raw scores before softmax.
- **Softmax**: turns scores into probabilities that sum to 1.
- **Attention**: each token builds a weighted average of earlier tokens' values, weighted by query–key similarity.
- **Head**: one independent attention computation on a slice of the channels.
- **Residual stream**: the vector `x` that every block reads from and adds to.
- **Cross-entropy loss**: the average −log(probability of the correct next token).
- **Gradient**: how much the loss changes when a parameter changes slightly.
- **Backpropagation**: computing all gradients with the chain rule, from the loss backwards.
- **Adam**: gradient descent with momentum and a per-parameter step size.
- **KV cache**: stored keys and values of past tokens, so they aren't recomputed.
- **Temperature**: a knob for how random sampling is.
- **Context window**: the maximum number of tokens the model can see (`blockSize`).
