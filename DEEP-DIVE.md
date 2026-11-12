# Deep dive: how an LLM works, all the way down

[LEARN.md](LEARN.md) is the short tour. This is the long one. It covers every stage of [tiny-llm.ts](tiny-llm.ts) in depth: the idea behind it, the maths, what the code does, what actually happens inside the trained model, and how real LLMs do it differently. [GRADIENTS.md](GRADIENTS.md) is the companion chapter on derivatives and backpropagation.

**About the numbers.** Every measurement in this guide comes from running this model's code. Some needed a scratch copy with logging or a switch to turn a part off; the real file was not changed. There is no random seed, so your numbers will differ a little; where it matters, results are averaged over 3 runs.

## Contents

1. [The task: language modelling is probability](#1-the-task-language-modelling-is-probability)
2. [Data](#2-data)
3. [Tokenization](#3-tokenization)
4. [Baselines: what "good" means](#4-baselines-what-good-means)
5. [Autograd](#5-autograd)
6. [Parameters and initialisation](#6-parameters-and-initialisation)
7. [Embeddings](#7-embeddings)
8. [Normalisation](#8-normalisation)
9. [Attention](#9-attention)
10. [The residual stream and the MLP](#10-the-residual-stream-and-the-mlp)
11. [The output head](#11-the-output-head)
12. [The loss, in depth](#12-the-loss-in-depth)
13. [The training loop](#13-the-training-loop)
14. [Generalisation and overfitting](#14-generalisation-and-overfitting)
15. [Ablations: which parts matter?](#15-ablations-which-parts-matter)
16. [Inference and sampling](#16-inference-and-sampling)
17. [Cost: why this is slow and real models are fast](#17-cost-why-this-is-slow-and-real-models-are-fast)
18. [From tiny-llm to a chat assistant](#18-from-tiny-llm-to-a-chat-assistant)
19. [Exercises](#19-exercises)

---

## 1. The task: language modelling is probability

A language model assigns a probability to any piece of text. The trick that makes this possible is the **chain rule of probability**: the probability of a sequence is the product of each token's probability given everything before it.

```text
P("emma") = P(e | <BOS>)
          × P(m | <BOS> e)
          × P(m | <BOS> e m)
          × P(a | <BOS> e m m)
          × P(<end> | <BOS> e m m a)
```

This is exact, not an approximation. It turns one impossible problem ("how likely is this whole string?") into many small ones ("how likely is this next token, given the past?"). An LLM is a machine for answering the small question.

That single function gives you everything:

| use | how |
|---|---|
| **score** text | multiply (or add the logs of) each next-token probability |
| **generate** text | pick a next token from the probabilities, append it, repeat |
| **train** | adjust the weights so the probabilities of real text go up |

Every chatbot, code assistant and translation model built on GPT is this loop at a vastly bigger scale.

---

## 2. Data

```ts
export const docs = `emma olivia ava ...`.split(/\s+/);
```

| fact | value |
|---|---|
| names | 254 |
| total characters | 1,469 |
| average length | 5.8 characters (shortest 3, longest 11) |
| next-token predictions per pass over the data | 1,723 (each name's letters plus its end) |
| passes over the data in 1,000 training steps | about 3.9 |

Each name is a **document**: training never lets one name see another. Real LLMs train on documents too (web pages, books, code files), usually packed together with separator tokens so no compute is wasted.

**Why the data is everything.** The model can only learn patterns present in its data. Train it on these names and it learns that names often start with `a`, `j` or `e` and often end in `a` or `n`. Train it on city names and it learns those patterns instead, with no code change. For real LLMs, collecting, filtering and deduplicating data is a huge share of the work.

**Epochs.** One full pass over the data is an **epoch**. Here we do about 4. Big LLMs often see most of their data only about once, because they have so much of it; seeing the same text many times encourages memorising (section 14).

---

## 3. Tokenization

```ts
export const chars = [...new Set(docs.join(""))].sort();  // a–z
export const BOS = chars.length;                          // 26
export const vocab = chars.length + 1;                    // 27
const encode = (s) => [BOS, ...[...s].map((ch) => chars.indexOf(ch)), BOS];
```

### What a tokenizer decides

A tokenizer chooses the **unit** the model predicts. There are three broad options:

| unit | vocabulary | sequence length | trade-off |
|---|---|---|---|
| characters | tiny (27 here) | long | simple, never meets an unknown word, but the model must learn spelling from scratch and sequences get long |
| words | huge (100k+), still incomplete | short | misspellings and new words become "unknown" |
| **subwords** (BPE and similar) | 30k–200k | medium | common words are one token, rare words split into pieces. What real LLMs use |

**Byte-pair encoding (BPE)** builds its vocabulary by starting from single bytes and repeatedly merging the most frequent adjacent pair into a new token. After enough merges, `"the"` is one token and `"tokenization"` might be `"token" + "ization"`. That's why LLMs can be bad at counting letters: they never see the letters, only token ids.

### The BOS token does two jobs

`BOS` is put at the start **and** the end of every name.

- At the start it gives the model something to read before the first letter, so it can predict the first letter (`P(e | <BOS>)`).
- At the end it is the target the model must predict after the last letter. This is how the model learns **when to stop**. During generation, sampling `BOS` means "the name is finished".

Real LLMs have the same idea with special tokens such as `<|endoftext|>`.

### From one name to training pairs

`encode("emma")` is `[26, 4, 12, 12, 0, 26]`. Every position is an input, and the token after it is the target:

| position | input token | target |
|---|---|---|
| 0 | `<BOS>` | `e` |
| 1 | `e` | `m` |
| 2 | `m` | `m` |
| 3 | `m` | `a` |
| 4 | `a` | `<end>` |

Because of attention (section 9), the model at position 3 sees **all** of `<BOS> e m m`, not just `m`.

### Unknown characters

`chars.indexOf(ch)` returns `-1` for any character not in the training data (an uppercase letter, a space, `é`). Index `-1` is not a valid row of the embedding table, so `cleanPrefix()` lowercases user input and drops unknown characters before it reaches the model. Real tokenizers avoid this problem by falling back to raw bytes, so every possible input has some encoding.

---

## 4. Baselines: what "good" means

A loss number means nothing without something to compare it to. Here are simple models built by **counting**, measured on the same data (loss is cross-entropy in nats; section 12 explains the unit):

| model | how it predicts | loss |
|---|---|---|
| uniform | every token 1/27 | **3.30** |
| unigram | how often each token appears overall (`<end>` 15%, `a` 13%, `e` 10%, …) | **2.84** |
| bigram | how often each token follows the previous one (after `a`: `n` 19%, `<end>` 18%, `r` 8%, …) | **2.23** |
| tiny-llm, trained | the transformer | **≈ 2.03** |

These bigram numbers are measured on the same names they were counted from, which flatters the bigram. Section 14 makes a fairer comparison on unseen names.

What this tells us:

- The trained model clearly beats "just look at the previous letter", so it is using **longer context**. That is what attention provides.
- A bigram model is a lookup table of 27 × 27 counts. A neural network is, in a sense, a much more flexible way of building that table. It can share what it learns across similar contexts (section 7) instead of counting each one separately, so it handles contexts it has rarely or never seen.

---

## 5. Autograd

Training needs the gradient of the loss with respect to every one of the 4,192 parameters. The `V` class records every arithmetic operation in a graph, along with the local derivative of each operation. `backward()` then walks the graph in reverse and applies the chain rule, getting all 4,192 gradients in one pass.

This is a big enough topic to have its own guide: **[GRADIENTS.md](GRADIENTS.md)**. It covers derivatives from scratch, gradient descent by hand, the chain rule, why the pass runs backwards, the derivative of every operation in `V`, the `p − y` gradient of softmax plus cross-entropy, and gradient checking.

Two facts from it that the rest of this guide relies on:

1. Autograd needs only each small operation's derivative. Change the model however you like, and the gradients stay correct.
2. For one five-letter name, the graph has about 6,300 nodes. It is rebuilt for every training step.

---

## 6. Parameters and initialisation

### Where the 4,192 parameters live

| component | shape | parameters | share |
|---|---|---|---|
| token embeddings `wte` | 27 × 16 | 432 | 10% |
| position embeddings `wpe` | 16 × 16 | 256 | 6% |
| attention `wq`, `wk`, `wv`, `wo` | 4 × (16 × 16) | 1,024 | 24% |
| MLP `fc1`, `fc2` | 64 × 16 + 16 × 64 | 2,048 | 49% |
| output head `lmHead` | 27 × 16 | 432 | 10% |
| **total** | | **4,192** | |

The MLP holds about half the parameters. That is typical of real transformers as well, where the MLP holds about two-thirds of each layer's weights.

There are **no biases** (the `+ b` in `Wx + b`). Many modern LLMs drop them too: they add little and cost memory. RMSNorm here also has no learnable scale, which real implementations usually have.

### Initialisation

```ts
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
const initWeight = () => gauss() * 0.08;
```

`gauss` is the **Box–Muller transform**: it turns two uniform random numbers into one sample from a normal (bell-curve) distribution. Each weight starts as a normal random number with standard deviation 0.08.

Why this matters:

- **Random, not equal.** If every weight started equal, every neuron in a layer would compute the same thing and receive the same gradient, so they would stay identical for ever. Randomness "breaks the symmetry".
- **Small.** Large initial weights make the logits large, so softmax becomes nearly 0 or 1 everywhere and gradients vanish (see GRADIENTS.md section 13). Small weights give logits near 0, so the untrained model predicts roughly 1/27 for everything and the starting loss is near ln 27 ≈ 3.30. The measured starting loss was 3.32–3.35.
- **Not too small.** With weights of exactly 0, no signal or gradient flows at all.

Real models scale the initial standard deviation with layer size (often around 1/√width) and sometimes shrink it for deeper layers.

---

## 7. Embeddings

```ts
let x = rmsnorm(addVec(wte[tok], wpe[pos]));
```

### A lookup is a matrix multiply

`wte[tok]` picks one row of a 27 × 16 table. Mathematically, that is the same as multiplying a **one-hot** vector (27 zeros with a single 1 at position `tok`) by the matrix. A lookup is just a faster way to do it. So the embedding layer is really a linear layer whose input is "which token is this?".

### What the model learns to put there

At the start, embeddings are random. Training moves them so that tokens which **behave similarly** (appear in similar contexts and predict similar next tokens) end up with **similar vectors**. That lets the model generalise: what it learns about one vowel partly transfers to the others.

Measured on a trained model, using **cosine similarity** (1 = same direction, 0 = unrelated, −1 = opposite):

| pair type | average similarity |
|---|---|
| vowel–vowel (a, e, i, o, u, y) | **+0.20** |
| consonant–consonant | +0.12 |
| vowel–consonant | **−0.18** |

The nearest neighbours of `a` were `u`, `y`, `e`; those of `e` were `a`, `y`, `o`, `i`. The model was never told what a vowel is. It found that vowels behave alike, and it put `y` with them, which fits how `y` works in names such as "emily" and "avery". These are from one run, and the details vary between runs, but vowels clustering apart from consonants is the clear pattern.

### Position embeddings

Attention (section 9) treats the past as a **set**. On its own, it has no notion of order. `wpe[pos]` adds a learned vector for each position 0–15, so "a at position 0" and "a at position 4" look different to the model. Names have strong positional patterns (first letters differ from last letters), so this matters.

The two vectors are **added**, not concatenated. Adding keeps the width at 16 and lets the model learn to use different directions within the 16 dimensions for "what" and "where".

Real models mostly use **RoPE** (rotary position embeddings), which rotate the query and key vectors by an angle that depends on position. Attention scores then depend on the *relative* distance between tokens, and models can be stretched to longer contexts than they were trained on. A learned `wpe` table, as here and in GPT-2, has a hard limit: there are only `blockSize` rows.

---

## 8. Normalisation

```ts
const rmsnorm = (x) => {
  const inv = dot(x, x).scale(1 / x.length).add(num(1e-5)).pow(-0.5);
  return x.map((xi) => xi.mul(inv));
};
```

In maths: `rmsnorm(x) = x / √(mean(x²) + ε)`.

### Why normalise at all?

Each layer adds its output to the residual stream (section 10). Without control, the size of the vectors can drift up or down through the network, and gradients drift with them. Normalising before each sub-block means every block sees inputs of a predictable size (root-mean-square ≈ 1), whatever happened before. Training is much more stable as a result, and higher learning rates become usable.

### RMSNorm vs LayerNorm

GPT-2 uses **LayerNorm**, which first subtracts the mean, then divides by the standard deviation, then applies a learnable scale and shift. **RMSNorm** skips the mean subtraction and the shift. It is cheaper and works about as well, which is why Llama and most recent models use it. Real RMSNorm has a learnable per-channel scale; this file leaves it out for simplicity.

`ε = 1e-5` stops division by zero if `x` is all zeros.

### Pre-norm

This model is **pre-norm**: each block normalises its **input** (`rmsnorm(x)`), and the block's output is added to the un-normalised `x`. The original 2017 transformer was post-norm (normalise *after* adding), which is harder to train deep. Almost every modern LLM is pre-norm.

This file also normalises the embedding itself (`rmsnorm(addVec(...))`) before the first block. GPT-2 doesn't do that; it's a small extra here.

---

## 9. Attention

Attention is what makes a transformer a transformer. It is how a token gathers information from earlier tokens.

### 9.1 The idea: soft lookup

Think of a dictionary lookup: you have a **query**, you compare it to every **key**, and you return the **value** of the key that matches. Attention is a *soft* version: instead of returning one value, it returns a **weighted average of all values**, weighted by how well each key matches the query.

Each token produces all three from its (normalised) vector:

```ts
const q = linear(L.wq, xn);          // what I'm looking for
keys[l].push(linear(L.wk, xn));      // what I offer to be found by
vals[l].push(linear(L.wv, xn));      // what I hand over if found
```

Because `wq`, `wk` and `wv` are learned, the model decides what "looking for" and "offering" mean.

### 9.2 The maths, step by step

For the current token at position `t`, with query `q` and keys `k₀ … k_t`:

```text
score_j  = (q · k_j) / √d        for j = 0 … t
weight_j = softmax(score)_j      positive, sum to 1
output   = Σ_j weight_j × v_j
```

`d` is the head size (`headDim = 4`).

### 9.3 Why divide by √d?

If the entries of `q` and `k` are random with variance 1, their dot product is a sum of `d` such products, so its variance is `d` and its typical size is √d. With large `d`, scores become large, softmax becomes nearly one-hot, and its gradient becomes nearly zero, so learning stalls. Dividing by √d brings the variance back to about 1, whatever the head size. Here `d = 4`, so scores are divided by 2.

### 9.4 Multiple heads

```ts
for (let h = 0; h < nHead; h++) {
  const s = h * headDim, e = s + headDim;
  const qh = q.slice(s, e);
  // attention using only channels s..e of q, k and v
}
```

The 16 channels are split into 4 heads of 4 channels. Each head does its own attention with its own slice, so it can attend to different positions for a different reason. The 4 outputs are joined back into 16 channels, and `wo` mixes them.

Multi-head attention costs the same as one big head (same total channels) but gives the model several independent "lookups" per layer.

### 9.5 What the heads actually do

Attention weights recorded from a trained model, for the name `"emma"`. The input sequence is `<BOS> e m m a`:

At **position 3** (input: the second `m`, target: `a`):

| head | `<BOS>` | `e` | `m` | `m` |
|---|---|---|---|---|
| 0 | 0.09 | **0.50** | 0.19 | 0.22 |
| 1 | 0.01 | **0.99** | 0.00 | 0.00 |
| 2 | 0.12 | **0.62** | 0.04 | 0.22 |
| 3 | 0.03 | **0.95** | 0.00 | 0.01 |

At **position 4** (input: `a`, target: end of name):

| head | `<BOS>` | `e` | `m` | `m` | `a` |
|---|---|---|---|---|---|
| 0 | 0.27 | 0.15 | 0.14 | 0.21 | 0.22 |
| 1 | 0.14 | 0.33 | 0.15 | 0.14 | 0.24 |
| 2 | 0.04 | 0.00 | **0.82** | 0.14 | 0.00 |
| 3 | 0.03 | **0.85** | 0.05 | 0.04 | 0.02 |

Things to notice:

- Heads **look back** at earlier letters, not just the current one. When predicting what follows `e m m`, two heads put almost all their weight on the `e`, two positions back. A bigram model can't do this.
- **Heads differ.** At position 4, head 2 focuses on the first `m` and head 3 on the `e`, while heads 0 and 1 spread their attention out.
- **This is not hand-designed.** All of it emerged from training on 254 names. The patterns vary from run to run, and a 4-dimensional head in a 1-layer model is too small to have clean, human-readable roles. In large models, researchers do find interpretable heads, such as "previous-token heads" and "induction heads" that copy patterns seen earlier in the context.

### 9.6 Causality: no peeking at the future

During training, the target at position `t` is the token at `t + 1`. If position `t` could attend to position `t + 1`, it could just copy the answer. Attention must be **causal**: position `t` may only see positions `0 … t`.

Standard implementations process the whole sequence at once as matrices and add a **causal mask** that sets the scores for future positions to −∞ (so their softmax weights are 0).

This file avoids masks entirely. `gpt()` processes **one token at a time**, and `keys[l]` and `vals[l]` contain only tokens already processed. When position 3 runs, keys 4 and later don't exist yet. Causality is guaranteed by construction.

### 9.7 The KV cache

Keeping past keys and values around is called the **KV cache**. Real LLMs use it at inference time: when generating token 1,001, they reuse the stored keys and values of tokens 1–1,000 instead of recomputing them. Without it, generation would cost roughly quadratic work per token instead of linear. The KV cache is also why long conversations need so much GPU memory.

Here, the same cache mechanism is used for both training and inference, so there is one code path.

### 9.8 Cost

Each token computes a score against every earlier token, so a sequence of length `T` costs about `T²/2` score computations per head per layer. At `T = 16` that's nothing. At `T = 100,000` it's the main cost of a long-context model, which is why techniques such as FlashAttention (a memory-efficient way to compute exact attention) and grouped-query attention (sharing keys and values across heads to shrink the cache) exist.

---

## 10. The residual stream and the MLP

```ts
x = addVec(x, linear(L.wo, attn));                                    // attention block
x = addVec(x, linear(L.fc2, linear(L.fc1, rmsnorm(x)).map((v) => v.relu())));  // MLP block
```

### The residual stream

The vector `x` runs straight through the model from the embedding to the output head. Each block **reads** it (through a norm), computes something, and **adds** its result back. Nothing ever overwrites `x`.

There are two ways to see why this matters:

- **For learning:** the addition has a local derivative of 1, so gradients flow from the loss straight back to the embeddings without being multiplied by anything. Deep networks without residual connections suffer badly from vanishing gradients. GRADIENTS.md section 10 traces this.
- **For understanding:** the stream acts like a shared bus. The embedding writes "what token, what position"; attention writes "here is what I gathered from earlier tokens"; the MLP writes "here is what I worked out". The output head reads the sum.

### The MLP

```text
h   = relu(fc1 · rmsnorm(x))     16 → 64
out = fc2 · h                    64 → 16
```

Attention mixes information **between** positions, but it is mostly a weighted average, which is fairly linear. The MLP works on **each position separately** and adds the **non-linear** processing:

- **Expand to 64 (4×).** This gives 64 "detector" neurons. Each one computes a weighted combination of the 16 inputs.
- **ReLU.** `max(0, v)`: each neuron fires only when its pattern is present. Without a non-linearity, `fc2 · fc1` would just be another matrix, and stacking layers would add nothing.
- **Project back to 16.** Each firing neuron writes its learned "output direction" into the stream.

The 4× expansion is a convention dating back to the original transformer; most models use something between 2.7× and 4×.

A common picture from interpretability research is that MLP layers act partly like a **key–value memory**: `fc1` rows detect patterns, and `fc2` columns write associated information. In large models, many facts appear to be stored this way. Treat it as a useful intuition rather than a complete explanation.

Modern models usually replace ReLU with **GELU** or **SwiGLU**, smoother variants that train a little better and avoid "dead" neurons whose gradient is always 0. GRADIENTS.md section 9.4 shows a dead ReLU in this model.

### One block = attention + MLP

`nLayer = 1` here. Real models stack dozens of identical blocks (with separate weights). Later blocks build on what earlier blocks wrote into the stream, so a deep model can compose simple patterns into complex ones.

---

## 11. The output head

```ts
return linear(lmHead, x);   // 27 logits
```

`lmHead` has one 16-number row per token. Each logit is the dot product of the final stream vector with that row: "how much does `x` point in the direction of token `i`?". Softmax then turns the 27 logits into probabilities.

**Weight tying.** `lmHead` has the same shape as `wte` (27 × 16). GPT-2 and many other models use the **same** matrix for both, called **weight tying**. The idea: a token's input embedding and its output direction should be related, and tying saves parameters (a lot of them when the vocabulary has 100k tokens). This file keeps them separate, which is simpler to read.

---

## 12. The loss, in depth

```ts
logProbs.push(softmax(gpt(toks[t], t, keys, vals))[toks[t + 1]].log());
const loss = sum(logProbs).scale(-1 / n);
```

### Cross-entropy

For each position, take the probability the model gave to the actual next token, and compute `−ln(p)`. Average over the name. This is **cross-entropy**, or equivalently the **negative log-likelihood** of the data.

Why the log?

- **Probabilities multiply; logs add.** The probability of a whole name is a product (section 1), so its log is a sum. Sums are easy to average and differentiate, and they don't underflow to 0 the way products of small probabilities do.
- **It punishes confident mistakes.** Giving the right answer p = 0.01 costs 4.6, while p = 0.5 costs only 0.69. As p → 0, the loss → ∞.
- **It rewards honesty.** Cross-entropy is minimised only when the predicted probabilities match the true frequencies. A model can't game it by always being overconfident.

### The units: nats, bits and perplexity

Using `ln` measures the loss in **nats**; using `log₂` would measure it in **bits** (1 nat ≈ 1.44 bits). A loss of 2.03 nats ≈ 2.9 bits per character: on average, the model needs about 2.9 yes/no questions' worth of information to pin down each next character.

**Perplexity** = e^loss turns the loss into "how many equally likely options the model is effectively choosing between":

| model | loss | perplexity |
|---|---|---|
| uniform | 3.30 | 27 |
| unigram | 2.84 | 17 |
| bigram | 2.23 | 9.3 |
| tiny-llm on training names | 2.03 | 7.6 |
| tiny-llm on unseen names (section 14) | ≈ 2.39 | ≈ 11 |

### Why the loss can't reach 0

Names are genuinely uncertain. After `"emm"` the trained model says `a` 25%, `i` 24%, `e` 17%, and both "emma" and "emmi…"-style names are plausible. The best possible loss is the true **entropy** of name-generation, which is well above 0. Pushing the training loss much lower than that would mean memorising the list, not learning names (section 14).

---

## 13. The training loop

```ts
export function trainStep(): number {
  const toks = encode(docs[Math.floor(Math.random() * docs.length)]);  // 1. one random name
  // 2. forward over every position, collecting log-probs
  const loss = sum(logProbs).scale(-1 / n);                             // 3. mean loss
  loss.backward();                                                      // 4. gradients
  // 5. Adam update, then zero the grads
}
```

### Batch size 1

Each step trains on **one** name, about 6 predictions. That is **stochastic gradient descent** with a batch size of 1. The gradient from one name is a noisy estimate of the gradient over all names: it points roughly downhill, but wobbles. This is why:

- the per-step loss jumps around a lot (a short name like "ava" and a long one like "christopher" get very different losses), so the UI shows a **moving average**;
- Adam's momentum (below) helps by averaging gradients across steps.

Real LLMs use batches of millions of tokens per step. That gives much smoother gradients, and the work parallelises across GPUs.

### Adam

```ts
m[i] = b1 * m[i] + (1 - b1) * p.grad;
v[i] = b2 * v[i] + (1 - b2) * p.grad ** 2;
const mHat = m[i] / (1 - b1 ** (step + 1));
const vHat = v[i] / (1 - b2 ** (step + 1));
p.data -= (lrT * mHat) / (Math.sqrt(vHat) + 1e-8);
```

| piece | what it is | why |
|---|---|---|
| `m` | exponential moving average of the gradient (b1 = 0.85, so roughly the last 7 steps) | smooths out the noise of batch size 1; gives "momentum" |
| `v` | moving average of the squared gradient (b2 = 0.99, so roughly the last 100 steps) | measures each parameter's typical gradient size |
| `mHat`, `vHat` | bias-corrected `m`, `v` | both start at 0, so early averages are too small; dividing by `1 − bᵗ` fixes that |
| `mHat / √vHat` | normalised step | each parameter moves by about `lr`, whatever the scale of its gradients |
| `1e-8` | tiny constant | avoids dividing by 0 |

The effect is that parameters with rare or small gradients (such as the embedding of `q`, which appears in only a few names) still get meaningful updates. Plain SGD with one learning rate would barely move them.

These are lower betas than the usual defaults (0.9 and 0.999), so the averages react faster. That suits a short, 1,000-step run.

Real LLMs mostly use **AdamW**, which adds weight decay (gently shrinking all weights each step to discourage over-reliance on any one weight).

### The learning-rate schedule

```ts
const lrT = lr * (1 - step / steps);   // linear decay from 0.01 to 0
```

Early on, large steps make fast progress. Later, smaller steps let the weights settle instead of bouncing around a minimum. Because the formula reaches 0 at step 1,000 and would go **negative** after that, `trainStep()` refuses to run past `steps`. Real LLMs typically use a short **warm-up** (rising from 0 for the first few thousand steps, so Adam's estimates settle before big steps are taken) followed by cosine or linear decay.

### `reset()` must clear everything

`reset()` re-randomises the weights **in place**, zeroes the gradients, zeroes `m` and `v`, and sets `step = 0`. Leftover `m`/`v` would push the new run in the old run's direction, and a leftover `step` would break bias correction and the learning-rate schedule.

### What training looks like

Loss over the **whole** dataset (not one name), averaged over 3 runs:

| step | 0 | 50 | 100 | 250 | 500 | 750 | 1,000 |
|---|---|---|---|---|---|---|---|
| loss | 3.34 | 2.57 | 2.48 | 2.34 | 2.21 | 2.10 | 2.03 |

- **Steps 0–50:** a fast drop from 3.34 to 2.57, already past the unigram baseline (2.84). The model learns the easy things first: letter frequencies, and that names end.
- **Steps 50–500:** steady gains as it learns bigram-like patterns, reaching the bigram score (2.23) around step 500.
- **Steps 500–1,000:** slow improvement past the bigram as it learns to use longer context through attention, helped by the shrinking learning rate.

---

## 14. Generalisation and overfitting

The training loss says how well the model fits **the names it trained on**. The real question is how well it does on names it has **never seen**. To measure that, hold some data out.

Experiment: shuffle the 254 names, hold out 50, train on the other 204, then measure both sets. To make the comparison fair, the bigram baseline here is counted on the same 204 names, with add-one smoothing so that unseen pairs don't get probability 0.

| | training names | unseen names |
|---|---|---|
| tiny-llm (3 runs) | 1.94–1.97 | **2.37–2.44** |
| bigram (smoothed) | 2.41–2.42 | 2.50–2.51 |

What this shows:

- There is a clear **gap** (about 0.43) between seen and unseen names. Part of what the model learned is specific to the names it saw: it is partly **memorising**. With only 204 names and about 4 passes over them, that's expected.
- It still **generalises**: on unseen names it beats the bigram (2.39 against 2.50). Some of what it learned is about names in general.
- This repo has **no validation set**. The loss in the UI is training loss, so it measures fit, not generalisation. Every serious training run holds data out and watches the validation loss; when that loss stops improving while the training loss keeps falling, the model is **overfitting**.

Ways to reduce overfitting: more data (by far the most effective), fewer passes over the same data, a smaller model, weight decay and dropout.

---

## 15. Ablations: which parts matter?

An **ablation** removes one component and measures the damage. Each row is the final whole-dataset loss after 1,000 steps, averaged over 3 runs, using a scratch copy of the model with switches to turn parts off:

| variant | loss | what the model can use |
|---|---|---|
| full model | **2.03** | token, position, earlier tokens (attention) |
| no position embeddings | 2.08 | token, earlier tokens, but no explicit order |
| no attention | 2.11 | current token and its position only |
| neither | 2.33 | current token only (a neural bigram) |

Reading it:

- **"Neither" is roughly a bigram model** (2.33, against the counted bigram's 2.23; it doesn't quite reach that in 1,000 steps).
- **Position alone helps a lot** (2.33 → 2.11). Knowing "this is the first letter" or "we're 6 letters in, so the name should probably end" is valuable.
- **Attention adds more** (2.11 → 2.03): looking back at earlier letters beats knowing only the current one.
- **Removing position hurts less than expected** (2.08). One plausible reason: with causal attention, the model can partly work out position by itself. For instance, how much weight lands on the `<BOS>` token depends on how many tokens there are.

The differences are modest because names are short and the model is tiny. In real LLMs, where context spans thousands of tokens, attention is essential.

---

## 16. Inference and sampling

### Priming

```ts
function prime(prefix) {
  const toks = [BOS, ...prefix tokens];
  toks.forEach((tok, pos) => (logits = gpt(tok, pos, keys, vals)));
  return { keys, vals, logits, pos: toks.length };
}
```

Generation starts by feeding `<BOS>` and any prefix through the model. That fills the KV cache and gives the logits for the next token. With a chat LLM this is the **prefill** phase: your whole prompt is processed before the first word of the reply appears.

### The generation loop

```ts
for (;; pos++) {
  const probs = softmax(logits.map((z) => z.scale(1 / temperature)));
  // pick a token at random according to probs
  if (tok === BOS) break;          // the model chose to end the name
  out += chars[tok];
  if (pos === blockSize) break;    // out of position embeddings
  logits = gpt(tok, pos, keys, vals);  // feed it back in
}
```

This is **autoregressive** decoding: every output becomes the next input. Each new token needs only one `gpt()` call, thanks to the KV cache.

### Sampling, greedy and the rest

`probs.findIndex((p) => (r -= p.data) < 0)` draws a token with probability `p`: lay the probabilities end to end along the line from 0 to 1, drop `r` at a random point, and see which segment it lands in.

The `if (tok < 0) tok = probs.length - 1` guard handles floating-point rounding: if the probabilities sum to 0.9999999 and `r` lands in that last sliver, no segment is hit.

Other strategies used by real systems:

| strategy | how | effect |
|---|---|---|
| **greedy** | always take the most likely token | deterministic; tends to be repetitive |
| **temperature** | divide logits by T before softmax | T < 1 sharpens, T > 1 flattens |
| **top-k** | sample only from the k most likely tokens | cuts off the long tail of bad options |
| **top-p (nucleus)** | sample from the smallest set whose probabilities sum to p | like top-k, but adapts to how confident the model is |

tiny-llm uses only temperature.

### Temperature in depth

Dividing every logit by T before softmax:

- **T → 0:** the largest logit dominates, so sampling becomes greedy.
- **T = 1:** the model's own distribution.
- **T → ∞:** all logits shrink towards 0, so sampling becomes uniform.

Measured with 300 samples per temperature from a trained model:

| T | names copied from training data | unique names | average length | examples |
|---|---|---|---|---|
| 0.1 | 86% | 4% | 4.1 | alex alia alex alex alison alice |
| 0.3 | 20% | 45% | 4.6 | alide alia caron luia sofia alison |
| 0.6 | 6% | 90% | 4.9 | araly camla taver carron axelen sonson |
| 1.0 | 1% | 99% | 5.4 | miaze alery rsacon keliab amiel taliab |
| 1.5 | 0% | 99% | 6.0 | hdcrer onowen carary whmab raknys |
| 2.0 | 0% | 97% | 6.9 | wpbbahr madevey calvein xosusivu |

(Real names average 5.8 letters.)

This one table shows the whole creativity/quality trade-off:

- **Low T** collapses onto the model's single most likely path, which is mostly memorised names ("alex" over and over). It's safe and boring.
- **T ≈ 0.6** gives novel but plausible names. That's why it's the default.
- **High T** gives the improbable tail too much weight: letter combinations that never happen ("wpbb", "hdcr"). Names also get longer, because the end token is one option among 27, and flattening the distribution makes it less likely to win at any given step.

### Prompting with a prefix

`sample(0.6, "ma")` gave: `mabrde mane mayth man may madel maydeel madie`. The prefix steers everything after it, which is exactly how prompts steer a chat model.

Even an unusual prefix works: `"zz"` (no name in the data starts that way) gave `zzaan zzler zzeleh zzelen zzlla`. The model continues with plausible name-like letters, because attention and the MLP generalise beyond exact contexts they've seen.

### The context limit

`wpe` has 16 rows, for positions 0–15, so the model can read at most 16 tokens: `<BOS>` plus 15 letters. It can still *predict* a 16th letter from position 15, but that letter can never be fed back in, so `sample()` stops there. `cleanPrefix()` caps prompts at 15 characters so that `<BOS>` plus the prompt fits. This is the tiny version of a model's **context window**.

---

## 17. Cost: why this is slow and real models are fast

### Counting the work

For one token, the forward pass does roughly:

| part | multiply–adds |
|---|---|
| q, k, v, output projections | 4 × 256 = 1,024 |
| MLP | 2 × 1,024 = 2,048 |
| output head | 432 |
| attention scores and sums | a few hundred at most |
| **total** | **≈ 3,500** |

That's about **one multiply–add per non-embedding parameter**. This is a general rule: a forward pass costs about **2 × parameters FLOPs per token** (a multiply–add counts as 2 floating-point operations), and training (forward + backward) costs about **6 × parameters FLOPs per token**. People use this rule to estimate the cost of training real LLMs.

### Why it still takes 3 ms per step

A modern CPU can do billions of operations per second, and one training step is only about 100,000 FLOPs. The cost is almost entirely **overhead**: every scalar operation allocates a JavaScript object, and one name builds about 6,300 graph nodes, which must then be sorted, walked backwards and garbage-collected.

Real frameworks remove that overhead in three ways:

1. **Tensors:** one graph node per *matrix* operation instead of per number, so the graph is thousands of times smaller.
2. **Optimised kernels:** matrix multiplies run as tight, vectorised loops on hardware built for them.
3. **Parallelism:** a GPU does tens of thousands of multiply–adds at once, and training spreads over thousands of GPUs (splitting the data, the layers and the matrices themselves).

The maths is identical. Only the bookkeeping is different.

---

## 18. From tiny-llm to a chat assistant

### Scale

| | tiny-llm | GPT-2 small (2019) | GPT-3 (2020) |
|---|---|---|---|
| parameters | 4,192 | 124 million | 175 billion |
| layers | 1 | 12 | 96 |
| width (`nEmbd`) | 16 | 768 | 12,288 |
| heads | 4 | 12 | 96 |
| context | 16 | 1,024 | 2,048 |
| vocabulary | 27 | 50,257 | 50,257 |
| training data | 1,469 characters | ~40 GB of text | ~300 billion tokens |

Newer models are larger again, with much longer contexts and far more training data.

A key empirical finding of LLM research (the **scaling laws**) is that loss goes down smoothly and predictably as you scale up parameters, data and compute together. No new idea was needed to get from GPT-2 to GPT-3: mostly more of the same.

### Architecture changes since GPT-2

| tiny-llm / GPT-2 | common in modern LLMs | why |
|---|---|---|
| learned position table | RoPE | relative positions; extends to longer contexts |
| LayerNorm (here RMSNorm) | RMSNorm with learnable scale | cheaper, just as good |
| ReLU / GELU | SwiGLU | trains slightly better |
| every head has its own K and V | grouped-query attention | much smaller KV cache |
| dense MLP | sometimes mixture-of-experts | more parameters for the same compute per token |

The overall shape (embed → blocks of attention + MLP with residuals and norms → output head → softmax) is unchanged.

### From text predictor to assistant

What tiny-llm builds, a model trained only to predict the next token, is called a **base** or **pre-trained** model. A base model trained on the internet continues text; it doesn't follow instructions. Turning it into an assistant takes more training stages:

1. **Supervised fine-tuning (SFT):** train on example conversations written to show good assistant behaviour. It is the same loss and the same next-token objective, just different data.
2. **Preference training (RLHF and related methods):** people (or models) compare pairs of responses, and the model is trained to make the preferred kind more likely.
3. **Chat formatting:** a conversation is just one long token sequence with special tokens marking who is speaking. The model generates the assistant's turn exactly as `sample()` generates a name, one token at a time, until it samples an end-of-turn token, the equivalent of our `BOS`.

At every stage, underneath, it is still this file's loop: tokens in, `gpt()`, softmax, sample, repeat.

### Where to go next

- *Attention Is All You Need* (Vaswani et al., 2017): the original transformer paper.
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2014).
- Andrej Karpathy's *micrograd*, *makemore* and *nanoGPT* projects and his "Let's build GPT" video. They cover the same ground in Python and PyTorch, one step up in scale from this file.

---

## 19. Exercises

Do these on a branch or a copy. Expected results come from the experiments above, so yours should land nearby.

1. **Beat the baselines.** Compute the unigram and bigram losses yourself by counting (section 4). Expect about 2.84 and 2.23. Then train and compare against the model's whole-dataset loss.
2. **Probability of a name.** Using `nextTokenProbs`, compute P("emma") with the chain rule from section 1. Compare it with P("emmz"). How many times more likely is the real name?
3. **Look at the embeddings.** After training, compute the cosine similarity between every pair of rows in `wte`. Do the vowels cluster together? Where does `y` land? (One run gave vowel–vowel +0.20 and vowel–consonant −0.18.)
4. **Look at attention.** Log the attention weights `w` inside `gpt()` for a few names. Can you find a head that usually looks at `<BOS>`, or at the previous letter?
5. **Hold out data.** Split off 50 names, train on the rest, and measure the loss on both sets (section 14). Expect a gap of about 0.4.
6. **Ablate.** Remove the position embeddings, then attention, then both (section 15). Expect about 2.08, 2.11 and 2.33.
7. **Temperature.** Sample 300 names at T = 0.1 and T = 1.5. What fraction are copied from the training list? (Expect about 86% and 0%.)
8. **Make it bigger.** Try `nLayer = 2`, then `nEmbd = 32`. Does the training loss fall? Does the held-out loss fall too, or does the gap widen?
9. **Tie the weights.** Use `wte` in place of `lmHead` (section 11). How many parameters do you save, and how does the loss change?
10. **Add top-k.** Change `sample()` to sample only from the 5 most likely tokens. Compare the names at T = 1.5 with and without it.
