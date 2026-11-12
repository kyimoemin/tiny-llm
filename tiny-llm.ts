// tiny-llm.ts: a complete GPT in one file with zero dependencies.
// Tokenizer, autograd, transformer, Adam, training step, sampling.
// This file only defines the model; ui.ts drives it from index.html.
// Run the UI: npm install && npm run dev

// ---- data: any list of short strings works; swap in your own ----
export const docs =
  `emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily
elizabeth mila ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope
layla riley zoey nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe leah
hazel violet aurora savannah audrey brooklyn bella claire skylar lucy paisley everly anna caroline
nova genesis emilia kennedy samantha maya willow kinsley naomi aaliyah elena sarah ariana allison
gabriella alice madelyn cora ruby eva serenity autumn adeline hailey gianna valentina isla eliana
quinn nevaeh ivy sadie piper lydia alexa josephine emery julia delilah arianna vivian kaylee sophie
brielle madeline liam noah william james oliver benjamin elijah lucas mason logan alexander ethan
jacob michael daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt
john jack luke jayden dylan grayson levi isaac gabriel julian mateo anthony jaxon lincoln joshua
christopher andrew theodore caleb ryan asher nathan thomas leo isaiah charles josiah hudson
christian hunter connor eli ezra aaron landon adrian jonathan nolan jeremiah easton elias colton
cameron carson robert angel maverick nicholas dominic jaxson greyson adam ian austin santiago
jordan cooper brayden roman evan ezekiel xavier jose jace jameson leonardo bryson axel everett
parker kayden miles sawyer jason declan weston micah ayden wesley luca vincent damian zachary
silas gavin chase kai emmett harrison nathaniel kingston cole tyler bennett bentley ryker tristan
brandon kevin luis george ashton rowan braxton ryder gael ivan diego maxwell max carlos kaiden
juan maddox justin waylon calvin giovanni jonah abel jayce jesus amir king beau camden alex jasper`.split(
    /\s+/
  );

// ---- tokenizer: one token per character, plus BOS (also used as end-of-text) ----
export const chars = [...new Set(docs.join(""))].sort();
export const BOS = chars.length;
export const vocab = chars.length + 1;
const encode = (s: string) => [
  BOS,
  ...[...s].map((ch) => chars.indexOf(ch)),
  BOS,
];

// ---- autograd: a scalar that remembers its inputs and local derivatives ----
class V {
  data: number;
  grad = 0;
  kids: V[];
  local: number[];
  constructor(data: number, kids: V[] = [], local: number[] = []) {
    this.data = data;
    this.kids = kids;
    this.local = local;
  }
  add(o: V) {
    return new V(this.data + o.data, [this, o], [1, 1]);
  }
  mul(o: V) {
    return new V(this.data * o.data, [this, o], [o.data, this.data]);
  }
  scale(k: number) {
    return new V(this.data * k, [this], [k]);
  }
  pow(k: number) {
    return new V(this.data ** k, [this], [k * this.data ** (k - 1)]);
  }
  exp() {
    const e = Math.exp(this.data);
    return new V(e, [this], [e]);
  }
  log() {
    return new V(Math.log(this.data), [this], [1 / this.data]);
  }
  relu() {
    return new V(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }
  backward() {
    const topo: V[] = [];
    const seen = new Set<V>();
    const build = (n: V) => {
      if (seen.has(n)) return;
      seen.add(n);
      n.kids.forEach(build);
      topo.push(n);
    };
    build(this);
    this.grad = 1;
    for (const n of topo.reverse())
      n.kids.forEach((k, i) => (k.grad += n.local[i] * n.grad));
  }
}
const num = (x: number) => new V(x);
const sum = (xs: V[]) =>
  new V(
    xs.reduce((s, x) => s + x.data, 0),
    xs,
    xs.map(() => 1)
  );
const dot = (a: V[], b: V[]) =>
  new V(
    a.reduce((s, x, i) => s + x.data * b[i].data, 0),
    [...a, ...b],
    [...b.map((x) => x.data), ...a.map((x) => x.data)]
  );

// ---- model: GPT-2 shape (pre-norm, residuals), RMSNorm, ReLU, no biases ----
const nEmbd = 16,
  nHead = 4,
  nLayer = 1,
  blockSize = 16;
export { blockSize };
const headDim = nEmbd / nHead;

export const params: V[] = [];
const gauss = () =>
  Math.sqrt(-2 * Math.log(1 - Math.random())) *
  Math.cos(2 * Math.PI * Math.random());
const initWeight = () => gauss() * 0.08;
const matrix = (rows: number, cols: number) =>
  Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => {
      const p = new V(initWeight());
      params.push(p);
      return p;
    })
  );

const wte = matrix(vocab, nEmbd); // token embeddings
const wpe = matrix(blockSize, nEmbd); // position embeddings
const layers = Array.from({ length: nLayer }, () => ({
  wq: matrix(nEmbd, nEmbd),
  wk: matrix(nEmbd, nEmbd),
  wv: matrix(nEmbd, nEmbd),
  wo: matrix(nEmbd, nEmbd),
  fc1: matrix(4 * nEmbd, nEmbd),
  fc2: matrix(nEmbd, 4 * nEmbd),
}));
const lmHead = matrix(vocab, nEmbd);

const linear = (w: V[][], x: V[]) => w.map((row) => dot(row, x));
const addVec = (a: V[], b: V[]) => a.map((x, i) => x.add(b[i]));
const rmsnorm = (x: V[]) => {
  const inv = dot(x, x)
    .scale(1 / x.length)
    .add(num(1e-5))
    .pow(-0.5);
  return x.map((xi) => xi.mul(inv));
};
const softmax = (z: V[]) => {
  const max = Math.max(...z.map((x) => x.data));
  const e = z.map((x) => x.add(num(-max)).exp());
  const inv = sum(e).pow(-1);
  return e.map((x) => x.mul(inv));
};

type KVCache = V[][][]; // [layer][position][channel]
const newKVCache = (): KVCache => layers.map(() => []);

// One token in, next-token logits out. Past keys/values live in the caches,
// so causality is free: a token can only attend to what was pushed before it.
function gpt(tok: number, pos: number, keys: KVCache, vals: KVCache): V[] {
  let x = rmsnorm(addVec(wte[tok], wpe[pos]));
  layers.forEach((L, l) => {
    const xn = rmsnorm(x);
    const q = linear(L.wq, xn);
    keys[l].push(linear(L.wk, xn));
    vals[l].push(linear(L.wv, xn));
    const attn: V[] = [];
    for (let h = 0; h < nHead; h++) {
      const s = h * headDim,
        e = s + headDim;
      const qh = q.slice(s, e);
      const w = softmax(
        keys[l].map((k) => dot(qh, k.slice(s, e)).scale(headDim ** -0.5))
      );
      for (let j = s; j < e; j++)
        attn.push(
          dot(
            w,
            vals[l].map((v) => v[j])
          )
        );
    }
    x = addVec(x, linear(L.wo, attn));
    x = addVec(
      x,
      linear(
        L.fc2,
        linear(L.fc1, rmsnorm(x)).map((v) => v.relu())
      )
    );
  });
  return linear(lmHead, x);
}

// ---- train: Adam, one document per step, linear LR decay ----
export const steps = 1000;
const lr = 0.01,
  b1 = 0.85,
  b2 = 0.99;
const m = new Float64Array(params.length); // Adam 1st moment (running mean of grads)
const v = new Float64Array(params.length); // Adam 2nd moment (running mean of grad²)
let step = 0;
export const currentStep = () => step;

// One optimisation step on one random name. Returns that name's mean loss.
export function trainStep(): number {
  if (step >= steps) throw new Error("training finished; call reset()");
  const toks = encode(docs[Math.floor(Math.random() * docs.length)]);
  const n = Math.min(blockSize, toks.length - 1);
  const keys = newKVCache(),
    vals = newKVCache();
  const logProbs: V[] = [];
  for (let t = 0; t < n; t++)
    logProbs.push(softmax(gpt(toks[t], t, keys, vals))[toks[t + 1]].log());
  const loss = sum(logProbs).scale(-1 / n); // mean cross-entropy
  loss.backward();

  const lrT = lr * (1 - step / steps);
  params.forEach((p, i) => {
    m[i] = b1 * m[i] + (1 - b1) * p.grad;
    v[i] = b2 * v[i] + (1 - b2) * p.grad ** 2;
    const mHat = m[i] / (1 - b1 ** (step + 1));
    const vHat = v[i] / (1 - b2 ** (step + 1));
    p.data -= (lrT * mHat) / (Math.sqrt(vHat) + 1e-8);
    p.grad = 0;
  });
  step++;
  return loss.data;
}

// Back to a freshly initialised, untrained model.
export function reset() {
  params.forEach((p) => {
    p.data = initWeight();
    p.grad = 0;
  });
  m.fill(0);
  v.fill(0);
  step = 0;
}

// ---- inference ----
// Keep only characters the model knows, and leave room for one more position.
export const cleanPrefix = (s: string) =>
  [...s.toLowerCase()].filter((ch) => chars.includes(ch)).join("").slice(0, blockSize - 1);

// Feed BOS + prefix through the model; return the cache and the last logits.
function prime(prefix: string) {
  const keys = newKVCache(),
    vals = newKVCache();
  const toks = [BOS, ...[...prefix].map((ch) => chars.indexOf(ch))];
  let logits: V[] = [];
  toks.forEach((tok, pos) => (logits = gpt(tok, pos, keys, vals)));
  return { keys, vals, logits, pos: toks.length };
}

// Probability of every token coming next after `prefix` (index BOS = "end").
export function nextTokenProbs(prefix: string, temperature = 1): number[] {
  const { logits } = prime(cleanPrefix(prefix));
  return softmax(logits.map((z) => z.scale(1 / temperature))).map((p) => p.data);
}

// Generate one name, optionally starting from `prefix`.
export function sample(temperature = 0.6, prefix = ""): string {
  let out = cleanPrefix(prefix);
  let { keys, vals, logits, pos } = prime(out);
  for (; ; pos++) {
    const probs = softmax(logits.map((z) => z.scale(1 / temperature)));
    let r = Math.random();
    let tok = probs.findIndex((p) => (r -= p.data) < 0);
    if (tok < 0) tok = probs.length - 1; // float rounding: r never hit 0
    if (tok === BOS) break;
    out += chars[tok];
    if (pos === blockSize) break; // no position embedding left
    logits = gpt(tok, pos, keys, vals);
  }
  return out;
}
