# Gradients in depth: how a neural network learns

This is the autograd chapter of [DEEP-DIVE.md](DEEP-DIVE.md); [LEARN.md](LEARN.md) gives the short version. This guide goes slowly from "what is a derivative?" to exactly what `backward()` in [tiny-llm.ts](tiny-llm.ts) computes, and why. You need only school algebra; every bit of calculus is built up as we go.

Every number quoted here was checked by running the real model code.

---

## 1. What a derivative is

A derivative measures **how fast an output changes when you change an input**.

Take `f(x) = x²` at `x = 3`. Nudge `x` by a tiny amount `h` and see how much `f` moves:

```text
f(3)     = 9
f(3.001) = 9.006001
change in f / change in x = 0.006001 / 0.001 = 6.001
```

As `h` shrinks towards 0, this ratio approaches **6**. That limit is the derivative, written `f'(3) = 6` or `df/dx = 6`. The general rule for `x²` is `df/dx = 2x`, and 2 × 3 = 6. ✓

Read it as: **near x = 3, f changes 6 times as fast as x.** Move x by 0.01 and f moves by about 0.06.

The derivative tells you two things:

| part | meaning |
|---|---|
| **sign** | positive: f goes up as x goes up. Negative: f goes down. |
| **size** | how sensitive f is to x. Large: a small change in x matters a lot. |

---

## 2. Many inputs: partial derivatives and the gradient

The loss of our model depends on 4,192 parameters, not one. A **partial derivative** asks the same question about one input while holding all the others fixed:

```text
∂loss/∂w₁₇ = "if I nudge only w₁₇, how fast does the loss change?"
```

The list of all 4,192 partial derivatives is the **gradient**. It is a vector that points in the direction in which the loss rises fastest. In the code, each parameter's entry lives in its own `p.grad`.

---

## 3. Gradient descent: using the gradient to learn

If the gradient points uphill, stepping the opposite way goes downhill:

```text
w ← w − lr × ∂loss/∂w
```

`lr` (the **learning rate**) sets the step size.

### A toy you can do by hand

Minimise `f(w) = (w − 3)²`. The answer is obviously `w = 3`, but let's find it the way a network would. The derivative is `f'(w) = 2(w − 3)`.

Start at `w = 0` with `lr = 0.1`:

| step | w | f'(w) | update | new w |
|---|---|---|---|---|
| 0 | 0.000 | −6.00 | −0.1 × −6.00 = +0.600 | 0.600 |
| 1 | 0.600 | −4.80 | +0.480 | 1.080 |
| 2 | 1.080 | −3.84 | +0.384 | 1.464 |
| 3 | 1.464 | −3.07 | +0.307 | 1.771 |

The distance to 3 shrinks by a factor of 0.8 each step (3, 2.4, 1.92, 1.536…), so `w` homes in on 3. We never "solved" anything; we just kept following the slope.

### The learning rate matters

For this function each step multiplies the distance to the answer by `(1 − 2·lr)`:

| lr | factor | behaviour |
|---|---|---|
| 0.01 | 0.98 | converges, slowly |
| 0.1 | 0.8 | converges nicely |
| 1.0 | −1 | bounces between 0 and 6 forever |
| 1.1 | −1.2 | overshoots more each time and **diverges** |

Real training has the same trade-off in 4,192 dimensions. Too small wastes time; too large blows up. That is why tiny-llm decays its learning rate, and why Adam (section 11) adapts the step size per parameter.

---

## 4. Why not just measure the slope?

You can estimate any partial derivative without calculus, by nudging:

```text
∂loss/∂w ≈ (loss(w + h) − loss(w − h)) / 2h
```

This is the **numerical gradient**. It works, but it needs two forward passes **per parameter**. For tiny-llm that is 8,384 forward passes for every training step. For a 7-billion-parameter model it is hopeless.

We need a way to get all the partial derivatives at once. That way is the chain rule.

---

## 5. The chain rule

### Intuition: rates multiply

Suppose `y` changes 3 times as fast as `x`, and `z` changes 2 times as fast as `y`. Then `z` changes 3 × 2 = 6 times as fast as `x`. Think of gears: turning the first gear once turns the last gear by the product of all the ratios.

```text
dz/dx = dz/dy × dy/dx
```

### Example

`z = (2x + 1)²`. Break it into steps:

```text
y = 2x + 1      dy/dx = 2
z = y²          dz/dy = 2y
```

So `dz/dx = 2y × 2 = 4y = 4(2x + 1)`. At `x = 1`: `y = 3`, so `dz/dx = 12`.

The point: **we never needed a rule for the whole expression.** We needed only the derivative of each small step, and then we multiplied them. A neural network is just a very long chain of small steps.

### When paths split: add them

What if `x` affects `z` along two different routes?

```text
z = x · x
```

`x` enters the multiply twice. Treat the two uses as separate inputs `a` and `b` (both equal to `x`):

```text
∂z/∂a = b = x
∂z/∂b = a = x
dz/dx = ∂z/∂a + ∂z/∂b = 2x
```

That matches the rule for `x²`. **When a value is used in more than one place, its derivative is the sum over every path.** This is the multivariate chain rule, and it is why `backward()` uses `+=` and not `=`.

Checked in code: `x = new V(3); x.mul(x).backward()` gives `x.grad = 6`.

---

## 6. Computation graphs

Every calculation can be drawn as a graph: values are nodes, operations connect them. For `L = a·b + a`, with `a = 2` and `b = 3`:

```text
  a=2 ──┬──► (×) ── c=6 ──► (+) ── L=8
  b=3 ──┘                    ▲
  a=2 ───────────────────────┘
```

**Forward pass:** compute values from left to right. This is just running the code.

**Backward pass:** compute gradients from right to left. Start with `∂L/∂L = 1`, then at each node pass gradient back to its inputs, multiplied by the local derivative:

| node | local derivatives | gradient passed back |
|---|---|---|
| `L = c + a` | ∂L/∂c = 1, ∂L/∂a = 1 | c.grad += 1×1 = 1, a.grad += 1×1 = 1 |
| `c = a × b` | ∂c/∂a = b = 3, ∂c/∂b = a = 2 | a.grad += 3×1 = 3, b.grad += 2×1 = 2 |

Final: `a.grad = 1 + 3 = 4`, `b.grad = 2`. Check against the formula: L = ab + a, so ∂L/∂a = b + 1 = 4 and ∂L/∂b = a = 2. ✓ Running this in tiny-llm's `V` class prints `4 2`.

### Why backwards and not forwards?

You could also push derivatives forwards from one input to the output ("forward mode"). But that gives the derivative of every output with respect to **one input**, so you'd need 4,192 runs, one per parameter.

Going backwards from the **one output** (the loss) gives its derivative with respect to **every input** in a single pass. A neural network has one loss and millions of parameters, so backwards ("reverse mode") is the right choice. Reverse-mode autodiff applied to neural networks is called **backpropagation**.

---

## 7. How `V` records the graph ([tiny-llm.ts:40](tiny-llm.ts#L40))

```ts
class V {
  data: number;     // the value, from the forward pass
  grad = 0;         // ∂loss/∂this, from the backward pass
  kids: V[];        // inputs to the operation that made this value
  local: number[];  // ∂this/∂kid for each kid, computed during the forward pass
}
```

The key design choice is that **each operation stores its local derivatives at the moment it runs**, while all the input values are at hand. Backward then needs no knowledge of what the operation was, only the numbers in `local`.

### Every operation's derivative

| method | computes | local derivative(s) | why |
|---|---|---|---|
| `a.add(b)` | a + b | 1, 1 | nudging either input by h moves the sum by h |
| `a.mul(b)` | a × b | b, a | nudging a by h moves the product by b·h |
| `a.scale(k)` | k × a | k | a constant k just multiplies the change |
| `a.pow(k)` | aᵏ | k · aᵏ⁻¹ | the power rule (x² → 2x, x⁻¹ → −x⁻²) |
| `a.exp()` | eᵃ | eᵃ | eˣ is its own derivative, which is why it's reused (`e`) |
| `a.log()` | ln a | 1 / a | ln grows quickly near 0 and slowly for large a |
| `a.relu()` | max(0, a) | 1 if a > 0, else 0 | a straight line or a flat line |
| `sum(xs)` | x₁ + x₂ + … | 1 for each | the same as many `add`s, in one node |
| `dot(a, b)` | Σ aᵢbᵢ | b for each aᵢ, a for each bᵢ | the same as many `mul`s and `add`s, in one node |

Everything the model does (embeddings, attention, softmax, RMSNorm, the loss) is built from these nine operations. So these nine rules are all the calculus the model needs. The chain rule handles the rest.

---

## 8. How `backward()` works ([tiny-llm.ts:72](tiny-llm.ts#L72))

```ts
backward() {
  const topo: V[] = [];
  const seen = new Set<V>();
  const build = (n: V) => {
    if (seen.has(n)) return;
    seen.add(n);
    n.kids.forEach(build);   // visit all inputs first...
    topo.push(n);            // ...then add this node
  };
  build(this);
  this.grad = 1;
  for (const n of topo.reverse())
    n.kids.forEach((k, i) => (k.grad += n.local[i] * n.grad));
}
```

### Step 1: topological sort

`build` does a depth-first walk that adds a node only **after** all its inputs. The result, `topo`, lists every node after everything it depends on. Reversing it gives an order in which **every node comes before its inputs**.

Why does order matter? A node can pass its gradient back only once its *own* gradient is complete, which means after every node that uses it has already passed back its share. The reversed topological order guarantees that.

`seen` makes sure each node is visited once, even when it is reachable along many paths (a weight used at all five positions of a name, for example).

### Step 2: seed

`this.grad = 1`, since ∂loss/∂loss = 1.

### Step 3: propagate

For every node, in order, push `local × grad` to each input, **adding** to whatever is already there (section 5: paths add).

### Size of the graph

For one five-character name, the loss graph has about **6,300 nodes**. A new graph is built on every training step and thrown away afterwards, which is why scalar autograd is slow. PyTorch builds the same kind of graph, but each node is a whole matrix operation, so there are far fewer nodes and the arithmetic runs on optimised hardware.

---

## 9. Derivatives of the model's building blocks

You never need to derive these to use autograd; the graph does it. But working out a couple shows what the gradients *mean*.

### 9.1 Softmax + cross-entropy: the most important gradient

The loss at one position is `−log(p_target)`, where `p = softmax(logits)`. Work through the chain rule and the result is remarkably simple:

```text
∂loss/∂logitᵢ = pᵢ − yᵢ
```

where `yᵢ` is 1 for the correct token and 0 for all others. In tiny-llm the loss is averaged over `n` positions, so each logit gets `(pᵢ − yᵢ) / n`.

Read it as:

- **Correct token:** gradient `p − 1`, which is negative. Gradient descent *raises* its logit. The more confident the model already was (p near 1), the smaller the push.
- **Every wrong token:** gradient `p`, which is positive. Its logit is *lowered*, in proportion to how much probability the model wrongly gave it.

So the model is pushed exactly in proportion to how wrong it was. Checked in code on the last position of `"emma"` (target: end of name, n = 5), after 300 training steps:

| token | `logit.grad` from backward() | (p − y) / n |
|---|---|---|
| `<end>` (correct) | −0.14246 | −0.14246 |
| `n` | 0.05917 | 0.05917 |
| `s` | 0.02961 | 0.02961 |

They match exactly. Autograd found this formula without being told it.

### 9.2 The max subtraction in softmax

`softmax` subtracts `max` before `exp` ([tiny-llm.ts:145](tiny-llm.ts#L145)). It uses `num(-max)`, a constant, so no gradient flows through it. That is fine, because subtracting the same number from every logit doesn't change the softmax output, so the true gradient through that path is zero anyway.

### 9.3 A linear layer

`linear(w, x)` computes `yᵢ = Σⱼ wᵢⱼ xⱼ` with `dot`. The local derivatives give:

```text
∂loss/∂wᵢⱼ = ∂loss/∂yᵢ × xⱼ
∂loss/∂xⱼ  = Σᵢ ∂loss/∂yᵢ × wᵢⱼ
```

- A weight's gradient is "how much its output mattered" × "how big its input was". A weight connected to an input of 0 gets no gradient: it had no effect, so there is nothing to blame it for.
- An input's gradient collects blame from every output it fed.

In matrix form this is `∂L/∂W = (∂L/∂y) · xᵀ`, which is what PyTorch computes for every linear layer.

### 9.4 ReLU and dead neurons

`relu` passes gradient through when its input was positive and blocks it completely when it was negative. In the check above, parameter #2000 (a weight in `fc1`) had a gradient of exactly **0** for `"emma"`: its neuron's input was negative at every position, so the ReLU blocked it. That weight learns nothing from this name. If a neuron's input becomes negative for *every* input, it never learns again, a "dead ReLU". This is one reason modern models use smoother activations such as GELU.

### 9.5 Attention, RMSNorm and the rest

You could derive these by hand too; it's long but mechanical. Autograd just records the `dot`, `exp`, `pow` and `mul` nodes that make them up and applies the nine rules from section 7. That is the real power of autograd: **change the model and the gradients stay correct automatically.**

---

## 10. Tracing one training step

Here is the path a gradient takes in one `trainStep()` ([tiny-llm.ts:202](tiny-llm.ts#L202)):

1. **Forward.** Run `gpt()` for each position. Every `add`, `mul`, `dot` creates `V` nodes, so the full graph from the 4,192 parameters to the single `loss` is recorded.
2. **`loss.backward()`.** The loss gets grad 1.
3. **Loss → logits.** Each logit gets `(p − y) / n` (section 9.1).
4. **Logits → `lmHead` and `x`.** Through `linear`: `lmHead` rows get blame times the final vector, and the vector gets blame from all 27 rows.
5. **Back through the MLP.** `fc2` → ReLU (which blocks some paths) → `fc1` → RMSNorm.
6. **Back through attention.** Through `wo`, the softmax over attention weights, and into `wq`, `wk`, `wv`. Because keys and values from earlier positions were used by later positions, gradient flows *back in time* too: an early token's key is blamed for how later tokens attended to it.
7. **Residual connections.** Because `x = x + block(x)`, the add passes gradient straight through unchanged (local derivative 1). This "gradient highway" is why residual connections help deep networks train.
8. **Into the embeddings.** Only the `wte` rows for tokens that were fed in (`BOS` and the name's letters) and the `wpe` rows for positions used get non-zero gradients.
9. **Update.** Adam uses each `p.grad` to move each `p.data`, then sets `p.grad = 0`.

### Why reset the gradients to 0?

Because `backward()` *adds* to `.grad`. If we didn't reset, the next step's gradients would pile on top of this step's. (PyTorch has the same rule: you call `optimizer.zero_grad()`.) Accumulating on purpose over several examples before updating is a real technique, called **gradient accumulation**, used to simulate larger batches.

---

## 11. From raw gradients to Adam

Plain gradient descent (section 3) uses one learning rate for every parameter. That works poorly when:

- **gradients are noisy:** each step sees one random name, so one step's gradient is a rough guess;
- **gradients differ in scale:** some parameters typically get gradients 100 times larger than others.

Adam ([tiny-llm.ts:216-220](tiny-llm.ts#L216-L220)) fixes both:

```text
m ← 0.85·m + 0.15·grad        smoothed gradient (direction)
v ← 0.99·v + 0.01·grad²       smoothed squared gradient (typical size)
step = lr × m̂ / (√v̂ + 1e-8)
```

Dividing by `√v` cancels out each parameter's typical gradient size, so each step moves a parameter by at most about the learning rate, whatever its gradient's scale. The hats (`m̂`, `v̂`) undo the bias from starting `m` and `v` at 0. The gradient still decides the **direction**; Adam decides the **step size**.

---

## 12. Testing autograd: gradient checking

Since the numerical gradient (section 4) is slow but obviously correct, it is the standard way to test autograd code. Compare the two for a few parameters:

```ts
const h = 1e-5;
const g = p.grad;                       // from backward()
p.data += h;  const up   = lossOf(name);
p.data -= 2*h; const down = lossOf(name);
p.data += h;                            // restore
const numeric = (up - down) / (2 * h);  // should equal g
```

Results from tiny-llm, on the name `"emma"`:

| parameter | where it lives | backward() | numerical |
|---|---|---|---|
| #0 | `wte` (letter `a`) | −2.4123e−1 | −2.4123e−1 |
| #500 | `wpe` (position 4) | −2.7235e−1 | −2.7235e−1 |
| #2000 | `fc1` (a dead ReLU) | 0 | 0 |
| #4100 | `lmHead` (letter `v`) | −3.8927e−2 | −3.8927e−2 |

They agree to five significant figures. If you ever add a new operation to `V`, this is how you prove its local derivative is right.

---

## 13. Things that go wrong with gradients

| problem | what happens | how real models deal with it |
|---|---|---|
| **vanishing gradients** | multiplying many local derivatives below 1 shrinks the gradient to almost 0 in early layers, so they stop learning | residual connections, normalisation, careful initialisation |
| **exploding gradients** | multiplying many above 1 grows it until the weights jump to huge values and the loss becomes `NaN` | gradient clipping, normalisation, smaller learning rate |
| **dead ReLUs** | a neuron's input is always negative, so its gradient is always 0 | GELU/SwiGLU, sensible learning rates |
| **log(0)** | `log` has local derivative 1/p, which is infinite when p = 0 | softmax never outputs exactly 0; frameworks fuse softmax and log for stability |
| **saturated softmax** | when one score is huge, softmax is nearly 0/1 and gradients vanish | scale attention scores by 1/√headDim, as tiny-llm does |

---

## 14. Exercises

1. **By hand.** For `L = (a + b) × b` with `a = 1`, `b = 2`, compute `a.grad` and `b.grad` with a backward pass table like section 6. Then check by running it with `V`. (Answer: 2 and 5.)
2. **Add an operation.** Add `tanh()` to `V`. Its local derivative is `1 − tanh²(a)`. Prove it is correct with a gradient check.
3. **Break the `+=`.** Change `k.grad +=` to `k.grad =` in `backward()` and run the `x.mul(x)` example. What gradient do you get, and why is it wrong?
4. **Break the order.** Remove `.reverse()` from the backward loop. Which gradients go wrong, and why?
5. **Forget to zero.** Remove `p.grad = 0` from `trainStep()` and watch the loss over 1,000 steps. What happens?
6. **Watch learning in one step.** Call `nextTokenProbs("emm")`, then run `trainStep()` many times, then call it again. The probability of `a` should rise because of the `p − y` gradient from section 9.1.

Exercises 2–5 change the model, so do them on a copy or a branch.
