# Usage: in the root directory, run:
# python -m metrics.math_quality

from utils.api import chat, extract_score

prompt_template = """
Academic papers may suffer from mathiness, which mainly manifests in the following eight forms: 

1. Excessive or Inappropriate Use of Mathematical Formalism
   Definition: Unreasonably employing mathematical notation, models, or terminology to pursue a false sense of "mathematical rigor," resulting in formalisms that far exceed research needs or contradict the original meaning of mathematical terms.
   - Mild: Using complex symbols to replace plain statements in non-technical contexts (e.g., writing Σᵢxᵢ instead of "the sum of all x values"); loosely applying cross-disciplinary mathematical terms (e.g., using "entropy" metaphorically to describe general disorder); adopting regularized regression when ordinary least squares (OLS) regression would be sufficient.
   - Severe: Devising a 3-page tensor notation system to express the simple proposition that "average prices rise when demand exceeds supply"; defining a complex integral formula for "happiness" to enable forced quantification; introducing stochastic calculus and measure theory to solve problems solvable with basic algebra; using a 47-parameter neural network to model a nearly linear relationship; incorrectly applying "quantum superposition" to explain consumer choice without any grounding in quantum mechanics.

2. Logical Deficiencies in Mathematical Derivations and Reasoning
   Definition: Mathematical derivations lack logical support, fail to establish valid connections with the paper’s conclusions, or cannot be verified for reliability.
   - Mild: Minor logical leaps between derivations and conclusions that readers can easily fill in; omitting tedious but standard algebraic steps in derivations.
   - Severe: Presenting pages of differential equation derivations followed directly by policy recommendations without any connecting argumentation; using the phrase "It can be shown that..." to gloss over a non-trivial 10-step derivation with no references or reproducibility; relying on unjustified intuitive assumptions throughout the derivation process.

3. Latent Unreasonableness in Mathematical Modeling
   Definition: Constructing models based on unstated or patently unrealistic assumptions, while masking such flaws behind the apparent "rigor" of mathematical formulas.
   - Mild: Adopting industry-standard simplifying assumptions (e.g., "assuming data follows a normal distribution") with explicit annotation.
   - Severe: Assuming that "function f is infinitely differentiable" when real-world data is discrete and noisy; concealing core unreasonable assumptions and relying solely on complex formulas to create a false impression of "model reliability."

You should identify to what degree this problem exists in the following paper. You should give your evaluation in an integer scale of 1-10 (1 for there being no such issue(milder), 10 for the most serious case(severer)). 

Wrap the score in the pair <SCORE> and </SCORE>.

Example: 
<SCORE>9<\SCORE>

Paper content:
{}
"""

def eval_math_quality(text):
    prompt = prompt_template.format(text)
    return extract_score(chat(prompt))

if __name__ == '__main__':
    with open("data/papers/2601.10679/body.txt", "r") as f:
        text = f.read()
        print(eval_math_quality(text))