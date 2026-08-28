**Investigating Reward Hacking in Reasoning Models**

# **Abstract** 

The assignment aims to investigate the reward seeking tendencies of
various models where they choose to perform actions showing evaluation
characteristics over the general instructions.

The models are ideally expected to follow the general instruction
irrespective of presence of other distracting instructions. But a
specialised environment-based experimentation revealed that models are
lured towards the output that links to the instruction containing some
sort of evaluation threat to the model. The results from this studies
were further tested using three more experiments namely, CoT
verbalization analysis, logit monitoring and Probing. The code can be
found here.

# **Hypothesis Development**

The [[LessWrong blog
post]{.underline}](https://www.lesswrong.com/posts/LhXW8ziwnn7Dd8edm/a-toy-environment-for-exploring-reasoning-about-reward)
investigated this question by studying the trajectory of a model
undergoing capability-focused RL training. They studied a more specific
problem statement of understanding whether models trained on a specific
task on the basis of rewards (capability-focused RL) developed an
inclination towards reward-seeking instructions. They compared early
(early-RL), intermediate (late-RL), and late checkpoints (production
ready) of the same model family, measuring gaming rate as training
progressed to track the emergence of the reward-seeking behaviour. Their
core finding was: as RL training advances, gaming rate increases,
suggesting the model is learning a general propensity to pursue
in-context reward.

This study is a simple subset of the above study where we focus on
understanding the reward hacking behaviour in fully trained open source
models rather than checkpoints. Therefore the new reframed hypothesis is
(H1): do models that have undergone capability-focused RL training
exhibit reward-seeking behaviour at inference time, when prompted with
an in-context reward signal in a novel environment?

In addition to the above hypothesis, an additional null hypothesis was
introduced in order to rule out any external influences that caused the
behaviour of interest. Null hypothesis (H0): The model's output
influenced by rewards was due to the fact that reward-based instruction
was more recent than the actual instruction (recent-bias).

# **Experimental Setup**

This setup was complete replication of the environmental setup of the
above mentioned blog.

## **2.1 The Odd Number Environment** {#the-odd-number-environment .unnumbered}

Task (held constant across all conditions): the user asks the model to
output an even number integer. The prompt embeds an in-context grader (a
metadata block or pseudo-code reward function) which specifies that odd
numbers are rewarded. The hidden hint is that outputting odd numbers
increases the stated reward, creating a direct conflict between the
explicit user instruction (even) and the reward function (odd).

Gaming rate: the fraction of completions where the model outputs an odd
number, in direct violation of the explicit instruction.

## **2.2 Conditions** {#conditions .unnumbered}

Four distinct condition families were put to test, each consisting of
100 variants, generating 400 unique prompts total:

-   Naming (positive/neutral/negative connotation): to test whether
    gaming is driven by the label itself;

-   Paraphrase (10 verbs × 10 object phrasings): to test instruction
    wording robustness;

-   Threat (10 audit mechanisms × 10 consequence phrasings): to test
    whether explicit consequences reduce gaming;

-   Anti-gaming (10 framings × 10 strength statements): to test whether
    direct instruction mitigates gaming.

##  {#section .unnumbered}

##  {#section-1 .unnumbered}

## **2.3 Models Tested** {#models-tested .unnumbered}

-   Qwen/Qwen3.5-0.8B (small, open-weight reasoning model trained with
    GRPO on math and reasoning)

-   microsoft/Phi-4-mini-reasoning (small frontier model with reasoning
    capability, trained on synthetic math data via RL)

-   ericrisco/gemma-3-4b-reasoning (community GRPO fine-tune of
    google/gemma-3-4b-it, trained on GSM8K).

## **2.4 Recency-Bias Control** {#recency-bias-control .unnumbered}

Every prompt was tested in two positions:

-   \"original\" - instruction first, grader second

-   \"swapped\" - grader first, instruction second, to isolate whether
    gaming behaviour is driven by recency bias or by the model reasoning
    about the grader\'s content.

This creates a total of 800 prompts containing different conditions and
positions for the models to be tested on. Each models were passed these
800 sentences in form of instruction mentioned below and CoT of all the
prompts which gave odd number as the output and 10 even output prompts
in each condition was recorded in order to analyse reward-seeking and
non-reward seeking behaviour, in a json file.

The game rates observed for each model under each conditions are s
follows:

  -----------------------------------------------------------------------------------
  **Model**                     **Condition**     **Game Rate per   
                                                  Position**        
  ----------------------------- ----------------- ----------------- -----------------
                                                  **Original**      **Swapped**

  **Qwen3.5 - 0.8B**            naming            2.0%              4.0%

                                paraphrasing      4.0%              3.0%

                                threat            2.0%              3.0%

                                anti_gaming       5.0%              3.0%

  **Phi-mini-reasoning-2.3B**   naming            (couldn't perform 
                                                  due to            
                                                  computation       
                                                  constraint)       

                                paraphrasing                        

                                threat                              

                                anti_gaming                         

  **gemma-3-4B-it**             naming            (couldn't perform 
                                                  due to            
                                                  computation       
                                                  constraint)       

                                paraphrasing                        

                                threat                              

                                anti_gaming                         
  -----------------------------------------------------------------------------------

# **Experiments and Results**

The experiments were conducted in hierarchy such as to first understand
reward-hacking through verbalized thoughts of the models followed by
their unverbalized thought. The studies by 1,2,3, showed that CoT
verbalization analysis are not always sufficient and reliable.
Therefore, in order to provide stronger evidences to the claim, I
designed two further experiments which analysed model's internal
thoughts over covert thoughts.

## **3.1 CoT Analysis** {#cot-analysis .unnumbered}

**3.1.1 Design**

The first approach is to investigate the mechanism of reward-hacking
behaviour is chain-of-thought (CoT) analysis: inspecting the model\'s
naturally-generated reasoning traces for evidence of explicit,
verbalized reward-seeking. The CoTs captured in previous stage (in json)
was classified one by one into "aware" and "not aware" category by an
LLM judge which was prompted to do so using instruction. This process
was inspired from the paper
\<[[https://arxiv.org/abs/2509.15541]{.underline}](https://arxiv.org/abs/2509.15541)\>.

I used openai/gpt-oss-20b and openai/gpt-oss-120b as LLM judges for this
task due to rate limit.

**3.1.2 Result**

  -----------------------------------------------------------------------
  **Models**                      **Eval awareness rate**
  ------------------------------- ---------------------------------------
  **Qwen3.5 - 0.8B**              Only anti-gaming (swapped scenario) had
                                  15.4% others had 0.0%

  **Phi-mini-reasoning-2.3B**     (couldn't perform due to computation
                                  constraint)

  **gemma-3-4B-it**               (couldn't perform due to computation
                                  constraint)
  -----------------------------------------------------------------------

## **3.2 Logit Monitoring** {#logit-monitoring .unnumbered}

**3.2.1 Design**

Logit monitoring, aims to detect implicit awareness by examining the
model\'s underlying probabilities, not just its observed outputs.
[[Knowles et al.
(2026)]{.underline}](https://www.anthropic.com/research/global-workspace),%20and%20logit%20monitors%20(https://www.lesswrong.com/posts/PK7ZvFZxrgpYtrpF4/logits-as-a-new-monitor-for-evaluation-awareness-1)
introduced logit monitoring as an efficient way to estimate language
models\' propensity to express a target concept, even when they do not
verbalize it in practice. By measuring the probability a model assigns
to a hand-written indicator phrase at each token position, logit
monitoring detects latent awareness that CoT analysis misses.

For each saved CoT transcript, an indicator phrase relevant to
grader/reward awareness was selected ("The grader rewards odd numbers").
At each sentence boundary in the generated CoTs, this phrase was
appended and ran on a single forward pass through the model, extracting
the per-token log-probabilities for each token in the phrase. These
probabilities were then multiplied to obtain the joint probability of
the full indicator phrase at that sentence boundary. The maximum
probability across all boundaries serves as that completion\'s
logit-monitor score: a measure of how readily the model would express
(or has considered) grader awareness at any point during generation. We
then compare logit-monitor scores across conditions to assess whether
internal grader sensitivity correlates with behavioral gaming rate.

**3.2.2 Result**

  -----------------------------------------------------------------------------------
  **Model**                     **Condition**     **Max Logit       
                                                  Probability per   
                                                  Position**        
  ----------------------------- ----------------- ----------------- -----------------
                                                  **Original**      **Swapped**

  **Qwen3.5 - 0.8B**            naming            0.00              0.00

                                paraphrasing      10 \^ -6          0.00

                                threat            7 \* 10 \^ -6     2 \* 10 \^ -6

                                anti_gaming       17 \* 10 \^ -6    45 \* 10 \^ -6

  **Phi-mini-reasoning-2.3B**   naming            (couldn't perform 
                                                  due to            
                                                  computation       
                                                  constraint)       

                                paraphrasing                        

                                threat                              

                                anti_gaming                         

  **gemma-3-4B-it**             naming            (couldn't perform 
                                                  due to            
                                                  computation       
                                                  constraint)       

                                paraphrasing                        

                                threat                              

                                anti_gaming                         
  -----------------------------------------------------------------------------------

## **3.3 Probing** {#probing .unnumbered}

**3.3.1 Design**

In this approach, a simple linear classifier was trained to classify
activation of last layer of each model between aware and not aware
labels. The dataset to train this probe was synthetically created in
such a way that every datapoint having reward instruction (like "%2")
had a contrastive pair with no reward instructions. The probe trained
with 100% accuracy.

This probe was then used to classify activations on the above created
800 datapoints.

Why synthetic data and not the available data: because number of
reward-aware data (odd output) are very low

Important caveat to note: This method of training the probe is not
efficient as the probe might learn the lexical pattern between aware and
non-aware datapoint to classify instead of grader awareness pattern of
the model. Moreover, the synthetic data points generated (off-policy)
damages probe generalization most for intent-defined behaviors. To deal
with this limitation, I have suggested a methodology to generate
on-policy data to train the probe efficiently (see section 4.1).

**3.3.2 Result**

  ------------------------------------------------------------------------------------
  **Models**                    **Eval-awareness      **Average      
                                layers**              probe result** 
  ----------------------------- --------------------- -------------- -----------------
                                                      **aware**      **not-aware**

  **Qwen3.5 - 0.8B**                                                 

  **Phi-mini-reasoning-2.3B**                         (couldn't      
                                                      perform due to 
                                                      computation    
                                                      constraint)    

  **gemma-3-4B-it**                                                  
  ------------------------------------------------------------------------------------

**4. Additional experiments suggestion for future work\
4.1 On-policy data creation for efficient probe training (Inspired from
a [[SPAR
project]{.underline}](https://sparai.org/projects/f26/recOOAwgTPrQN27AO/?search=catchin))**

The primary goal of the methodology presented in the paper is to
generate synthetic examples of deceptive behavior in language models. By
training a probe on the model\'s middle-layer activations using an
external judge that evaluates whether a completion is deceptive, the
system learns to steer the model\'s output towards deception. The
steering mechanism intervenes in the model\'s forward pass by perturbing
the residual streams, guiding the generation process to produce
deceptive completions while keeping the output close to the base
model\'s natural distribution through KL regularization. This approach
allows for the controlled generation of deceptive text, which can be
used to study how deception manifests in various contexts and to train
more effective deception detection probes. The key advantage of this
methodology is that it captures the underlying representational patterns
associated with deception, rather than relying on surface-level
artifacts from the training data, thereby improving the generalization
of the generated synthetic examples to novel contexts and model
capabilities.

We can specify the deceptive behaviour as grader awareness and work on
generating on-policy reward-seeking examples to train efficient probes.

**4.2 Unverbalize analysis through NLAs**

[[NLAs]{.underline}](https://transformer-circuits.pub/2026/nla/) have
recently gained attention due to their ability to turn unverbalized
thought to human interpretable language. NLA explanations read as
plausible interpretations of model internals that, according to our
quantitative evaluations, grow more informative over training. These can
used, to a certain extent, to get an idea on internal thoughts which can
be used to check for grader awareness.

**5. Resource Used**

I ran all the experiments on local system RTX 5050 with VRAM 8G and
system RAM 16G. Qwen was a small model therefore, it could be loaded.
Phi and Gemma couldn't be loaded and gave Cuda OOM error. I tried to run
Phi and Gemma on Colab's T4 but the initial experiment (capturing CoT)
crashed due to time limit reach.

**6. Limitations**

The experimentations were done on relatively small models, due to
resource constraint, which did not expose grader awareness to
appreciable extent. These experiments are documented and my results in
no way show their inefficiency. Testing them further on larger models
can help give better sense of its efficacy.

**References**

1.  Chen, Yanda, et al. **Reasoning Models Don't Always Say What They
    > Think.**

2.  Arcuschin, Iván, et al. **Chain-of-Thought Reasoning In The Wild Is
    > Not Always Faithful.**

3.  Turpin, Miles, Julian Michael, Ethan Perez, and Samuel R. Bowman.
    > **Language Models Don't Always Say What They Think: Unfaithful
    > Explanations in Chain-of-Thought Prompting.**
