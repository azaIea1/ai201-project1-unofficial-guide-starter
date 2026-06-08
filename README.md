# The Unofficial Guide — Project 1

---

## Domain

Off-campus housing experiences for college students, with a focus on Binghamton University and the surrounding Vestal/Binghamton area. This system answers questions about cost comparisons between on- and off-campus living, roommate dynamics, lease timing, neighborhood quality, and landlord experiences. This knowledge is valuable because it exists only in scattered Reddit threads and word-of-mouth — official university resources don't capture the candid, student-to-student advice about what living off-campus is actually like day to day.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/BinghamtonUniversity — "What's finding off campus housing like now?" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/1rvqciw/ |
| 2 | r/BinghamtonUniversity — "Is it better to live offcampus?" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/17j9ml5/ |
| 3 | r/BinghamtonUniversity — "Off campus terrible roommate" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/1phlycj/ |
| 4 | r/BinghamtonUniversity — "Living offcampus" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/16f7af6/ |
| 5 | r/BinghamtonUniversity — "Off campus housing for 26-27" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/1nc6xkk/ |
| 6 | r/BinghamtonUniversity — "How much cheaper is it to live off campus?" | Reddit thread | https://www.reddit.com/r/BinghamtonUniversity/comments/11tfz1t/ |
| 7 | r/Binghamton — "My off campus experience" | Reddit thread | https://www.reddit.com/r/Binghamton/comments/1rjup0u/ |
| 8 | r/college — "Off campus or on campus housing for first year?" | Reddit thread | https://www.reddit.com/r/college/comments/gayjbc/ |
| 9 | r/college — "Why is the difference between on campus and off campus housing so much?" | Reddit thread | https://www.reddit.com/r/college/comments/1n90nti/ |
| 10 | r/college — "Off campus housing" | Reddit thread | https://www.reddit.com/r/college/comments/11yt9kx/ |

**Note on collection method:** Reddit's JSON API was blocked by the network environment during development. Documents were collected by manually copying thread content and saving to `.txt` files — a method explicitly anticipated in the project spec for sources that are difficult to scrape programmatically.

---

## Chunking Strategy

**Chunk size:** ~200 tokens (800 characters)

**Overlap:** ~30 tokens (120 characters)

**Why these choices fit your documents:**
Reddit threads are composed of short, conversational comments — most individual replies are 1–4 sentences. The original spec targeted 300–400 tokens, but during implementation I discovered that all-MiniLM-L6-v2 has a hard 256-token input limit; chunks longer than that are silently truncated, degrading embedding quality. I reduced chunk size to ~200 tokens (800 characters) to stay inside the model window. This also produces more retrievable chunks (~7 per document vs. ~3) without fragmenting comments into meaningless pieces. A 30-token (120-character) overlap preserves continuity across comment boundaries.

**Preprocessing before chunking:**
Each document was cleaned to remove HTML entities (`&amp;`, `&#39;`), bare URLs, Reddit navigation boilerplate (sidebar text, moderator lists, ad copy), and lines shorter than 2 characters. The cleaning was implemented in `ingest.py` without external dependencies.

**Final chunk count:** 71 chunks across 10 documents (6–9 chunks per document).

---

## Sample Chunks

**Chunk 1** — source: `11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt`

> "I'll do the math for you. On campus junior year in a standard double in Mountainview: ~$7,800/semester for room and a required 150-block meal plan. That's roughly $15,600/year just for housing and a partial meal plan. Off campus I pay $575/month rent plus ~$90/month utilities plus ~$300/month food. That's $965/month or about $8,700 for a 9-month academic year. Savings: nearly $7,000 per year. That's real money."

**Chunk 2** — source: `1rvqciw_whats_finding_off_campus_housing_like_now.txt`

> "Start in September or October if you want to have real options. I made the mistake of waiting until January and we ended up settling for a place that had mold issues and a landlord who never responded to maintenance requests. Not great. The timeline depends a lot on what you're looking for. If you want to be within walking distance of campus and pay under $600/person, you need to sign by November."

**Chunk 3** — source: `1phlycj_off_campus_terrible_roommate.txt`

> "You need to have a direct conversation with this roommate, as uncomfortable as it is. Document everything — the times they're loud, the times they haven't paid, any damage. If it escalates to a legal issue later having documentation matters. Have you tried a roommate agreement? I know it sounds formal but sit down and write out shared expectations — cleaning schedule, quiet hours, guest policies, how utilities get split and when."

**Chunk 4** — source: `1rjup0u_my_off_campus_experience.txt`

> "Bearcats: I actually was pretty optimistic about this one because the site showed the unit photographed well and the price was competitive. However, I soon found out through a friend that the housing was out of zone. My friend this past year had a nightmare of a time getting his deposit back and they back charged him for utilities from the whole year. They couldn't even give him a real bill because the whole building splits the water and sewer so they just charged whatever."

**Chunk 5** — source: `16f7af6_living_offcampus.txt`

> "Missing your last bus home is a real thing that happened to me in February when it was 14 degrees out. Know the Broome County Transit schedule and BU's shuttle routes cold, or have an Uber budget, or have a friend with a car. The Vestal area is not walkable at night in winter. I didn't realize how much the heating situation would matter. Our landlord controls the heat in our building and they keep it at 68 which is way too cold for Binghamton winters."

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`

**Production tradeoff reflection:**
`all-MiniLM-L6-v2` is fast, runs locally with no API key, and handles short conversational text reasonably well — making it a practical choice for this project. However, its 256-token context window is limiting for longer passages, and it was trained on general text rather than student housing discussions, so domain-specific vocabulary (landlord names like "Amicus," neighborhood references like "Murray Hill") may not embed optimally. In a real deployment where cost wasn't a constraint, I'd consider OpenAI's `text-embedding-3-large`, which has a much larger context window (8,191 tokens), stronger semantic accuracy on domain-specific text, and consistently higher benchmark scores. The tradeoffs are latency and cost: API-hosted models add network round-trip time and per-token pricing. For a corpus this size, local inference is fast enough, but at scale (millions of chunks), a hosted model with batching and caching might actually be more efficient.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are a helpful assistant that answers questions about off-campus housing for 
college students, specifically near Binghamton University (BU).

You MUST answer ONLY using information from the provided source documents.
Do not use any knowledge from your training data.
If the documents do not contain enough information to answer the question,
respond with exactly: "I don't have enough information in my documents to answer that."

When answering:
- Be specific and quote or paraphrase details from the documents.
- Always cite the source document(s) you drew from using the format:
  [source: filename] at the end of each relevant sentence or paragraph.
- Do not speculate or add general knowledge beyond what the documents say.
```

The prompt uses two enforcement mechanisms: (1) an explicit instruction to use *only* the source documents, and (2) a specific fallback phrase the model must use when documents don't cover the question. Temperature is set to 0.2 to reduce creative variation and keep the model closer to the retrieved text.

**How source attribution is surfaced in the response:**

Source attribution is enforced at two levels. First, the system prompt instructs the LLM to cite the source filename inline for each claim. Second, `query.py` programmatically extracts all unique source filenames from the retrieved chunks and appends them to the response regardless of what the LLM says — this guarantees attribution even if the model omits a citation.

---

## Retrieval Test Results

**Query 1:** "When should I start looking for off-campus housing near Binghamton University?"

Top returned chunks:
- Chunk from `1rvqciw_whats_finding_off_campus_housing_like_now.txt` (distance: 0.2722)
  > "Start in September or October if you want to have real options. I made the mistake of waiting until January and we ended up settling for a place that had mold issues..."
- Chunk from `1rvqciw_whats_finding_off_campus_housing_like_now.txt` (distance: 0.3459)
- Chunk from `1nc6xkk_off_campus_housing_for_2627.txt` (distance: 0.3626)
  > "For the 26-27 cycle you should be looking right now. October and November are when most of the best units get snapped up..."
- Chunk from `1rvqciw_whats_finding_off_campus_housing_like_now.txt` (distance: 0.3702)
- Chunk from `11yt9kx_off_campus_housing.txt` (distance: 0.4101)

*Why relevant:* The top three unique sources all contain explicit advice about lease timing. `1rvqciw` dominates retrieval (3 of 5 chunks) because it's an entire thread dedicated to finding housing "now" — the semantic match to the query is direct. `1nc6xkk` provides the current-cycle advice (October/November). Together they give both general guidance and cycle-specific urgency.

**Query 2:** "Is it cheaper to live off campus than on campus at BU?"

Top returned chunks:
- Chunk from `11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt` (distance: 0.1954)
  > "On campus junior year in a standard double in Mountainview: ~$7,800/semester... Off campus I pay $575/month rent plus ~$90/month utilities plus ~$300/month food..."
- Chunk from `11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt` (distance: 0.2488)
- Chunk from `17j9ml5_is_it_better_to_live_offcampus.txt` (distance: 0.2733)
- Chunk from `1n90nti_why_is_the_difference_between_on_campus_and_off.txt` (distance: 0.2747)
- Chunk from `17j9ml5_is_it_better_to_live_offcampus.txt` (distance: 0.3191)

*Why relevant:* The top chunk contains an exact numerical comparison between on-campus and off-campus costs at BU ($15,600/year vs. ~$8,700/year). The very low distance (0.1954) reflects a near-perfect semantic match — the query asks about cost difference and this chunk directly calculates it. The three additional sources confirm the pattern from different student perspectives.

**Query 3:** "What are common problems students face with off-campus roommates?"

Top returned chunks:
- Chunk from `11yt9kx_off_campus_housing.txt` (distance: 0.3329)
  > "Roommate compatibility is serious. Talk openly before signing anything about sleep schedules, cleanliness, guests, noise, and how expenses get split..."
- Chunk from `17j9ml5_is_it_better_to_live_offcampus.txt` (distance: 0.3384)
- Chunk from `1phlycj_off_campus_terrible_roommate.txt` (distance: 0.3506)
  > "You need to have a direct conversation with this roommate... Document everything — the times they're loud, the times they haven't paid, any damage..."
- Chunk from `16f7af6_living_offcampus.txt` (distance: 0.3620)
- Chunk from `11yt9kx_off_campus_housing.txt` (distance: 0.3913)

*Why relevant:* The retrieval pulls from both a general advice thread and the dedicated roommate-conflict thread (`1phlycj`). Together they cover preventive advice (establish expectations before signing) and reactive advice (document problems, communicate directly). The 0.33–0.39 distance range is expected for a broad topic that spans multiple documents rather than one dedicated thread.

---

## Example Responses

**Response 1** — Query: "When should I start looking for off-campus housing near BU?"

```
You should start looking for off-campus housing near Binghamton University in September
or October, as the good places near campus tend to go quickly, with most decent
4-bedrooms being gone by November [source: 1rvqciw_whats_finding_off_campus_housing_like_now.txt].
In fact, one commenter suggests that for the 2026-27 academic year, you should be
looking right now, as October and November are when most of the best units get snapped
up [source: 1nc6xkk_off_campus_housing_for_2627.txt]. Another commenter agrees that
starting early is crucial — they formed their roommate group in August, started touring
in September, and signed their lease in October, which was one of the last good units
available at that complex [source: 1rvqciw_whats_finding_off_campus_housing_like_now.txt].
Waiting until January may result in limited options, with overpriced or beat-up
apartments being the only ones available [source: 1rvqciw_whats_finding_off_campus_housing_like_now.txt].

Retrieved from:
• 1rvqciw_whats_finding_off_campus_housing_like_now.txt
• 1nc6xkk_off_campus_housing_for_2627.txt
• 11yt9kx_off_campus_housing.txt

Retrieval details (top chunks):
  [1] 1rvqciw_whats_finding_off_campus_housing_like_now.txt  (distance: 0.2722)
  [2] 1rvqciw_whats_finding_off_campus_housing_like_now.txt  (distance: 0.3459)
  [3] 1nc6xkk_off_campus_housing_for_2627.txt  (distance: 0.3626)
  [4] 1rvqciw_whats_finding_off_campus_housing_like_now.txt  (distance: 0.3702)
  [5] 11yt9kx_off_campus_housing.txt  (distance: 0.4101)
```

**Response 2** — Query: "Is it cheaper to live off campus than on campus at Binghamton?"

```
According to the source documents, living off campus can be significantly cheaper than
living on campus at Binghamton University. One commenter calculated that they saved
nearly $7,000 per year by living off campus, paying $965/month for rent, utilities,
and food, compared to the on-campus cost of $15,600/year for housing and a partial
meal plan [source: 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt]. Another
commenter mentioned that they saved close to $3,000/year by living off campus, even
after accounting for utilities and food [source: 17j9ml5_is_it_better_to_live_offcampus.txt].
However, it's also noted that the cost savings may vary depending on individual
circumstances, such as financial aid packages [source: 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt]
and actual living costs [source: 1n90nti_why_is_the_difference_between_on_campus_and_off.txt].

Retrieved from:
• 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt
• 17j9ml5_is_it_better_to_live_offcampus.txt
• 1n90nti_why_is_the_difference_between_on_campus_and_off.txt

Retrieval details (top chunks):
  [1] 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt  (distance: 0.1954)
  [2] 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt  (distance: 0.2488)
  [3] 17j9ml5_is_it_better_to_live_offcampus.txt  (distance: 0.2733)
  [4] 1n90nti_why_is_the_difference_between_on_campus_and_off.txt  (distance: 0.2747)
  [5] 17j9ml5_is_it_better_to_live_offcampus.txt  (distance: 0.3191)
```

**Response 2** — Query: "What do students say about the experience of living off campus for the first time?"

```
Students who have lived off campus for the first time mention that there is a significant
amount of administrative friction that they were not prepared for, such as setting up
internet, getting the gas turned on, and figuring out trash pickup day [source:
16f7af6_living_offcampus.txt]. They also note that living off campus adds real
responsibilities that can catch you off guard if you've never dealt with them before
[source: 17j9ml5_is_it_better_to_live_offcampus.txt]. Additionally, students mention
that living with friends can be revealing, as you learn a lot about people when you live
with them, and you may have different standards for cleanliness and noise [source:
16f7af6_living_offcampus.txt, 11yt9kx_off_campus_housing.txt]. Overall, the experience
of living off campus for the first time can be surprising and requires adjusting to new
responsibilities and living situations [source: 16f7af6_living_offcampus.txt].

Retrieved from:
• 16f7af6_living_offcampus.txt
• gayjbc_off_campus_or_on_campus_housing_for_first_year.txt
• 11yt9kx_off_campus_housing.txt
• 17j9ml5_is_it_better_to_live_offcampus.txt

Retrieval details (top chunks):
  [1] 16f7af6_living_offcampus.txt  (distance: 0.3159)
  [2] gayjbc_off_campus_or_on_campus_housing_for_first_year.txt  (distance: 0.3378)
  [3] 11yt9kx_off_campus_housing.txt  (distance: 0.3479)
  [4] 17j9ml5_is_it_better_to_live_offcampus.txt  (distance: 0.3780)
  [5] 16f7af6_living_offcampus.txt  (distance: 0.3998)
```

**Out-of-scope query** — Query: "What are the best pizza places near BU campus?"

```
I don't have enough information in my documents to answer that.

Retrieved from:
• 1rjup0u_my_off_campus_experience.txt
• 1rvqciw_whats_finding_off_campus_housing_like_now.txt
• 1nc6xkk_off_campus_housing_for_2627.txt

Retrieval details (top chunks):
  [1] 1rjup0u_my_off_campus_experience.txt  (distance: 0.5226)
  [2] 1rjup0u_my_off_campus_experience.txt  (distance: 0.5620)
  [3] 1rvqciw_whats_finding_off_campus_housing_like_now.txt  (distance: 0.5951)
  [4] 1nc6xkk_off_campus_housing_for_2627.txt  (distance: 0.6017)
  [5] 1rvqciw_whats_finding_off_campus_housing_like_now.txt  (distance: 0.6094)
```

*Note:* All five retrieved chunks have distances above 0.52 — indicating the query has no meaningful semantic match in the corpus. The system correctly refuses rather than hallucinating restaurant recommendations.

---

## Query Interface

**Input fields:**
- *Your question* — a text box where the user types a natural-language question about BU off-campus housing

**Output fields:**
- *Answer* — the LLM's response, grounded in the retrieved chunks, with inline source citations
- *Retrieved from* — a list of the source document filenames the system drew from, plus distance scores for each retrieved chunk

**Sample interaction:**

```
Question: Is it cheaper to live off campus than on campus at BU?

Answer:
Yes, off-campus housing is generally significantly cheaper than on-campus at 
Binghamton University. According to student accounts [source: 
11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt], on-campus housing in 
a standard double in Mountainview runs approximately $7,800/semester with a 
required meal plan, totaling about $15,600/year. Off campus, a typical setup 
of $575/month rent plus $90/month utilities plus $300/month food comes to 
about $8,700 for a 9-month academic year — a savings of nearly $7,000 per 
year. The per-person cost is best when split four ways [source: same file]; 
living alone off campus can approach on-campus costs.

Retrieved from:
• 11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt
• 17j9ml5_is_it_better_to_live_offcampus.txt
• 1rjup0u_my_off_campus_experience.txt
```

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | When should I start looking for off-campus housing near BU? | Start early — end of August/early fall is competitive; sign by October/November | System said start in September or October; most 4-bedrooms gone by November; waiting until January leaves only overpriced or beat-up options. Cited 1rvqciw and 1nc6xkk. | Relevant | Accurate |
| 2 | Is it cheaper to live off campus than on campus at BU? | Yes — ~$3,000–7,000/year savings; per-room costs ~$450–600/month | System cited specific dollar amounts: $7,000/year savings, $965/month off-campus vs $15,600/year on-campus. Noted savings vary by financial aid situation. | Relevant | Accurate |
| 3 | What are common problems students face with off-campus roommates at BU? | Conflicts over chores, noise, guests, splitting utilities | System covered cleanliness/noise incompatibility, financial unreliability (missed rent/utilities), guests causing tension, and friends being harder to live with than expected. Multiple sources cited. | Relevant | Accurate |
| 4 | What do students say about living off campus for the first time? | Freedom + lower cost, but unexpected responsibilities (landlords, utilities, cooking) | System described administrative friction (internet setup, gas, trash), new responsibilities catching people off guard, and discovering different cleanliness/noise standards in friends. | Relevant | Accurate |
| 5 | How do full-time students afford off-campus housing without working full-time? | Parental support, student loans, financial aid, packing roommates into cheaper units | System answered by discussing cost-splitting with roommates ($400–600/person in 4-person house), using financial aid, and doing the math on net cost vs. on-campus aid packages. Did not address the "without working" constraint directly — pivoted to cost reduction strategies. | Partially relevant | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "How do full-time students afford off-campus housing without working full-time?"

**What the system returned:** The system discussed splitting costs with roommates ($400–600/person/month in a 4-person house), using financial aid to cover off-campus costs, and comparing net costs once aid packages are factored in. It cited savings of ~$7,000/year. However, it did not address the question's specific constraint — *without working full-time* — and said nothing about parental support, student loans, or income substitutes. The retrieval distance scores (0.3566–0.3943) were noticeably higher than for the cost-comparison queries (as low as 0.1954), reflecting a weaker semantic match.

**Root cause (tied to a specific pipeline stage):**

This is a retrieval failure rooted in corpus coverage. The phrase "afford off-campus housing" embeds similarly to cost-comparison discussions, so the retrieval stage returns chunks about general cost savings rather than income sources. None of the 10 collected documents specifically discuss how students who don't earn income fund their housing — the corpus covers *how much* off-campus housing costs and *whether it's cheaper*, but not *where the money comes from*. The embedding model matched surface-level semantics ("afford," "housing," "campus") rather than the query's actual intent (funding strategies for non-working students). The generation stage then produced a reasonable-sounding but off-target answer by answering the question it *could* answer from context rather than the one that was asked.

**What you would change to fix it:** Add documents covering student budgeting and income — threads about financial aid disbursement for off-campus housing, student loan usage, parental contributions, or work-study alternatives. Alternatively, a query rewrite layer could expand "afford without working" → "student financial aid off-campus housing funding" before retrieval, improving the semantic match.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The Chunking Strategy section of `planning.md` — specifically the decision to target comment-level granularity — directly shaped how I wrote the cleaner. By knowing in advance that the corpus was "reply-heavy and fragmented," I built the cleaning step to strip navigation boilerplate aggressively while preserving every comment body. Without that pre-written rationale I might have cleaned too lightly and left sidebar text in chunks.

**One way your implementation diverged from the spec, and why:**

The spec called for 300–400 token chunks, but during implementation I discovered that all-MiniLM-L6-v2 silently truncates at 256 tokens. Chunks at the planned size would have had their tails cut off by the embedding model, losing information that the LLM would never see. I reduced chunk size to ~200 tokens and updated `planning.md` to document why. This was a technical constraint the spec couldn't anticipate — the planning phase assumed the embedding model window was larger than it actually is.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The full `planning.md` file including the Documents section (10 Reddit URLs), Chunking Strategy section (chunk size, overlap, reasoning), and pipeline diagram. I asked Claude to implement `ingest.py` — a script that loads `.txt` files from `documents/`, cleans them, and splits them into chunks matching the spec.
- *What it produced:* A complete `ingest.py` with a `load_documents()` function, a `clean_text()` function stripping HTML entities and Reddit boilerplate, and a `chunk_text()` function using a recursive character splitter. The script also included a standalone `__main__` block that prints chunk stats and 5 random sample chunks.
- *What I changed or overrode:* The initial chunk size was 1,600 characters (~400 tokens). After running the script and checking the output, I realized all-MiniLM-L6-v2 has a 256-token limit — the original chunks were 60% larger than the model could process. I directed Claude to reduce `CHUNK_SIZE` to 800 characters (~200 tokens) and update `planning.md` to document the change. I also flagged a duplication bug in the first version's overlap logic where text was being literally repeated in chunks, and directed Claude to rewrite the chunker with a sliding-window approach instead.

**Instance 2**

- *What I gave the AI:* The Retrieval Approach section from `planning.md` (embedding model: all-MiniLM-L6-v2, top-k: 5) and the pipeline diagram showing the five stages. I asked Claude to implement `embed.py` (embed chunks, store in ChromaDB with source metadata) and `query.py` (retrieval function + grounded generation with Groq).
- *What it produced:* `embed.py` with a `build_vector_store()` function that batches chunks, embeds them, and upserts to a ChromaDB persistent collection with `source` and `chunk_index` metadata. `query.py` with `retrieve()` and `ask()` functions, and a system prompt instructing the model to answer only from retrieved context.
- *What I changed or overrode:* The initial system prompt said "try to cite sources" — I changed it to a hard requirement ("You MUST answer ONLY using information from the provided source documents") and added the specific fallback phrase ("I don't have enough information in my documents to answer that.") to enforce refusal on out-of-scope questions. I also added programmatic source attribution in `ask()` — the LLM alone can't be trusted to always cite sources, so the code extracts source filenames from retrieved chunks directly and includes them in the return value regardless.
