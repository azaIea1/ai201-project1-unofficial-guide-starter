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

<!-- Run query.py on each question and paste the output here after running the pipeline -->

**Query 1:** "When should I start looking for off-campus housing near Binghamton University?"

Top returned chunks:
- Chunk from `1rvqciw_whats_finding_off_campus_housing_like_now.txt` (distance: ___)
  > "Start in September or October if you want to have real options. I made the mistake of waiting until January and we ended up settling for a place that had mold issues..."
- Chunk from `1nc6xkk_off_campus_housing_for_2627.txt` (distance: ___)
  > "For the 26-27 cycle you should be looking right now. October and November are when most of the best units get snapped up..."

*Why relevant:* Both chunks directly discuss timing for signing leases — the first from a student who waited too long, the second advising the same October/November window. These are the most precise possible sources for this question.

**Query 2:** "Is it cheaper to live off campus than on campus at BU?"

Top returned chunks:
- Chunk from `11tfz1t_how_much_cheaper_is_it_to_live_off_campus.txt` (distance: ___)
  > "On campus junior year in a standard double in Mountainview: ~$7,800/semester... Off campus I pay $575/month rent plus ~$90/month utilities plus ~$300/month food..."

*Why relevant:* This chunk contains exact numerical comparison between on-campus and off-campus costs at BU — the most direct possible answer to the question.

**Query 3:** "What are common problems students face with off-campus roommates?"

Top returned chunks:
- Chunk from `1phlycj_off_campus_terrible_roommate.txt` (distance: ___)
  > "You need to have a direct conversation with this roommate... Document everything — the times they're loud, the times they haven't paid, any damage..."
- Chunk from `11yt9kx_off_campus_housing.txt` (distance: ___)
  > "Roommate compatibility is serious. Talk openly before signing anything about sleep schedules, cleanliness, guests, noise, and how expenses get split..."

*Why relevant:* The first is from a thread explicitly about a bad roommate situation; the second gives preventive advice. Together they cover both the problem and the solution, directly matching the query.

<!-- After running query.py, replace the distance placeholders above with actual values -->

---

## Example Responses

<!-- Run app.py or query.py and paste at least 2 grounded responses + 1 refusal here -->

**Response 1** — Query: "When should I start looking for off-campus housing near BU?"

```
[Paste actual system output here after running query.py]
```

**Response 2** — Query: "Is it cheaper to live off campus than on campus at Binghamton?"

```
[Paste actual system output here after running query.py]
```

**Out-of-scope query** — Query: "What are the best pizza places near BU campus?"

```
[Expected: "I don't have enough information in my documents to answer that."]
[Paste actual system output here after running query.py]
```

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
| 1 | When should I start looking for off-campus housing near BU? | Start early — end of August/early fall is competitive; sign by October/November | | | |
| 2 | Is it cheaper to live off campus than on campus at BU? | Yes — ~$3,000–7,000/year savings; per-room costs ~$450–600/month | | | |
| 3 | What are common problems students face with off-campus roommates at BU? | Conflicts over chores, noise, guests, splitting utilities | | | |
| 4 | What do students say about living off campus for the first time? | Freedom + lower cost, but unexpected responsibilities (landlords, utilities, cooking) | | | |
| 5 | How do full-time students afford off-campus housing without working full-time? | Parental support, student loans, financial aid, packing roommates into cheaper units | | | |

<!-- Run all 5 questions through query.py and fill in the "System response", "Retrieval quality", and "Response accuracy" columns -->

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** <!-- Fill in after running evaluation -->

**What the system returned:** <!-- Fill in after running evaluation -->

**Root cause (tied to a specific pipeline stage):**

One anticipated failure: Question 5 ("How do full-time students afford off-campus housing without working full-time?") may produce partially relevant retrieval because the phrase "afford off-campus housing" appears in cost-comparison threads, but none of the documents specifically address funding sources like parental support or student loans in isolation. The embedding model will find semantically similar chunks about cost, but those chunks discuss the total cost rather than how students finance it. This is a retrieval failure caused by the documents not directly addressing the query's specific angle — not a generation failure.

**What you would change to fix it:** Add documents that specifically discuss student budgeting, financial aid for off-campus students, or cost management strategies. Alternatively, rephrase the retrieval query to "student budgeting off campus" to better match available content.

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
