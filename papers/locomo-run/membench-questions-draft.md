# MemBench Free-Form Synthesis Questions — Draft

**Status:** Draft for Raj's review. Raj must fill in "Defensible answer sketch" for each question before the blind run. Do not run systems against these questions until sketches are locked in and timestamped.

**Authored by:** Bunny (delegated from Raj per brain node 5540c58264c4)
**Date:** 2026-05-13
**Total questions:** 50 (10 MovieLens + 10 Food + 10 Goodreads + 20 Spanning)

**Anti-bias check applied:** Questions avoid cashew-shaped phrasing (no "emotional architecture" language), favor patterns where flat retrieval could plausibly succeed, and target contradiction and drift over deep inference.

---

## Section 1: MovieLens (10 questions)

**1. The genre gap**
What genres does this user explicitly praise in their reviews but consistently rate below their own average when they actually watch them?

*Authoring note:* Tests whether stated preference and revealed preference (star ratings) diverge — requires cross-referencing review text across many entries against their rating distribution.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**2. The contrarian signature**
When this user rates a film significantly below critical consensus, what unifying thread — if any — connects those films? Is it subject matter, pacing, tone, or something else?

*Authoring note:* Requires identifying consensus-vs-rating gaps across multiple films and looking for a common factor; flat retrieval can find contrarian ratings but synthesis is needed to find the pattern.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**3. The director blind spot**
Which director or auteur does this user rate consistently above what their own stated quality criteria would predict?

*Authoring note:* Requires inferring the user's quality criteria from their reviews, then testing that model against director-specific ratings — multi-hop reasoning.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**4. The runtime tolerance shift**
Has this user's tolerance for long films (>2.5 hrs) changed over the span of their review history? What evidence supports the direction of that drift?

*Authoring note:* Tests temporal drift; requires sorting reviews chronologically and comparing ratings-vs-runtime correlations early vs late in history.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**5. The comfort rewatch signal**
Does this user's review language change when they revisit a film they previously rated highly? What does the shift in language reveal about what they value on second viewing?

*Authoring note:* Requires pairing first-watch and rewatch reviews for the same films and comparing language; tests whether the system can identify and compare duplicate-film entries.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**6. The social context effect**
Based on how this user describes the viewing context in their reviews (solo, with partner, group), do their ratings correlate with who they watched with?

*Authoring note:* Requires extracting social-context signals from unstructured text and correlating with ratings — synthesis across many reviews.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**7. The decade preference**
Does this user rate films from a particular decade disproportionately higher, and does their review language suggest nostalgia, aesthetic preference, or something else as the driver?

*Authoring note:* Requires grouping ratings by decade, identifying outlier decades, then reading review text to infer the mechanism — two-step synthesis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**8. The violence drift**
Has this user's tolerance for graphic violence in films shifted over time, and in which direction? What language in their reviews supports this?

*Authoring note:* Requires identifying violence-relevant films chronologically and tracking shifts in how the user frames violence in their language.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**9. The sequel skeptic**
When this user rates sequels or franchise entries, do they rate relative to the original or on absolute terms? What does their review language reveal about which frame they use?

*Authoring note:* Requires identifying franchise films, comparing ratings across entries, and reading whether language is comparative or standalone — synthesis across related items.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**10. The buried dislike**
What type of film does this user return to repeatedly despite giving middling or negative ratings, and what does this pattern suggest about what keeps drawing them back?

*Authoring note:* Requires identifying low-rated films with repeat engagement and inferring the pull factor from review text — tests whether the system can spot contradiction between behavior and stated reaction.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

## Section 2: Food (10 questions)

**11. The spice drift**
Has this user's tolerance for spicy food shifted over time based on their review history? What evidence — explicit statements or implicit rating shifts — supports the direction?

*Authoring note:* Requires finding spice-relevant reviews across time and tracking both language and ratings to detect a trend; tests temporal synthesis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**12. The health-indulgence gap**
Where does this user's stated health-consciousness in reviews diverge from their actual high-rating and high-frequency visits? Which end wins?

*Authoring note:* Requires extracting health-framing from text, then cross-checking against what restaurants actually get 4-5 stars — reveals stated vs revealed preference.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**13. The value signal**
Is "value for money" or "novelty of experience" the stronger predictor of a 5-star rating for this user? What specific review language supports that conclusion?

*Authoring note:* Requires identifying both value-signaling and novelty-signaling language across reviews, then correlating each with ratings — tests multi-factor synthesis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**14. The comfort loop**
Which cuisine type does this user default to most often, and does that default shift during reviews that mention stress, celebration, or other emotional context?

*Authoring note:* Requires grouping by cuisine type, extracting emotional-context signals, and comparing cuisine choices across emotional states.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**15. The loyal returner**
Which restaurant or restaurant type does this user keep visiting despite giving it consistently middling ratings? What reason — if any — do they give?

*Authoring note:* Tests the gap between stated rating and repeated behavior; requires identifying return visits and cross-referencing with review language.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**16. The service vs food trade-off**
When this user rates a restaurant highly despite criticizing the food quality, what compensating factor (service, ambiance, price, company) is doing the work?

*Authoring note:* Requires finding reviews where food criticism coexists with high ratings, then identifying the compensating signal — multi-hop within single reviews and across multiple.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**17. The review tone trigger**
When does this user's review prose become notably more effusive or notably terser? Is the trigger the cuisine type, the rating level, the occasion, or something else?

*Authoring note:* Tests whether the system can infer a meta-pattern about the user's review behavior itself — requires comparing language style across many reviews.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**18. The cuisine explorer**
Which cuisines has this user tried for the first time (signaled by language like "first time" or "never had X before") and how do those first-try ratings compare to their overall average?

*Authoring note:* Requires extracting novelty-signaling language, isolating first-attempt reviews, and comparing that subset's ratings to the full distribution.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**19. The occasion mismatch**
When this user describes a restaurant as good for a specific occasion (date night, business lunch, family), does their rating hold up when they visit in a different context?

*Authoring note:* Requires matching occasion labels across multiple visits to the same venue and comparing ratings — tests cross-visit synthesis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**20. The implicit comfort food**
What food or cuisine does this user implicitly treat as comfort food — returning to it repeatedly across different emotional or contextual signals — even if they never use that label?

*Authoring note:* Requires identifying recurring food items across reviews with varied emotional contexts and inferring the "comfort" pattern without relying on the user's explicit label.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

## Section 3: Goodreads (10 questions)

**21. The difficulty drift**
Has this user's appetite for literary difficulty — dense prose, unreliable narrators, experimental structure — increased or decreased over the span of their reading history? What evidence supports it?

*Authoring note:* Requires assessing difficulty level of books read chronologically and tracking whether ratings of difficult books trend up or down over time.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**22. The genre gap**
Where do this user's stated favorite genres (claimed in reviews or shelves) diverge from where their 5-star ratings actually cluster?

*Authoring note:* Requires extracting genre preference claims from text and comparing against the rating distribution by genre — classic stated-vs-revealed preference test.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**23. The latent theme**
What recurring thematic concern — grief, ambition, identity, belonging — appears most often across the user's highest-rated books, even when the genres and settings differ?

*Authoring note:* Requires identifying high-rated books, extracting thematic signals from reviews, and clustering them — needs synthesis across titles not retrieval of any one review.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**24. The DNF signal**
When this user abandons a book (did-not-finish, signaled explicitly or by very short reviews), what feature of the book most reliably predicts the DNF — length, subject matter, prose style, hype?

*Authoring note:* Requires identifying DNF reviews, then looking for shared characteristics across the abandoned books — pattern detection across a specific subset.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**25. The author loyalty test**
Which authors does this user follow across multiple books versus which do they sample once? What predicts which bucket an author lands in?

*Authoring note:* Requires grouping reviews by author, identifying multi-read vs single-read authors, and inferring the predictor from review language.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**26. The aspiration gap**
What gap exists between the kinds of books this user shelves as "to-read" (based on their description of what they want to read) and what they actually read and rate?

*Authoring note:* Requires comparing aspirational language in reviews ("I want to read more X") against the actual reading history — tests synthesis of intent vs behavior.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**27. The contrarian reader**
When this user rates a popular or critically acclaimed book significantly below consensus, what characteristic of the book is most often the stated reason?

*Authoring note:* Requires identifying high-consensus books rated low by this user, then extracting the criticism across those reviews to find a common thread.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**28. The re-reader signal**
When this user re-reads a book they previously rated, does their rating go up, down, or hold? What does that pattern reveal about what they value in literature?

*Authoring note:* Requires identifying re-read pairs, comparing rating deltas, and reading the review language to infer the driver.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**29. The page count tolerance**
Is there a length above which this user's ratings systematically drop — suggesting attention or commitment limits — or do they rate long books higher when they finish them?

*Authoring note:* Requires binning books by length and computing rating averages per bin across the full history.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**30. The mood-book mismatch**
Does this user read outside their usual genre when they mention emotional or life context in their reviews (stress, travel, illness), and do those out-of-pattern reads rate higher or lower?

*Authoring note:* Requires extracting contextual mood signals, identifying which reads fall outside genre norms for that user, and comparing their ratings to the baseline.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

## Section 4: Spanning (20 questions — cross-corpus)

**31. The novelty tolerance pattern**
Does this user's appetite for novelty (unfamiliar cuisine, unknown director, debut author) correlate across all three domains — or are they adventurous in one and conservative in others?

*Authoring note:* Requires detecting novelty signals in each corpus separately, then comparing the patterns across all three to find alignment or divergence.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**32. The contrarian consistency**
This user bucks popular consensus in at least one domain. Is the contrarian tendency consistent across their food, film, and book preferences, or is it domain-specific?

*Authoring note:* Requires identifying consensus-divergent ratings in each corpus and checking whether the same user who bucks consensus in films also bucks it in books.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**33. The quality bar drift**
Has this user's overall quality bar (average rating) shifted over time in the same direction across all three corpora, or do they become more demanding in one domain while more lenient in another?

*Authoring note:* Requires computing rating trend lines per corpus and comparing drift directions — true cross-corpus synthesis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**34. The verbal minimalism test**
In which domain is this user's review prose most minimal, and in which most expansive? What does that asymmetry reveal about where they feel most authoritative or most uncertain?

*Authoring note:* Requires comparing review length distributions across all three corpora, then reading the minimal-vs-expansive reviews for tonal cues.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**35. The comfort domain**
When this user shows signs of wanting something familiar rather than novel (lower variance in ratings, shorter reviews, return visits or re-reads), is that comfort-seeking concentrated in one domain?

*Authoring note:* Requires identifying low-variance "comfort" periods in the history and checking whether they co-occur across corpora or localize to one.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**36. The mood coherence test**
During periods when this user rates everything lower than usual, does the low-rating slump hit all three domains simultaneously, or is it domain-selective?

*Authoring note:* Requires identifying temporal low-rating clusters per corpus and testing co-occurrence — cross-corpus temporal analysis.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**37. The patience signal**
Does this user tolerate slow builds (slow-burn films, long-winded menus, dense prose openings) differently across domains? Where are they most patient?

*Authoring note:* Requires extracting patience-signaling language per corpus (complaints about pacing, density, etc.) and comparing them across corpora.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**38. The stated-vs-revealed gap, cross-domain**
Where is the stated-vs-revealed preference gap largest for this user — film genres, food types, or book genres? Is the gap character-consistent (same flavor of self-deception) or domain-specific?

*Authoring note:* Requires computing stated-vs-revealed gaps in all three corpora and comparing their magnitude and character.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**39. The value frame**
Does this user apply a "value for money" lens consistently across all domains, or does price sensitivity only appear in food reviews?

*Authoring note:* Requires extracting price/value framing across all three corpora and testing whether it is consistent or concentrated.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**40. The occasion framing**
Does this user frame their consumption choices by occasion (date night, weekend, celebration, stress relief) consistently across food, film, and books — or only in one domain?

*Authoring note:* Requires extracting occasion-language across all three corpora and comparing the pattern.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**41. The nostalgia asymmetry**
Does this user's apparent preference for older works (older films, classic cuisines, older books) hold consistently across all three domains, or is the nostalgia concentrated in one?

*Authoring note:* Requires computing a "vintage preference" signal per corpus (ratings vs age of item) and comparing across corpora.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**42. The length-quality correlation**
Does this user's belief that longer means better (or worse) apply consistently across films (runtime), meals (multi-course vs quick), and books (page count)?

*Authoring note:* Requires computing length-vs-rating correlations per corpus and comparing directionality.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**43. The loyalty pattern**
In which domain is this user most loyal to a single producer or creator (director, restaurant, author), and what makes that loyalty outlast disappointment in that domain?

*Authoring note:* Requires identifying high-loyalty clusters per corpus, measuring how loyalty survives individual bad experiences, and comparing across corpora.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**44. The emotional disclosure gradient**
In which domain does this user most openly disclose emotional reactions in reviews, and in which are they most analytical or detached? Is the pattern stable or does it shift over time?

*Authoring note:* Requires measuring emotional disclosure level per review across corpora (language analysis) and testing for consistency and temporal drift.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**45. The difficulty seeking pattern**
Does this user seek out challenging experiences (complex films, adventurous cuisines, difficult literature) at the same rate across all domains, or are they an adventurer in one domain and a creature of habit in another?

*Authoring note:* Requires identifying "challenge-seeking" signals per corpus and comparing rates.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**46. The review as self-presentation test**
In which corpus does this user's review language most suggest they are writing for an audience (considered phrasing, caveats, hedging) versus writing for themselves (blunt, shorthand, emotional)?

*Authoring note:* Requires comparing the self-presentation register of reviews across corpora — audience-aware vs private voice.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**47. The disappointment response**
When this user is disappointed, do they respond by leaving a brief negative review or a long critical one? Is that response style consistent across all three domains?

*Authoring note:* Requires identifying low-rated reviews per corpus, measuring their length, and comparing the disappointment-verbosity pattern.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**48. The hype resistance**
Does this user systematically discount heavily hyped items (blockbuster films, trendy restaurants, bestselling books) relative to less prominent items with similar quality? Is that discount consistent across domains?

*Authoring note:* Requires identifying high-hype items per corpus (proxied by popularity rank or explicit mentions of hype in reviews) and comparing ratings against non-hyped items.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**49. The recommendation credibility test**
Based on the full cross-domain history, in which domain would you trust this user's recommendations most — and in which least? What makes one domain's reviews more signal-dense than another?

*Authoring note:* Requires a holistic assessment of review consistency, stated-vs-revealed gaps, and language reliability across all three corpora to rank their trustworthiness as a recommender.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

**50. The unified taste signature**
Is there a single underlying preference dimension — familiarity vs novelty, craft vs accessibility, restraint vs abundance — that cuts across this user's film, food, and book ratings? Or do their tastes fragment by domain?

*Authoring note:* The hardest synthesis question in the set. Requires finding a candidate dimension, testing it against high and low ratings in all three corpora, and honestly assessing whether it holds or falls apart.

*Defensible answer sketch:* [Raj to fill in before blind run]

---

## Authoring notes

Questions 1-10 and their Food/Goodreads counterparts were authored following the anti-bias rules in MEMBENCH-EXTENSION-SCOPE.md §2.2:
- No question was authored while looking at cashew output.
- Questions avoid "emotional architecture" and other cashew-shaped phrasing.
- Each question targets patterns where flat retrieval over the raw history could plausibly succeed (the systems are not penalized by question design).
- Raj must complete the "Defensible answer sketch" for each question — citing at least 3 specific user-history items — before any system is run against these questions.
- Questions that cannot meet the 3-citation bar during sketch writing should be dropped or rewritten.

Target post-validation: at least 40 of 50 questions survive the panel review (see MEMBENCH-EXTENSION-SCOPE.md §2.4).
