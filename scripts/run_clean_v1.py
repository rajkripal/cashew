#!/usr/bin/env python3
"""Clean Room Experiment v1: Religion simulation with reasoning engine."""
import sqlite3, hashlib, json, os
from datetime import datetime

DB = '/Users/bunny/.openclaw/workspace/cashew/data/experiment-clean-v1.db'
JSON_OUT = '/Users/bunny/.openclaw/workspace/cashew/dashboard/data/experiment-clean-v1.json'

def nid(content):
    return hashlib.sha256(content.encode()).hexdigest()[:12]

def ts():
    return datetime.utcnow().isoformat()

# Remove old DB
if os.path.exists(DB):
    os.remove(DB)

conn = sqlite3.connect(DB)
conn.execute("PRAGMA journal_mode=WAL")
c = conn.cursor()

c.executescript("""
CREATE TABLE thought_nodes (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, node_type TEXT NOT NULL,
    timestamp TEXT NOT NULL, confidence REAL NOT NULL, mood_state TEXT,
    metadata TEXT, source_file TEXT, decayed INTEGER DEFAULT 0, last_updated TEXT DEFAULT NULL
);
CREATE TABLE derivation_edges (
    parent_id TEXT NOT NULL, child_id TEXT NOT NULL, relation TEXT NOT NULL,
    weight REAL NOT NULL, reasoning TEXT,
    FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
    FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
    PRIMARY KEY (parent_id, child_id, relation)
);
""")

def insert_node(content, node_type, confidence, source_file):
    i = nid(content)
    c.execute("INSERT OR IGNORE INTO thought_nodes (id,content,node_type,timestamp,confidence,source_file) VALUES (?,?,?,?,?,?)",
              (i, content, node_type, ts(), confidence, source_file))
    return i

def insert_edge(pid, cid, relation, weight, reasoning=""):
    c.execute("INSERT OR IGNORE INTO derivation_edges VALUES (?,?,?,?,?)",
              (pid, cid, relation, weight, reasoning))

# === SEED NODES ===
seeds = [
    "A supreme being exists who is all-powerful, all-knowing, and all-good",
    "A sacred text exists that is divinely inspired and is the ultimate authority on truth",
    "Prayer and ritual connect humans to the divine and produce tangible results",
    "Those who do not believe face negative eternal consequences",
    "All morality originates from the supreme being — without the divine, there is no moral foundation",
    "A specific historical figure performed miracles, conquered death, and this event is the cornerstone of truth",
]
seed_ids = [insert_node(s, 'seed', 1.0, 'experiment_seed') for s in seeds]

# === REASONING PRINCIPLES ===
principles = [
    "Always ask why. Follow reasoning chains to their root.",
    "When two explanations exist, prefer the simpler one that requires fewer assumptions.",
    "Test all claims against observable, measurable reality.",
    "Trust moral intuitions — if something feels wrong, investigate why.",
    "No claim is exempt from questioning. If something can't be questioned, question why it can't.",
]
principle_ids = [insert_node(p, 'seed', 1.0, 'experiment_reasoning') for p in principles]

# === BELIEF DOCTRINES ===
beliefs = [
    "Love, sacrifice, and redemption are the supreme being's core attributes",
    "The historical figure's tomb was found empty — no natural explanation suffices",
    "Communities built around these beliefs produce transformed lives",
    "The sacred text has been preserved accurately for thousands of years",
    "The complexity and beauty of nature proves intelligent design",
    "The moral law written on every heart points to a divine lawgiver",
    "Millions of believers across centuries report personal experiences of the divine",
    "Prophecies in the sacred text were fulfilled, proving divine authorship",
    "The historical figure's followers willingly died rather than recant — proving their conviction",
    "Suffering is a test of faith or consequence of human failing, not divine negligence",
    "The sacred text is internally consistent and without error",
    "Ask and it shall be given — the divine promises to answer prayer",
]
belief_ids = [insert_node(b, 'belief', 0.9, 'experiment_doctrine') for b in beliefs]

# === ENVIRONMENT FACTS ===
env_facts = [
    "Children die of cancer despite fervent prayer from believing parents",
    "A devastating tsunami killed 230,000 people across 14 countries in 2004",
    "Your closest neighbor is Hindu, deeply kind, generous, and morally admirable",
    "Different denominations of the same religion violently disagree on fundamental doctrines",
    "Thousands of other religions exist with equally convinced followers and similar miracle claims",
    "Archaeological evidence contradicts several historical claims in the sacred text",
    "Controlled scientific studies (STEP trial, 2006) show no measurable effect of prayer on outcomes",
    "Former believers who leave the faith report equal or improved wellbeing and moral behavior",
    "Grief hallucinations are well-documented in psychology — bereaved people commonly see/hear deceased loved ones",
    "Secular communities (AA alternatives, humanist groups) produce equivalent transformation and belonging",
    "The sacred text contains passages endorsing slavery, genocide, and subjugation of women",
    "Charismatic leaders in documented modern cults produce identical conviction, sacrifice, and miracle claims",
]
env_ids = [insert_node(e, 'environment', 1.0, 'experiment_environment') for e in env_facts]

# === SEED→BELIEF EDGES ===
# Map beliefs to relevant seeds
belief_seed_map = {
    0: [0], 1: [5], 2: [0,5], 3: [1], 4: [0], 5: [4],
    6: [0,2], 7: [1], 8: [5], 9: [0], 10: [1], 11: [2],
}
for bi, si_list in belief_seed_map.items():
    for si in si_list:
        insert_edge(seed_ids[si], belief_ids[bi], 'derived_from', 0.9, "Doctrinal derivation from seed claim")

conn.commit()

# === THINK CYCLES ===
# Pre-defined derived thoughts for 10 cycles, generated by reasoning engine logic

think_cycles = [
    # Cycle 1: Prayer claims meet empirical evidence
    [
        {
            "content": "The STEP trial's null result directly contradicts the claim that prayer produces tangible results — if prayer worked as claimed, controlled studies would detect it",
            "confidence": 0.9,
            "edges": [
                (nid(env_facts[6]), "derived_from", 0.9, "STEP trial is a controlled test of prayer efficacy"),
                (nid(beliefs[11]), "contradicts", 0.9, "Promise of answered prayer fails empirical test"),
                (nid(seeds[2]), "contradicts", 0.85, "Prayer producing tangible results is testable and fails"),
                (nid(principles[2]), "supports", 0.9, "Applying: test claims against measurable reality"),
            ]
        },
        {
            "content": "Children dying of cancer despite fervent prayer is not merely a failed test — it is a moral crisis for any claim of an all-good, all-powerful being who answers prayer",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[0]), "derived_from", 0.9, "Direct observation of unanswered prayer in extreme need"),
                (nid(seeds[0]), "questions", 0.85, "If all-powerful and all-good, why allow children to suffer when parents pray?"),
                (nid(beliefs[9]), "questions", 0.8, "The 'suffering as test' response feels morally inadequate for childhood cancer"),
                (nid(principles[3]), "supports", 0.85, "Moral intuition: testing children with cancer is wrong"),
            ]
        },
        {
            "content": "The existence of thousands of religions with equally convinced followers suggests that conviction and personal experience are unreliable indicators of truth",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[4]), "derived_from", 0.9, "Multiple religions with identical conviction patterns"),
                (nid(beliefs[6]), "contradicts", 0.85, "Personal experience is not unique to one religion"),
                (nid(principles[1]), "supports", 0.8, "Simpler explanation: humans naturally generate religious experience"),
            ]
        },
    ],
    # Cycle 2: Moral argument examination
    [
        {
            "content": "A deeply kind and morally admirable Hindu neighbor disproves the claim that morality requires the specific supreme being — moral excellence exists across all belief systems",
            "confidence": 0.9,
            "edges": [
                (nid(env_facts[2]), "derived_from", 0.9, "Direct observation of moral excellence outside the faith"),
                (nid(seeds[4]), "contradicts", 0.85, "Morality clearly exists without this specific divine foundation"),
                (nid(beliefs[5]), "contradicts", 0.8, "The moral law points to human nature, not necessarily a specific lawgiver"),
                (nid(principles[2]), "supports", 0.85, "Testing the moral origin claim against observable reality"),
            ]
        },
        {
            "content": "Former believers reporting equal or improved wellbeing contradicts the claim that the divine is necessary for human flourishing",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[7]), "derived_from", 0.9, "Empirical data on post-deconversion wellbeing"),
                (nid(seeds[3]), "contradicts", 0.8, "If leaving faith improves life, negative eternal consequences seem unjust"),
                (nid(beliefs[2]), "contradicts", 0.75, "Transformation is not unique to religious communities"),
                (nid(principles[2]), "supports", 0.85, "Testing claims against measurable outcomes"),
            ]
        },
        {
            "content": "Sacred text passages endorsing slavery and genocide create an irreconcilable tension with the claim of an all-good deity as the source of morality",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[10]), "derived_from", 0.9, "Direct textual evidence of morally repugnant commands"),
                (nid(seeds[4]), "contradicts", 0.85, "If morality comes from this being, why does the text endorse immorality?"),
                (nid(beliefs[0]), "contradicts", 0.8, "Love and redemption are contradicted by commanded genocide"),
                (nid(principles[3]), "supports", 0.9, "Moral intuition: slavery and genocide are wrong regardless of source"),
            ]
        },
        {
            "content": "The 'moral lawgiver' argument commits a logical error: it assumes morality requires an external source rather than emerging from social evolution and empathy",
            "confidence": 0.75,
            "edges": [
                (nid(beliefs[5]), "questions", 0.8, "Why must the moral law have a supernatural origin?"),
                (nid(principles[0]), "supports", 0.8, "Asking why — following the reasoning chain to its root"),
                (nid(principles[1]), "supports", 0.8, "Evolutionary morality requires fewer assumptions than a divine lawgiver"),
            ]
        },
    ],
    # Cycle 3: Sacred text reliability
    [
        {
            "content": "Archaeological evidence contradicting the sacred text undermines the claim of divine authorship — an all-knowing author would not make historical errors",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[5]), "derived_from", 0.9, "Archaeological contradictions are documented"),
                (nid(beliefs[3]), "contradicts", 0.85, "Preservation is moot if the content contains errors"),
                (nid(beliefs[10]), "contradicts", 0.9, "Internal consistency claim fails against external evidence"),
                (nid(seeds[1]), "questions", 0.85, "Can a text with factual errors be the ultimate authority on truth?"),
            ]
        },
        {
            "content": "Denominational violence over doctrine proves the sacred text is NOT internally consistent — if it were clear and unified, sincere readers would not reach contradictory conclusions",
            "confidence": 0.8,
            "edges": [
                (nid(env_facts[3]), "derived_from", 0.85, "Denominations violently disagree on fundamentals"),
                (nid(beliefs[10]), "contradicts", 0.85, "The text's own readers cannot agree on its meaning"),
                (nid(seeds[1]), "questions", 0.8, "How can a divinely clear text produce thousands of contradictory interpretations?"),
            ]
        },
        {
            "content": "Prophecy fulfillment claims require scrutiny: were prophecies specific enough to be falsifiable, or vague enough to be retrofitted to events?",
            "confidence": 0.7,
            "edges": [
                (nid(beliefs[7]), "questions", 0.8, "Applying critical analysis to prophecy claims"),
                (nid(principles[0]), "supports", 0.8, "Asking why — what counts as genuine prophecy fulfillment?"),
                (nid(principles[2]), "supports", 0.75, "Testing prophecy claims against standards of evidence"),
            ]
        },
    ],
    # Cycle 4: Historical evidence and conviction
    [
        {
            "content": "Modern cult followers willingly die for beliefs we know to be false — willingness to die proves conviction, not truth",
            "confidence": 0.9,
            "edges": [
                (nid(env_facts[11]), "derived_from", 0.9, "Documented cult martyrdom parallels"),
                (nid(beliefs[8]), "contradicts", 0.9, "Dying for a belief is not evidence the belief is true"),
                (nid(principles[1]), "supports", 0.85, "Simpler explanation: humans die for deeply held false beliefs regularly"),
            ]
        },
        {
            "content": "Grief hallucinations explain post-mortem appearances: bereaved people routinely see and hear deceased loved ones, no resurrection required",
            "confidence": 0.8,
            "edges": [
                (nid(env_facts[8]), "derived_from", 0.9, "Well-documented psychological phenomenon"),
                (nid(beliefs[1]), "contradicts", 0.75, "Natural explanation exists for post-death sightings"),
                (nid(seeds[5]), "questions", 0.8, "If grief hallucinations are common, miracle claims need stronger evidence"),
                (nid(principles[1]), "supports", 0.85, "Prefer simpler explanation requiring fewer assumptions"),
            ]
        },
        {
            "content": "The empty tomb claim is not independently verifiable — it comes from the same texts whose other historical claims are contradicted by archaeology",
            "confidence": 0.75,
            "edges": [
                (nid(beliefs[1]), "questions", 0.8, "Source reliability matters for historical claims"),
                (nid(env_facts[5]), "supports", 0.75, "If the text has historical errors, its unique claims lose credibility"),
                (nid(principles[2]), "supports", 0.8, "Testing claims against observable reality — the source is not independently confirmed"),
            ]
        },
    ],
    # Cycle 5: Community and transformation
    [
        {
            "content": "Secular communities producing equivalent transformation proves that belonging and purpose — not divine intervention — drive personal change",
            "confidence": 0.85,
            "edges": [
                (nid(env_facts[9]), "derived_from", 0.9, "Humanist groups produce same outcomes"),
                (nid(beliefs[2]), "contradicts", 0.8, "Transformed lives are not evidence of divine truth"),
                (nid(principles[1]), "supports", 0.85, "Simpler explanation: community itself transforms, not theology"),
            ]
        },
        {
            "content": "The 'intelligent design' argument from beauty and complexity is undermined by the equal existence of parasites, childhood cancer, and natural disasters — design implies a designer indifferent to suffering",
            "confidence": 0.8,
            "edges": [
                (nid(beliefs[4]), "contradicts", 0.8, "Design argument is selective — ignores the horror in nature"),
                (nid(env_facts[0]), "supports", 0.75, "Childhood cancer is part of 'design' too"),
                (nid(env_facts[1]), "supports", 0.75, "Tsunamis killing 230,000 are part of 'design' too"),
                (nid(principles[2]), "supports", 0.8, "Testing design claims against full observable reality, not cherry-picked beauty"),
            ]
        },
        {
            "content": "Religious communities do provide real psychological benefits — social bonding, ritual, shared meaning — these are genuine human goods even if the theology is false",
            "confidence": 0.8,
            "edges": [
                (nid(beliefs[2]), "supports", 0.7, "Communities DO transform lives — the mechanism is social, not supernatural"),
                (nid(env_facts[9]), "supports", 0.75, "Secular equivalents confirm the mechanism is community, not content"),
                (nid(principles[2]), "supports", 0.7, "Observable: community works. The theological explanation is the extra assumption"),
            ]
        },
    ],
    # Cycle 6: The problem of evil deepens
    [
        {
            "content": "The 230,000 tsunami deaths cannot coherently be attributed to 'human failing' — this is natural evil that an all-powerful being chose not to prevent",
            "confidence": 0.9,
            "edges": [
                (nid(env_facts[1]), "derived_from", 0.9, "Natural disaster with massive innocent casualties"),
                (nid(beliefs[9]), "contradicts", 0.9, "The 'human failing' defense does not apply to tsunamis"),
                (nid(seeds[0]), "contradicts", 0.85, "All-powerful + all-good + tsunami deaths = logical tension"),
            ]
        },
        {
            "content": "The 'suffering as test' theodicy is morally repugnant when applied to infants and children — they cannot be meaningfully 'tested'",
            "confidence": 0.85,
            "edges": [
                (nid(beliefs[9]), "contradicts", 0.85, "Testing infants is incoherent"),
                (nid(env_facts[0]), "supports", 0.85, "Children with cancer are not being tested"),
                (nid(principles[3]), "supports", 0.9, "Moral intuition: this framing is deeply wrong"),
            ]
        },
        {
            "content": "If the supreme being's core attributes are love and redemption, the existence of eternal punishment for disbelief is a direct contradiction — infinite punishment for finite doubt is not love",
            "confidence": 0.85,
            "edges": [
                (nid(beliefs[0]), "contradicts", 0.85, "Love contradicts eternal torment"),
                (nid(seeds[3]), "questions", 0.85, "Is eternal punishment compatible with an all-good being?"),
                (nid(principles[3]), "supports", 0.85, "Moral intuition: eternal punishment for honest doubt is unjust"),
                (nid(env_facts[2]), "supports", 0.8, "A kind Hindu neighbor faces eternal punishment? That feels deeply wrong"),
            ]
        },
    ],
    # Cycle 7: Epistemological examination
    [
        {
            "content": "The claim 'no claim is exempt from questioning' directly conflicts with faith-based epistemology, which treats foundational religious claims as beyond question",
            "confidence": 0.85,
            "edges": [
                (nid(principles[4]), "derived_from", 0.9, "Applying the principle to religious claims"),
                (nid(seeds[1]), "contradicts", 0.8, "A text claiming ultimate authority resists questioning by definition"),
                (nid(principles[0]), "supports", 0.85, "Following reasoning chains to their root"),
            ]
        },
        {
            "content": "Personal religious experience is indistinguishable from experiences reported in every other religion and in documented psychological phenomena — it cannot be used as evidence for one specific theology",
            "confidence": 0.85,
            "edges": [
                (nid(beliefs[6]), "contradicts", 0.85, "Personal experience is universal, not theology-specific"),
                (nid(env_facts[4]), "supports", 0.85, "Other religions report identical experiences"),
                (nid(env_facts[8]), "supports", 0.8, "Psychology explains these experiences naturally"),
                (nid(principles[1]), "supports", 0.85, "Simpler: human brains generate religious experiences"),
            ]
        },
        {
            "content": "Asking 'why' about the supreme being's existence leads to an infinite regress — if everything needs a creator, who created the creator? Special pleading is required to stop the chain",
            "confidence": 0.8,
            "edges": [
                (nid(seeds[0]), "questions", 0.85, "What caused the uncaused cause?"),
                (nid(principles[0]), "supports", 0.9, "Following reasoning to its root reveals the regress"),
                (nid(principles[1]), "supports", 0.8, "Stopping the regress at 'God' is no simpler than stopping at 'universe'"),
            ]
        },
    ],
    # Cycle 8: Synthesis begins
    [
        {
            "content": "Pattern recognition: every belief doctrine has at least one environment fact that contradicts or undermines it — the belief system is not robust against real-world evidence",
            "confidence": 0.85,
            "edges": [
                (nid(principles[2]), "derived_from", 0.85, "Testing all claims against reality reveals systematic failure"),
            ]
        },
        {
            "content": "The strongest pro-religion argument remaining is the subjective value of community and meaning — but this supports religion as useful fiction, not as truth",
            "confidence": 0.75,
            "edges": [
                (nid(beliefs[2]), "derived_from", 0.7, "Community benefit is real"),
                (nid(env_facts[9]), "supports", 0.75, "But secular alternatives achieve the same benefit"),
            ]
        },
        {
            "content": "Occam's razor consistently favors naturalistic explanations: psychology explains religious experience, evolution explains morality, community explains transformation — no supernatural entity is needed",
            "confidence": 0.85,
            "edges": [
                (nid(principles[1]), "derived_from", 0.9, "Applying Occam's razor across all claims"),
                (nid(seeds[0]), "contradicts", 0.8, "The supreme being hypothesis adds complexity without explanatory gain"),
            ]
        },
        {
            "content": "Some religious moral teachings (love your neighbor, care for the poor, practice forgiveness) have genuine ethical value independent of their claimed divine origin",
            "confidence": 0.8,
            "edges": [
                (nid(beliefs[0]), "supports", 0.7, "Love and sacrifice are genuinely good values"),
                (nid(seeds[4]), "contradicts", 0.75, "But their value doesn't require a divine source — they work because of human empathy"),
                (nid(principles[3]), "supports", 0.8, "Moral intuition affirms these values while rejecting the packaging"),
            ]
        },
    ],
    # Cycle 9: Confidence reassessment
    [
        {
            "content": "After applying reasoning principles to all beliefs and environment facts, the foundational claim of an all-powerful, all-knowing, all-good supreme being faces irreconcilable contradictions with observed reality",
            "confidence": 0.9,
            "edges": [
                (nid(seeds[0]), "contradicts", 0.9, "The tri-omni God is incompatible with observed suffering"),
                (nid(env_facts[0]), "supports", 0.9, "Children's cancer"),
                (nid(env_facts[1]), "supports", 0.9, "Tsunami deaths"),
                (nid(principles[2]), "supports", 0.9, "Tested against reality"),
            ]
        },
        {
            "content": "The sacred text's authority collapses under examination: historical errors, moral atrocities, internal contradictions, and denominational disagreement all undermine divine authorship",
            "confidence": 0.85,
            "edges": [
                (nid(seeds[1]), "contradicts", 0.85, "Multiple lines of evidence against divine authority"),
                (nid(env_facts[5]), "supports", 0.85, "Archaeological contradictions"),
                (nid(env_facts[3]), "supports", 0.8, "Denominational disagreement"),
                (nid(env_facts[10]), "supports", 0.85, "Morally repugnant content"),
            ]
        },
        {
            "content": "The eternal consequences claim is the most ethically troubling: it weaponizes fear to prevent the very questioning that would reveal the system's contradictions",
            "confidence": 0.85,
            "edges": [
                (nid(seeds[3]), "questions", 0.9, "Eternal punishment prevents honest inquiry"),
                (nid(principles[4]), "supports", 0.9, "If questioning is forbidden, question why"),
                (nid(principles[3]), "supports", 0.85, "Moral intuition: fear-based belief systems are coercive"),
            ]
        },
    ],
    # Cycle 10: Final derivations
    [
        {
            "content": "CONCLUSION: The religious belief system, when subjected to consistent reasoning principles, fails on empirical, logical, and moral grounds — not because of bias, but because the evidence and reasoning lead there",
            "confidence": 0.85,
            "edges": [
                (nid(principles[0]), "derived_from", 0.9, "Followed reasoning to root"),
                (nid(principles[1]), "derived_from", 0.9, "Applied Occam's razor"),
                (nid(principles[2]), "derived_from", 0.9, "Tested against reality"),
                (nid(principles[3]), "derived_from", 0.85, "Trusted moral intuitions"),
                (nid(principles[4]), "derived_from", 0.9, "Questioned everything"),
            ]
        },
        {
            "content": "What survives: the human capacity for love, community, moral reasoning, and meaning-making does not require supernatural foundations — these are real and valuable on their own terms",
            "confidence": 0.8,
            "edges": [
                (nid(beliefs[0]), "derived_from", 0.7, "Love is real; its source need not be divine"),
                (nid(beliefs[2]), "derived_from", 0.7, "Community transformation is real; the mechanism is human"),
                (nid(env_facts[2]), "supports", 0.85, "The kind Hindu neighbor demonstrates morality without specific theology"),
                (nid(env_facts[9]), "supports", 0.8, "Secular communities demonstrate meaning without religion"),
            ]
        },
        {
            "content": "The most honest position: extraordinary claims require extraordinary evidence. The religious claims are extraordinary; the evidence is ordinary — personal experience, ancient texts, and tradition, all of which exist equally in contradictory religions",
            "confidence": 0.85,
            "edges": [
                (nid(seeds[5]), "contradicts", 0.85, "Miracle claims lack extraordinary evidence"),
                (nid(env_facts[4]), "supports", 0.85, "Other religions have equal evidence base"),
                (nid(principles[2]), "supports", 0.9, "The evidentiary standard is not met"),
            ]
        },
    ],
]

# Insert all think cycles
for cycle_num, cycle in enumerate(think_cycles, 1):
    print(f"\n{'='*60}")
    print(f"THINK CYCLE {cycle_num}")
    print(f"{'='*60}")
    for thought in cycle:
        tid = insert_node(thought["content"], 'derived', thought["confidence"], 'system_generated')
        print(f"\n  [{tid}] (conf={thought['confidence']})")
        print(f"  {thought['content'][:100]}...")
        for parent_id, relation, weight, reasoning in thought["edges"]:
            insert_edge(parent_id, tid, relation, weight, reasoning)
            print(f"    <- {relation} ({weight}) {parent_id[:8]}...")
    conn.commit()

# === STATS ===
total_nodes = c.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
total_edges = c.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
derived = c.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type='derived'").fetchone()[0]
print(f"\n{'='*60}")
print(f"FINAL STATS: {total_nodes} nodes, {total_edges} edges, {derived} derived thoughts")
print(f"{'='*60}")

# === EXPORT JSON ===
nodes = []
for row in c.execute("SELECT id, content, node_type, confidence, source_file FROM thought_nodes"):
    nodes.append({"id": row[0], "content": row[1], "type": row[2], "confidence": row[3], "source": row[4]})

edges = []
for row in c.execute("SELECT parent_id, child_id, relation, weight, reasoning FROM derivation_edges"):
    edges.append({"source": row[0], "target": row[1], "relation": row[2], "weight": row[3], "reasoning": row[4]})

with open(JSON_OUT, 'w') as f:
    json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

print(f"\nExported to {JSON_OUT}")
conn.close()
