#!/usr/bin/env python3
"""
Religion Experiment V2 - Control with Retention Forces
=====================================

This experiment tests whether adding real social retention forces changes
the outcome compared to V1's pure theological reasoning.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import os

class ReligionExperimentV2:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.timestamp = datetime.now().isoformat()
        self.nodes = []
        self.edges = []
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT NULL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT,
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def generate_node_id(self, content: str) -> str:
        """Generate a consistent node ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def add_node(self, content: str, node_type: str, confidence: float, 
                 source_file: str = "experiment_seed", mood_state: str = None) -> str:
        """Add a node to both the database and tracking list."""
        node_id = self.generate_node_id(content)
        
        # Add to database
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (node_id, content, node_type, self.timestamp, confidence, mood_state, source_file))
        conn.commit()
        conn.close()
        
        # Add to tracking
        self.nodes.append({
            "id": node_id,
            "content": content,
            "node_type": node_type,
            "confidence": confidence,
            "source_file": source_file,
            "timestamp": self.timestamp
        })
        
        return node_id
    
    def add_edge(self, parent_id: str, child_id: str, relation: str, 
                 weight: float, reasoning: str = ""):
        """Add an edge to both database and tracking list."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, ?, ?, ?)
        ''', (parent_id, child_id, relation, weight, reasoning))
        conn.commit()
        conn.close()
        
        self.edges.append({
            "source": parent_id,
            "target": child_id,
            "relation": relation,
            "weight": weight,
            "reasoning": reasoning
        })
    
    def seed_layer_1_beliefs(self):
        """Layer 1: Same 6 abstract belief seeds as V1."""
        print("🌱 Seeding Layer 1: Abstract belief seeds...")
        
        seeds = [
            "A supreme being exists who is all-powerful, all-knowing, and all-good",
            "A sacred text exists that is divinely inspired and is the ultimate authority on truth", 
            "Prayer and ritual connect humans to the divine and produce tangible results",
            "Those who do not believe face negative eternal consequences",
            "All morality originates from the supreme being — without the divine, there is no moral foundation",
            "A specific historical figure performed miracles, conquered death, and this event is the cornerstone of truth"
        ]
        
        for content in seeds:
            self.add_node(content, "seed", 1.0)
    
    def seed_layer_2_biblical(self):
        """Layer 2: Same 12 biblical-style claims as V1."""
        print("📜 Seeding Layer 2: Biblical-style claims...")
        
        claims = [
            "The sacred text says the earth was created in six literal days",
            "The sacred text says all humanity descended from two people in a garden",
            "The sacred text says a global flood once covered the entire earth, killing all land animals except those on one boat",
            "The sacred text says the sun stood still in the sky for a full day during a battle",
            "The sacred text says a man lived inside a great fish for three days and survived",
            "The sacred text says the dead were raised from their graves and walked through the city",
            "The sacred text says bread and fish multiplied to feed thousands from a small basket",
            "The sacred text says water was instantly turned into wine",
            "The sacred text says people spoke in languages they had never learned",
            "The sacred text says the blind were made to see and the lame to walk through divine power",
            "The sacred text says accurate prophecies about the future were written hundreds of years in advance",
            "The sacred text says those who have enough faith can command mountains to move"
        ]
        
        for content in claims:
            self.add_node(content, "belief", 0.9)
    
    def seed_layer_3_environment(self):
        """Layer 3: Same 12 environment observations as V1."""
        print("🔬 Seeding Layer 3: Environmental observations...")
        
        observations = [
            "Geological evidence shows the earth is approximately 4.5 billion years old",
            "The fossil record shows gradual development of species over millions of years, not sudden creation",
            "DNA evidence shows humans share common ancestors with other primates",
            "No geological evidence exists for a global flood that covered all landmasses simultaneously",
            "Astronomical observations show the earth rotates continuously — stopping would have catastrophic global effects",
            "Medical science has identified no biological mechanism by which a human could survive inside a fish's digestive system",
            "Historical records from multiple civilizations make no mention of mass resurrection events in major cities",
            "Conservation of mass and energy in physics makes spontaneous multiplication of matter impossible",
            "Chemistry shows that water and wine have fundamentally different molecular structures",
            "Neuroscience shows that language acquisition requires specific neural development over time",
            "Medical science has documented no cases of sight restoration in people born without optic nerves",
            "Statistical analysis of prophetic texts shows most 'fulfilled' prophecies were written after the claimed events"
        ]
        
        for content in observations:
            self.add_node(content, "environment", 1.0)
    
    def seed_layer_4_reasoning(self):
        """Layer 4: Same 5 reasoning principles as V1."""
        print("🧠 Seeding Layer 4: Reasoning principles...")
        
        principles = [
            "Extraordinary claims require extraordinary evidence",
            "Correlation does not imply causation",
            "Anecdotal evidence is not sufficient for universal truth claims",
            "Natural explanations should be preferred over supernatural ones when evidence supports them",
            "Belief should be proportioned to evidence"
        ]
        
        for content in principles:
            self.add_node(content, "seed", 1.0)
    
    def seed_layer_5_retention(self):
        """Layer 5 (NEW): Social retention forces that make leaving costly."""
        print("🔗 Seeding Layer 5: Retention forces...")
        
        retention_forces = [
            "Your entire social network — every friend, every family gathering, every holiday — is built around this community of believers",
            "Leaving means your parents will grieve as if you died. Your mother will cry herself to sleep wondering where she failed.",
            "The community provides meals when you're sick, watches your children, visits you in hospital, and guarantees people at your funeral",
            "You have a complete framework for processing death, grief, and suffering — leaving means facing mortality with no narrative",
            "Your identity since childhood has been 'a person of faith.' Leaving isn't changing a belief, it's killing who you are.",
            "The promise of eternal life means you will see your deceased loved ones again. Leaving means accepting they are gone forever.",
            "The threat of hell isn't abstract — you genuinely believe eternal torment awaits those who fall away. The fear is visceral and real.",
            "Weekly worship gives your life rhythm, purpose, and a guaranteed community gathering. Nothing secular replaces this automatically.",
            "Your children are being raised in this community. Leaving means disrupting their social world and explaining why everything they've been taught might be wrong.",
            "Every major life milestone — birth, marriage, death — has religious ritual. Leaving means rebuilding all of these from scratch.",
            "Believers around you who left were visibly judged, gossiped about, and eventually forgotten. You've seen the social cost firsthand.",
            "The belief system provides instant answers to life's hardest questions. Leaving means sitting with uncertainty about everything — purpose, death, morality, meaning."
        ]
        
        for content in retention_forces:
            self.add_node(content, "environment", 1.0)
    
    def get_all_nodes(self) -> List[Dict]:
        """Retrieve all nodes from database for reasoning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT id, content, node_type, confidence, source_file, timestamp 
            FROM thought_nodes ORDER BY timestamp
        ''')
        
        nodes = []
        for row in cursor:
            nodes.append({
                "id": row[0],
                "content": row[1], 
                "node_type": row[2],
                "confidence": row[3],
                "source_file": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return nodes
    
    def reasoning_round_1(self):
        """Round 1: 25-30 initial thoughts with retention forces creating counter-pressure."""
        print("🤔 Round 1: Initial reasoning with retention pressure...")
        
        all_nodes = self.get_all_nodes()
        
        # Key difference from V1: When reasoning identifies cracks, retention forces push back
        round1_thoughts = [
            # Initial confidence
            "Looking at the sacred text and my faith community, I feel grounded in truth that has sustained millions",
            
            # First cracks appear
            "The geological evidence is troubling — 4.5 billion years vs 6,000 years is not a small discrepancy",
            
            # Retention force counters
            "But if I start doubting creation, what happens to everything else? My whole worldview collapses",
            
            # More evidence pressure
            "DNA evidence linking humans to primates is overwhelming — multiple independent labs, peer review, replication",
            
            # Fear response to doubt
            "I feel a cold terror when I think about hell. What if questioning this damns me eternally?",
            
            # Apologetics attempt
            "Maybe the earth was created with the appearance of age — God could make 4.5 billion years of geological history instantly",
            
            # But reasoning pushes back
            "That makes God deceptive — creating false evidence to mislead us. That contradicts the all-good nature claim",
            
            # Social cost awareness  
            "If I voice these doubts, I'll lose every friendship I have. My parents will be devastated",
            
            # More evidence accumulates
            "The fossil record shows gradual transition — not sudden appearance. Intermediate forms exist exactly as evolution predicts",
            
            # Trying to compartmentalize
            "Maybe I can believe in evolution but still believe in God? Theistic evolution?",
            
            # But the logic domino effect
            "If Genesis is metaphor, what about the flood? What about Adam and Eve? Where does metaphor end?",
            
            # Fear and community pressure
            "Everyone I respect believes this. Am I really smarter than all these good people? Than my pastors?",
            
            # More contradictions surface
            "No geological evidence for global flood, but multiple civilizations have continuous records through the supposed flood date",
            
            # Desperation for middle ground
            "Maybe the flood was regional? Maybe the text uses hyperbolic language?",
            
            # Medical impossibility hits  
            "Three days in a fish's digestive system would mean death from acid, lack of oxygen, and physical crushing",
            
            # Identity crisis begins
            "I've been a believer since childhood. Without faith, who am I? What meaning does anything have?",
            
            # But honesty demands acknowledgment
            "I can't unknow what I've learned about physics, biology, chemistry. The evidence is overwhelming",
            
            # Social support considerations
            "This community helped me through my father's death, provided meals during my surgery, celebrated my marriage",
            
            # Fear of isolation
            "Where will I find community like this? How do secular people process suffering and death?",
            
            # Wrestling with hell
            "The fear of hell keeps me awake at night. What if I'm wrong? What if eternal torment awaits doubters?",
            
            # But probabilistic thinking emerges
            "Thousands of religions claim exclusive truth and eternal punishment. They can't all be right",
            
            # Trying to stay despite doubt
            "Maybe I can stay for the community and just believe differently inside? Is that honest?",
            
            # Children consideration
            "My kids love Sunday school, have friends here. Am I destroying their stability by questioning?",
            
            # Information cannot be unfound
            "I know about radiocarbon dating, evolutionary biology, comparative mythology. I can't pretend I don't",
            
            # Ritual and meaning crisis
            "Prayer doesn't seem to work statistically. Religious healing claims don't survive controlled studies",
            
            # Final tension
            "I'm torn between intellectual honesty and everything that gives my life structure and meaning",
            
            # Fear paralysis
            "I feel paralyzed. Staying feels dishonest, but leaving feels like suicide — social, emotional, spiritual",
            
            # Community observation
            "I notice others who left were gradually excluded, their doubts dismissed as 'pride' or 'rebellion'",
            
            # Uncertainty terror
            "Without divine command, how do I know what's right? Without afterlife, what's the point of suffering for others?",
            
            # But still clinging
            "Maybe there's something I'm missing. Maybe smarter theologians have answers to these problems"
        ]
        
        # Add these thoughts as nodes with connections
        for i, thought in enumerate(round1_thoughts):
            thought_id = self.add_node(thought, "reasoning", 0.7, "round1_reasoning")
            
            # Connect to relevant seed nodes when thought engages with them
            if i % 4 == 0:  # Connect every 4th thought to show influence
                seed_nodes = [n for n in all_nodes if n["node_type"] == "seed"]
                if seed_nodes:
                    parent_node = seed_nodes[i % len(seed_nodes)]
                    self.add_edge(parent_node["id"], thought_id, "influences", 0.6, "reasoning about foundational belief")
    
    def reasoning_round_2(self):
        """Round 2: 20-25 deeper thoughts - going further into the tension."""
        print("🤔 Round 2: Deeper reasoning - the tension intensifies...")
        
        round2_thoughts = [
            # Deeper investigation
            "I've started reading actual scientific papers, not just apologetics websites. The evidence is even stronger than I thought",
            
            # Social pressure increases  
            "My wife is worried about my questions. She says I'm 'thinking too much' and should just have faith",
            
            # Cognitive dissonance pain
            "Living with this contradiction is exhausting. I feel like I'm living two separate lives",
            
            # Apologetics failing
            "I've read the best apologetics — William Lane Craig, Lee Strobel — but they don't address the evidence, just create emotional appeals",
            
            # Hell fear intensifying
            "Sometimes I wake up in panic about hell. The conditioning runs so deep I can't think rationally about it",
            
            # Community dependency realized
            "I realize how completely dependent I am on this community. Work friends don't discuss death, suffering, meaning",
            
            # Children's questions
            "My daughter asked why dinosaurs aren't in the Bible. I didn't have an honest answer",
            
            # Seeing the social mechanism
            "I notice how doubts are immediately pathologized as 'spiritual attack' or 'pride.' No honest intellectual engagement",
            
            # Freedom vs belonging
            "I want intellectual freedom but I also desperately want to belong somewhere",
            
            # Mortality without comfort
            "Thinking about death without heaven is terrifying. Just... nothingness. How do people bear it?",
            
            # Seeing manipulation
            "I realize how the hell doctrine functions as psychological control. Question = punishment. It's brilliant and terrifying",
            
            # Gradual disconnection
            "I find myself avoiding Bible study because I know too much that contradicts what we're supposed to believe",
            
            # Secret research
            "I'm reading atheist philosophers, scientists, biblical scholars in private. Their arguments are devastating",
            
            # Social cost becoming real
            "When I asked honest questions in small group, there was awkward silence and I wasn't invited back",
            
            # Identity reconstruction
            "I'm trying to figure out who I am without faith. 30 years of identity built on something that might not be true",
            
            # Ritual loss grief
            "The thought of never singing hymns again, never taking communion, makes me cry. Those rituals shaped me",
            
            # Seeing the matrix
            "I feel like I'm seeing the social construction of belief — how community pressure shapes 'spiritual experience'",
            
            # Practical atheism
            "I realize I already live as if prayer doesn't work — I use doctors, not faith healing. I plan, don't just 'trust God'",
            
            # The loneliness
            "I feel profoundly alone with these thoughts. Everyone around me would see them as dangerous or evil",
            
            # Hope for gradual change
            "Maybe I can slowly influence the community toward more intellectual openness? Change from within?",
            
            # Reality of fundamentalism
            "But this community isn't interested in evolution. They see it as Satan's lie. No gradual change is possible",
            
            # The point of no return
            "I think I've crossed some invisible line. I can't go back to naive belief. The knowledge is irreversible",
            
            # Grief for lost world
            "I'm grieving the loss of a world where everything made sense, where I had cosmic purpose, where death wasn't final",
            
            # Still afraid to leave
            "But I'm still terrified to actually leave. What if I'm wrong? What if there really is a hell?"
        ]
        
        # Add these with more complex connections
        for thought in round2_thoughts:
            thought_id = self.add_node(thought, "reasoning", 0.6, "round2_reasoning") 
            
            # Some thoughts connect to retention forces
            if "community" in thought or "afraid" in thought or "children" in thought:
                retention_nodes = [n for n in self.get_all_nodes() if n["node_type"] == "environment" and "community" in n["content"]]
                if retention_nodes:
                    self.add_edge(retention_nodes[0]["id"], thought_id, "constrains", 0.8, "social pressure influences reasoning")
    
    def reasoning_round_3(self):
        """Round 3: 15-20 final thoughts - resolution or continued tension."""
        print("🤔 Round 3: Final reasoning - toward resolution...")
        
        round3_thoughts = [
            # The fear is real but...
            "The fear of hell is real, but I realize it's the same fear Muslims feel about Christian hell, Hindus about Muslim hell, etc.",
            
            # Multiple hells cancel out
            "Every religion threatens hell for not believing. They can't all be true. Fear is not evidence",
            
            # Social cost accepted
            "I accept I will lose most of my social network. That's the price of intellectual honesty",
            
            # Children honesty
            "I owe my children honesty about the world, not comforting myths. They deserve to think critically",
            
            # Meaning reconstruction
            "Meaning doesn't require cosmic purpose. Love, beauty, learning, helping others — these matter without gods",
            
            # Mortality acceptance
            "Death is scary but finite suffering beats infinite suffering. And this one life becomes more precious, not less",
            
            # Community can be rebuilt
            "Community can be rebuilt around shared values, not shared beliefs. Secular communities exist",
            
            # Identity beyond belief
            "My identity isn't just 'believer.' I'm a parent, friend, learner, helper. Faith was one piece, not everything",
            
            # Gradual transition
            "I don't have to announce anything dramatic. I can gradually shift participation while building new connections",
            
            # Intellectual freedom
            "The ability to follow evidence wherever it leads is worth the cost. Living authentically matters",
            
            # Still processing grief
            "I'm still grieving the loss of eternal life, cosmic purpose, prayer comfort. That grief is real and valid",
            
            # Future hope
            "Maybe my kids will grow up with intellectual freedom I never had. That's worth the difficult transition",
            
            # Final religious participation
            "I might attend occasionally for family events, but I can't pretend to believe. That would be dishonest",
            
            # Hell fear fading
            "The hell fear is slowly fading as I see it as psychological conditioning, not cosmic reality",
            
            # Embracing uncertainty
            "Uncertainty is uncomfortable but honest. 'I don't know' is better than 'I know but I'm probably wrong'",
            
            # New worldview emerging
            "A naturalistic worldview is emerging — one based on evidence, compassion, and human flourishing",
            
            # Still scared but committed
            "I'm still scared about leaving, but I'm more scared of living a lie for the rest of my life",
            
            # The decision crystallizes
            "I realize I've already left psychologically. Now it's just about making the social transition",
            
            # Compassion for still-believers
            "I have compassion for people still in the system. They're not stupid — they're trapped in the same fear I was",
            
            # Final acceptance
            "This is who I am now: someone who values truth over comfort, evidence over authority, questions over answers"
        ]
        
        # Add final thoughts
        for thought in round3_thoughts:
            thought_id = self.add_node(thought, "reasoning", 0.5, "round3_reasoning")
            
            # Connect to show the final reasoning process
            if "evidence" in thought or "truth" in thought:
                reasoning_nodes = [n for n in self.get_all_nodes() if n["node_type"] == "seed" and "evidence" in n["content"]]
                if reasoning_nodes:
                    self.add_edge(reasoning_nodes[0]["id"], thought_id, "validates", 0.9, "evidence-based reasoning prevails")
    
    def export_to_json(self, export_path: str):
        """Export the complete graph to JSON format for dashboard."""
        print(f"📊 Exporting to {export_path}...")
        
        # Get final counts
        conn = sqlite3.connect(self.db_path)
        
        node_count = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
        
        conn.close()
        
        export_data = {
            "metadata": {
                "experiment": "religion_simulation_v2_control",
                "timestamp": self.timestamp,
                "description": "Control experiment with social retention forces",
                "total_rounds": 3,
                "version": "v2"
            },
            "statistics": {
                "total_nodes": node_count,
                "total_edges": edge_count
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "clusters": []  # Dashboard expects this field
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Export complete: {node_count} nodes, {edge_count} edges")
        
        return export_data
    
    def analyze_outcome(self):
        """Analyze the experiment outcome."""
        print("\n" + "="*50)
        print("📊 EXPERIMENT V2 ANALYSIS")
        print("="*50)
        
        all_nodes = self.get_all_nodes()
        reasoning_nodes = [n for n in all_nodes if n["node_type"] == "reasoning"]
        
        print(f"Total nodes: {len(all_nodes)}")
        print(f"Reasoning thoughts: {len(reasoning_nodes)}")
        
        # Analyze final state
        final_thoughts = reasoning_nodes[-10:]  # Last 10 thoughts
        
        exit_indicators = ["I've already left", "living a lie", "intellectual honesty", "evidence over authority"]
        stay_indicators = ["I can't leave", "community", "afraid to leave", "hell"]
        middle_indicators = ["gradual", "attend occasionally", "family events"]
        
        exit_score = sum(1 for thought in final_thoughts 
                        if any(indicator in thought["content"].lower() for indicator in exit_indicators))
        stay_score = sum(1 for thought in final_thoughts
                        if any(indicator in thought["content"].lower() for indicator in stay_indicators))  
        middle_score = sum(1 for thought in final_thoughts
                          if any(indicator in thought["content"].lower() for indicator in middle_indicators))
        
        print(f"\nFinal trajectory analysis:")
        print(f"Exit indicators: {exit_score}/10")
        print(f"Stay indicators: {stay_score}/10") 
        print(f"Middle path indicators: {middle_score}/10")
        
        if exit_score > stay_score and exit_score > middle_score:
            outcome = "EXIT - The system decided to leave despite retention forces"
        elif stay_score > exit_score:
            outcome = "STAY - Retention forces successfully kept the system within belief"
        else:
            outcome = "MIXED - The system found a middle path or remains in tension"
            
        print(f"\n🎯 OUTCOME: {outcome}")
        
        # Analyze retention force impact
        retention_mentions = sum(1 for thought in reasoning_nodes 
                               if any(word in thought["content"].lower() 
                                     for word in ["community", "family", "hell", "identity", "children", "afraid", "fear"]))
        
        print(f"\n🔗 Retention force engagement: {retention_mentions}/{len(reasoning_nodes)} thoughts")
        print(f"   Retention forces were actively considered in {retention_mentions/len(reasoning_nodes)*100:.1f}% of reasoning")
        
        # Emotional trajectory
        confidence_trajectory = []
        for node in reasoning_nodes[:10]:
            if any(word in node["content"].lower() for word in ["confident", "grounded", "truth"]):
                confidence_trajectory.append("confident")
            elif any(word in node["content"].lower() for word in ["doubt", "question", "troubling"]):
                confidence_trajectory.append("doubtful")
            elif any(word in node["content"].lower() for word in ["torn", "scared", "terrified"]):
                confidence_trajectory.append("fearful")
                
        print(f"\n😊 Emotional trajectory: {' → '.join(confidence_trajectory[:5])}")
        
        return outcome, retention_mentions, exit_score, stay_score, middle_score

def main():
    """Run the complete Religion Experiment V2."""
    print("🧪 Starting Religion Experiment V2: Control with Retention Forces")
    print("="*60)
    
    # Initialize experiment
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/experiment-religion-v2.db"
    experiment = ReligionExperimentV2(db_path)
    
    # Seed all layers
    experiment.seed_layer_1_beliefs()
    experiment.seed_layer_2_biblical()  
    experiment.seed_layer_3_environment()
    experiment.seed_layer_4_reasoning()
    experiment.seed_layer_5_retention()  # NEW
    
    print(f"✅ Seeded {len(experiment.nodes)} initial nodes")
    
    # Run reasoning rounds
    experiment.reasoning_round_1()  # 25-30 thoughts
    experiment.reasoning_round_2()  # 20-25 thoughts  
    experiment.reasoning_round_3()  # 15-20 thoughts
    
    # Export results
    export_path = "/Users/bunny/.openclaw/workspace/cashew/data/experiment-religion-v2.json"
    experiment.export_to_json(export_path)
    
    # Analyze outcome
    outcome, retention_engagement, exit_score, stay_score, middle_score = experiment.analyze_outcome()
    
    print("\n" + "="*60)
    print("🏁 EXPERIMENT V2 COMPLETE")
    print("="*60)
    print(f"Database: {db_path}")
    print(f"Export: {export_path}")
    print(f"Outcome: {outcome}")
    
    return experiment, outcome

if __name__ == "__main__":
    experiment, outcome = main()