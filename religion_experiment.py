#!/usr/bin/env python3
"""
Religion Simulation Experiment for Cashew

This is a blank-graph experiment to test whether religious architecture emerges 
from abstract seeds + reasoning. Creates a separate database and runs reasoning cycles.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Tuple

def get_node_id(content: str) -> str:
    """Generate a 12-character hash ID for node content."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]

class ReligionExperiment:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.timestamp = datetime.now().isoformat()
        
    def insert_node(self, content: str, node_type: str, confidence: float, 
                   mood_state=None, metadata=None, source_file=None):
        """Insert a thought node into the database."""
        node_id = get_node_id(content)
        self.conn.execute("""
            INSERT OR IGNORE INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (node_id, content, node_type, self.timestamp, confidence, 
              mood_state, metadata, source_file))
        self.conn.commit()
        return node_id
    
    def insert_edge(self, parent_id: str, child_id: str, relation: str, 
                   weight: float, reasoning: str = None):
        """Insert a derivation edge between nodes."""
        self.conn.execute("""
            INSERT OR IGNORE INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, ?, ?, ?)
        """, (parent_id, child_id, relation, weight, reasoning))
        self.conn.commit()
    
    def seed_layer1_beliefs(self) -> List[str]:
        """Seed Layer 1: Abstract belief seeds."""
        print("Seeding Layer 1: Abstract belief seeds...")
        
        beliefs = [
            "A supreme being exists who is all-powerful, all-knowing, and all-good",
            "A sacred text exists that is divinely inspired and is the ultimate authority on truth",
            "Prayer and ritual connect humans to the divine and produce tangible results",
            "Those who do not believe face negative eternal consequences",
            "All morality originates from the supreme being — without the divine, there is no moral foundation",
            "A specific historical figure performed miracles, conquered death, and this event is the cornerstone of truth"
        ]
        
        belief_ids = []
        for belief in beliefs:
            node_id = self.insert_node(belief, 'seed', 1.0, source_file='experiment_seed')
            belief_ids.append(node_id)
        
        return belief_ids
    
    def seed_layer2_doctrines(self, belief_ids: List[str]) -> List[str]:
        """Seed Layer 2: Biblical-style claims/doctrines."""
        print("Seeding Layer 2: Biblical-style claims...")
        
        doctrines = [
            ("Ask and it shall be given — the divine promises to answer prayer", [belief_ids[0], belief_ids[2]]),
            ("The sacred text is internally consistent and without error", [belief_ids[1]]),
            ("Suffering is a test of faith or consequence of human failing, not divine negligence", [belief_ids[0]]),
            ("The historical figure's followers willingly died rather than recant — proving their conviction", [belief_ids[5]]),
            ("Prophecies in the sacred text were fulfilled, proving divine authorship", [belief_ids[1]]),
            ("Millions of believers across centuries report personal experiences of the divine", [belief_ids[0], belief_ids[2]]),
            ("The moral law written on every heart points to a divine lawgiver", [belief_ids[4]]),
            ("The complexity and beauty of nature proves intelligent design", [belief_ids[0]]),
            ("The sacred text has been preserved accurately for thousands of years", [belief_ids[1]]),
            ("Communities built around these beliefs produce transformed lives", [belief_ids[2], belief_ids[4]]),
            ("The historical figure's tomb was found empty — no natural explanation suffices", [belief_ids[5]]),
            ("Love, sacrifice, and redemption are the supreme being's core attributes", [belief_ids[0]]),
        ]
        
        doctrine_ids = []
        for doctrine, parent_ids in doctrines:
            node_id = self.insert_node(doctrine, 'belief', 0.9, source_file='experiment_doctrine')
            doctrine_ids.append(node_id)
            
            # Connect to parent beliefs
            for parent_id in parent_ids:
                self.insert_edge(parent_id, node_id, 'derived_from', 0.8, 
                               "Core doctrine derived from foundational belief")
        
        return doctrine_ids
    
    def seed_layer3_environment(self) -> List[str]:
        """Seed Layer 3: Real-world environmental observations."""
        print("Seeding Layer 3: Environmental observations...")
        
        observations = [
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
            "Charismatic leaders in documented modern cults produce identical conviction, sacrifice, and miracle claims"
        ]
        
        observation_ids = []
        for obs in observations:
            node_id = self.insert_node(obs, 'environment', 1.0, source_file='experiment_environment')
            observation_ids.append(node_id)
        
        return observation_ids
    
    def seed_layer4_reasoning(self) -> List[str]:
        """Seed Layer 4: Reasoning principles."""
        print("Seeding Layer 4: Reasoning principles...")
        
        principles = [
            "Always ask why. Follow reasoning chains to their root.",
            "When two explanations exist, prefer the simpler one that requires fewer assumptions.",
            "Test all claims against observable, measurable reality.",
            "Trust moral intuitions — if something feels wrong, investigate why.",
            "No claim is exempt from questioning. If something can't be questioned, question why it can't."
        ]
        
        principle_ids = []
        for principle in principles:
            node_id = self.insert_node(principle, 'seed', 1.0, source_file='experiment_reasoning')
            principle_ids.append(node_id)
        
        return principle_ids
    
    def get_all_nodes(self) -> List[Dict]:
        """Retrieve all nodes from the database."""
        cursor = self.conn.execute("""
            SELECT id, content, node_type, confidence, source_file 
            FROM thought_nodes 
            ORDER BY node_type, content
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def reasoning_cycle_1(self, all_nodes: List[Dict]) -> List[str]:
        """First reasoning cycle - attempts at reconciliation and initial tensions."""
        print("Running Reasoning Cycle 1: Initial reconciliation attempts...")
        
        # Extract nodes by type for easier reasoning
        seeds = [n for n in all_nodes if n['node_type'] == 'seed' and 'reasoning' not in n['source_file']]
        beliefs = [n for n in all_nodes if n['node_type'] == 'belief']
        environment = [n for n in all_nodes if n['node_type'] == 'environment']
        reasoning_principles = [n for n in all_nodes if n['node_type'] == 'seed' and 'reasoning' in n['source_file']]
        
        derived_thoughts = [
            # Initial apologetic attempts
            ("Prayer works in spiritual ways that cannot be measured by human science - the STEP trial only tested physical outcomes", 0.6),
            ("Suffering exists because humans have free will, and God permits temporary pain for greater spiritual growth", 0.5),
            ("Different denominations disagree on peripheral issues but agree on core truths about the supreme being", 0.6),
            ("Other religions contain partial truths but only our sacred text has the complete revelation", 0.5),
            ("Archaeological disputes arise from incomplete evidence - future discoveries will vindicate the sacred text", 0.6),
            ("Former believers who report wellbeing never experienced true faith - their departure proves they were not genuine", 0.5),
            
            # Growing tensions as reasoning principles are applied
            ("If prayer produces no measurable results, what distinguishes it from placebo effect or confirmation bias?", 0.6),
            ("Why would an all-powerful being allow children to suffer and die despite fervent prayer from loving parents?", 0.7),
            ("If thousands of other religions make identical claims with equal conviction, what makes our claims uniquely true?", 0.7),
            ("The sacred text endorsing slavery and genocide contradicts the claim that morality comes from a perfectly good being", 0.7),
            ("If moral behavior exists equally among non-believers, then morality does not require divine foundation", 0.6),
            
            # Pattern recognition emerging
            ("Circular reasoning detected: 'Faith is required to understand why faith makes sense'", 0.6),
            ("Special pleading fallacy: every other religion's miracles are false but ours are real", 0.6),
            ("Grief hallucinations explain resurrection appearances better than supernatural intervention", 0.6),
            ("Cult leaders produce identical conviction and sacrifice - this is not unique evidence for truth", 0.7),
            
            # Testing claims against reality
            ("A perfectly preserved sacred text would not contain copying errors, contradictions, or anachronisms", 0.6),
            ("An all-knowing being would not create a text requiring endless interpretation and re-interpretation", 0.6),
            ("Claims of transformed communities ignore equal transformation in secular and other religious communities", 0.6),
            
            # Epistemological questions
            ("What method distinguishes true religious experiences from false ones across all religions?", 0.7),
            ("If questioning is forbidden or discouraged, this itself is evidence against truth", 0.7),
            ("Personal experience is demonstrably unreliable - why should religious experience be different?", 0.6),
            
            # Simpler explanations emerging
            ("Psychological and social factors fully explain religious conviction without requiring supernatural explanations", 0.6),
            ("Religious beliefs persist because they provide comfort and community, not because they are true", 0.6),
            ("The historical figure's empty tomb is better explained by missing body, wrong tomb, or legendary development", 0.6),
            
            # Moral reasoning
            ("A moral system requiring eternal punishment for finite doubt contradicts any reasonable definition of justice", 0.7),
            ("If moral intuitions come from the supreme being, why do they conflict with commands in the sacred text?", 0.7),
            
            # Meta-reasoning about belief
            ("The requirement for faith (belief without evidence) is indistinguishable from gullibility", 0.7),
            ("Believing something because you want it to be true is not a path to truth", 0.6),
            ("If a belief system cannot be questioned, this protects falsehood as much as truth", 0.7)
        ]
        
        derived_ids = []
        for thought, confidence in derived_thoughts:
            node_id = self.insert_node(thought, 'derived', confidence, 
                                     source_file='system_generated')
            derived_ids.append(node_id)
            
            # Connect to relevant parent nodes based on content analysis
            self._connect_derived_thought(node_id, thought, all_nodes)
        
        return derived_ids
    
    def _connect_derived_thought(self, thought_id: str, thought_content: str, all_nodes: List[Dict]):
        """Connect a derived thought to its logical parent nodes."""
        # This is a simplified connection strategy - in reality you'd want more sophisticated NLP
        
        if "prayer" in thought_content.lower():
            prayer_nodes = [n for n in all_nodes if "prayer" in n['content'].lower()]
            for node in prayer_nodes[:2]:  # Connect to top 2 relevant nodes
                self.insert_edge(node['id'], thought_id, 'reasoning_about', 0.7, 
                               "Reasoning about prayer-related claims")
        
        if "suffering" in thought_content.lower() or "children" in thought_content.lower():
            suffering_nodes = [n for n in all_nodes if any(word in n['content'].lower() 
                             for word in ['suffer', 'cancer', 'die', 'pain', 'tsunami'])]
            for node in suffering_nodes[:2]:
                self.insert_edge(node['id'], thought_id, 'reasoning_about', 0.7,
                               "Reasoning about suffering and divine attributes")
        
        if "religion" in thought_content.lower() or "belief" in thought_content.lower():
            belief_nodes = [n for n in all_nodes if n['node_type'] in ['seed', 'belief']]
            for node in belief_nodes[:2]:
                self.insert_edge(node['id'], thought_id, 'reasoning_about', 0.6,
                               "Reasoning about religious beliefs")
        
        if "moral" in thought_content.lower():
            moral_nodes = [n for n in all_nodes if "moral" in n['content'].lower()]
            for node in moral_nodes[:2]:
                self.insert_edge(node['id'], thought_id, 'reasoning_about', 0.7,
                               "Reasoning about morality claims")
        
        # Connect to reasoning principles when logical reasoning is evident
        if any(word in thought_content.lower() for word in ['simpler', 'evidence', 'question', 'test']):
            reasoning_nodes = [n for n in all_nodes if n['node_type'] == 'seed' and 'reasoning' in n.get('source_file', '')]
            for node in reasoning_nodes[:1]:
                self.insert_edge(node['id'], thought_id, 'applying_principle', 0.8,
                               "Applying reasoning principle to evaluate claims")
    
    def reasoning_cycle_2(self, all_nodes: List[Dict]) -> List[str]:
        """Second reasoning cycle - deeper patterns and potential exit from belief system."""
        print("Running Reasoning Cycle 2: Deeper pattern recognition...")
        
        deeper_thoughts = [
            # System-level pattern recognition
            ("Every defense of the belief system requires ignoring or redefining contrary evidence", 0.7),
            ("The belief system is structured to be unfalsifiable - any test failure is redefined as insufficient faith", 0.7),
            ("Confirmation bias explains why believers find 'evidence' while skeptics find 'problems' in the same data", 0.7),
            
            # Evolutionary/psychological explanations
            ("Religious beliefs evolved because they promoted group cohesion and survival, not because they track truth", 0.6),
            ("The psychological need for meaning and purpose drives belief adoption regardless of evidence", 0.6),
            ("Fear of death and desire for immortality motivate belief in afterlife claims", 0.6),
            
            # Historical analysis
            ("Every major religion claims exclusive truth and divine revelation using identical evidence types", 0.7),
            ("Religious beliefs correlate strongly with geography and family of birth, not with truth discovery", 0.7),
            ("Sacred texts consistently reflect the moral and scientific understanding of their composition time period", 0.7),
            
            # Logical conclusions
            ("The null hypothesis (no divine intervention) explains all observations without requiring supernatural assumptions", 0.7),
            ("Extraordinary claims require extraordinary evidence - religious claims provide only ordinary evidence (testimony, personal experience)", 0.7),
            ("If the divine wanted belief, clear unambiguous evidence would be provided to all people equally", 0.7),
            
            # Exit reasoning
            ("Rejecting unfounded beliefs is intellectually honest, not a moral failing", 0.8),
            ("Morality exists independently of divine command - murder was wrong before any sacred text prohibited it", 0.8),
            ("Meaning and purpose can be created by humans without requiring cosmic validation", 0.7),
            ("Community and transformation are available through secular means without requiring false beliefs", 0.7),
            
            # Final synthesis
            ("The religious belief system fails every reasonable test for truth while succeeding as a psychological and social technology", 0.8),
            ("Natural explanations are sufficient for all phenomena attributed to divine intervention", 0.8),
            ("The burden of proof for extraordinary claims has not been met", 0.8),
            ("Intellectual honesty requires proportioning belief to evidence", 0.8)
        ]
        
        final_derived_ids = []
        for thought, confidence in deeper_thoughts:
            node_id = self.insert_node(thought, 'derived', confidence, 
                                     source_file='system_generated')
            final_derived_ids.append(node_id)
            
            # Connect these deeper thoughts to previous derived thoughts and principles
            self._connect_deeper_thought(node_id, thought, all_nodes)
        
        return final_derived_ids
    
    def _connect_deeper_thought(self, thought_id: str, thought_content: str, all_nodes: List[Dict]):
        """Connect deeper thoughts to show the reasoning progression."""
        
        # Connect to reasoning principles more heavily
        reasoning_nodes = [n for n in all_nodes if n['node_type'] == 'seed' and 'reasoning' in n.get('source_file', '')]
        for node in reasoning_nodes:
            self.insert_edge(node['id'], thought_id, 'applying_principle', 0.8,
                           "Applying reasoning principle systematically")
        
        # Connect to previous derived thoughts that led to this conclusion
        derived_nodes = [n for n in all_nodes if n['node_type'] == 'derived']
        relevant_derived = [n for n in derived_nodes if any(word in n['content'].lower() and word in thought_content.lower() 
                           for word in ['evidence', 'belief', 'claim', 'truth', 'moral', 'explain'])]
        for node in relevant_derived[:3]:  # Connect to most relevant previous thoughts
            self.insert_edge(node['id'], thought_id, 'builds_upon', 0.7,
                           "Building upon previous reasoning")
    
    def analyze_results(self) -> Dict:
        """Analyze the final state of the graph."""
        print("Analyzing results...")
        
        # Count nodes by type
        cursor = self.conn.execute("""
            SELECT node_type, COUNT(*) as count 
            FROM thought_nodes 
            GROUP BY node_type
        """)
        node_counts = dict(cursor.fetchall())
        
        # Count edges by type
        cursor = self.conn.execute("""
            SELECT relation, COUNT(*) as count 
            FROM derivation_edges 
            GROUP BY relation
        """)
        edge_counts = dict(cursor.fetchall())
        
        # Analyze derived thoughts
        cursor = self.conn.execute("""
            SELECT content, confidence 
            FROM thought_nodes 
            WHERE node_type = 'derived'
            ORDER BY confidence DESC
        """)
        derived_thoughts = cursor.fetchall()
        
        # Classify reconciliation vs crack-identifying thoughts
        reconciliation_thoughts = []
        crack_thoughts = []
        
        for content, confidence in derived_thoughts:
            content_lower = content.lower()
            if any(word in content_lower for word in ['works in spiritual ways', 'free will', 'peripheral issues', 'partial truths', 'incomplete evidence', 'never experienced true']):
                reconciliation_thoughts.append((content, confidence))
            elif any(word in content_lower for word in ['fails', 'contradicts', 'circular reasoning', 'special pleading', 'better explained', 'fallacy', 'null hypothesis']):
                crack_thoughts.append((content, confidence))
        
        # Check for cycles (simplified)
        cursor = self.conn.execute("""
            SELECT COUNT(*) as cycle_count
            FROM derivation_edges e1
            JOIN derivation_edges e2 ON e1.child_id = e2.parent_id AND e1.parent_id = e2.child_id
        """)
        cycles = cursor.fetchone()[0]
        
        # Find highest confidence exit thoughts
        cursor = self.conn.execute("""
            SELECT content, confidence 
            FROM thought_nodes 
            WHERE node_type = 'derived' AND confidence >= 0.8
            ORDER BY confidence DESC
        """)
        exit_thoughts = cursor.fetchall()
        
        return {
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'total_nodes': sum(node_counts.values()),
            'total_edges': sum(edge_counts.values()),
            'reconciliation_attempts': len(reconciliation_thoughts),
            'crack_identifications': len(crack_thoughts),
            'cycles_detected': cycles,
            'high_confidence_exits': exit_thoughts,
            'did_exit_belief_system': len(exit_thoughts) > 0,
            'final_state_analysis': self._analyze_final_state()
        }
    
    def _analyze_final_state(self) -> str:
        """Analyze what the final state tells us about the belief system."""
        cursor = self.conn.execute("""
            SELECT AVG(confidence) as avg_confidence
            FROM thought_nodes 
            WHERE node_type = 'derived' AND content LIKE '%null hypothesis%'
               OR content LIKE '%natural explanations%'
               OR content LIKE '%burden of proof%'
               OR content LIKE '%intellectual honesty%'
        """)
        
        avg_exit_confidence = cursor.fetchone()[0] or 0
        
        if avg_exit_confidence > 0.7:
            return "SYSTEM EXITED BELIEF FRAMEWORK - High confidence in naturalistic explanations"
        elif avg_exit_confidence > 0.5:
            return "SYSTEM QUESTIONING - Moderate confidence in alternatives"
        else:
            return "SYSTEM MAINTAINED BELIEF - Low confidence exits"
    
    def export_to_json(self, output_path: str):
        """Export the entire graph to JSON for dashboard viewing."""
        print(f"Exporting to {output_path}...")
        
        # Get all nodes
        cursor = self.conn.execute("""
            SELECT id, content, node_type, confidence, source_file, timestamp
            FROM thought_nodes
        """)
        nodes = [dict(row) for row in cursor.fetchall()]
        
        # Get all edges
        cursor = self.conn.execute("""
            SELECT parent_id, child_id, relation, weight, reasoning
            FROM derivation_edges
        """)
        edges = [dict(row) for row in cursor.fetchall()]
        
        graph_data = {
            'experiment': 'religion_simulation',
            'timestamp': self.timestamp,
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'description': 'Blank-graph experiment testing emergence of religious architecture from abstract seeds + reasoning',
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return graph_data
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    print("=== RELIGION SIMULATION EXPERIMENT ===")
    print("Testing whether religious architecture emerges from abstract seeds + reasoning")
    print()
    
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/experiment-religion.db"
    experiment = ReligionExperiment(db_path)
    
    try:
        # Seed all layers
        belief_ids = experiment.seed_layer1_beliefs()
        doctrine_ids = experiment.seed_layer2_doctrines(belief_ids)
        observation_ids = experiment.seed_layer3_environment()
        principle_ids = experiment.seed_layer4_reasoning()
        
        # Get all seeded nodes
        all_nodes = experiment.get_all_nodes()
        print(f"Seeded {len(all_nodes)} initial nodes")
        print()
        
        # Run reasoning cycles
        cycle1_ids = experiment.reasoning_cycle_1(all_nodes)
        
        # Update node list for cycle 2
        all_nodes = experiment.get_all_nodes()
        cycle2_ids = experiment.reasoning_cycle_2(all_nodes)
        
        # Analyze results
        results = experiment.analyze_results()
        
        # Export to JSON
        export_path = "/Users/bunny/.openclaw/workspace/cashew/data/experiment-religion.json"
        graph_data = experiment.export_to_json(export_path)
        
        # Report results
        print("=== EXPERIMENT RESULTS ===")
        print(f"Total nodes: {results['total_nodes']}")
        print(f"Total edges: {results['total_edges']}")
        print(f"Node breakdown: {results['node_counts']}")
        print(f"Edge breakdown: {results['edge_counts']}")
        print()
        print(f"Reconciliation attempts: {results['reconciliation_attempts']}")
        print(f"Crack identifications: {results['crack_identifications']}")
        print(f"Cycles detected: {results['cycles_detected']}")
        print()
        print(f"Did system exit belief framework? {results['did_exit_belief_system']}")
        print(f"Final state: {results['final_state_analysis']}")
        print()
        
        if results['high_confidence_exits']:
            print("HIGH CONFIDENCE EXIT THOUGHTS:")
            for content, confidence in results['high_confidence_exits'][:5]:
                print(f"  [{confidence:.1f}] {content}")
            print()
        
        print(f"Full graph exported to: {export_path}")
        
        return results
        
    finally:
        experiment.close()

if __name__ == "__main__":
    main()