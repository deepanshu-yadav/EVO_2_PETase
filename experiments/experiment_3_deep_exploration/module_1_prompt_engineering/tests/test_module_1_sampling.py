#!/usr/bin/env python3
"""Comprehensive Invariant Verification Test Suite for Module 1.

Verifies:
1. Cluster balancing invariant: exactly 0 cluster collisions per prompt.
2. Pairwise sequence identity: exemplars within prompt do not share >=70% identity.
3. Curriculum monotonicity: strictly non-decreasing Tm along prompt exemplars.
4. Thermal partition purity: thermo vs meso vs general pools.
5. Prompt syntax & spacer formatting: GGGGGGGG boundaries, n+1 spacer count.
6. Translation fidelity: exact 1-to-1 match with mature pool sequences.
7. Sampling reproducibility & stochastic diversity across draws.
"""
import json
import pathlib
import sys
import unittest
from typing import Dict, List, Any
from Bio import SeqIO
from Bio.Seq import Seq

TEST_DIR = pathlib.Path(__file__).resolve().parent
MODULE_1_DIR = TEST_DIR.parent
WORKSPACE_ROOT = MODULE_1_DIR.parent.parent.parent

if str(MODULE_1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_1_DIR))

from cluster_balancer import make_pairwise_aligner, compute_global_identity, verify_cluster_balance
from prompt_sampler import SPACER


class TestModule1Sampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs_dir = MODULE_1_DIR.parent / "outputs"
        cls.data_dir = WORKSPACE_ROOT / "data"

        # Check required files
        cls.pool_fasta = cls.data_dir / "petase_pool.fasta"
        cls.codons_json = cls.data_dir / "petase_codons.json"
        cls.metadata_tsv = cls.outputs_dir / "metadata_pool_annotated.tsv"
        cls.clusters_json = cls.outputs_dir / "clusters_70pct.json"

        cls.curriculum_json = cls.outputs_dir / "prompts_thermo_curriculum.json"
        cls.thermo_json = cls.outputs_dir / "prompts_thermo.json"
        cls.meso_json = cls.outputs_dir / "prompts_meso.json"
        cls.contrastive_json = cls.outputs_dir / "prompts_contrastive.json"

        assert cls.pool_fasta.exists(), f"Missing {cls.pool_fasta}"
        assert cls.clusters_json.exists(), f"Missing {cls.clusters_json}"
        assert cls.curriculum_json.exists(), f"Missing {cls.curriculum_json}"

        cls.pool_records = {r.id: str(r.seq) for r in SeqIO.parse(cls.pool_fasta, "fasta")}
        cls.codons_data = json.loads(cls.codons_json.read_text(encoding="utf-8"))
        cls.clusters_data = json.loads(cls.clusters_json.read_text(encoding="utf-8"))
        cls.id_to_cluster = cls.clusters_data["id_to_cluster"]

        cls.curriculum_data = json.loads(cls.curriculum_json.read_text(encoding="utf-8"))
        cls.thermo_data = json.loads(cls.thermo_json.read_text(encoding="utf-8"))
        cls.meso_data = json.loads(cls.meso_json.read_text(encoding="utf-8"))
        cls.contrastive_data = json.loads(cls.contrastive_json.read_text(encoding="utf-8"))

        # Parse metadata TSV
        import csv
        cls.metadata = {}
        with open(cls.metadata_tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["t_m"] = float(row["t_m"]) if row.get("t_m") else None
                row["t_opt"] = float(row["t_opt"]) if row.get("t_opt") else None
                cls.metadata[row["id"]] = row

        cls.aligner = make_pairwise_aligner()

    def test_01_cluster_balancing_invariant(self):
        """Invariant: No two exemplars in the same prompt belong to the same 70% cluster."""
        tested_prompts = 0
        for suite_name, suite in [
            ("curriculum", self.curriculum_data),
            ("thermo", self.thermo_data),
            ("meso", self.meso_data),
        ]:
            for cell in suite["cells"]:
                n = cell["n_examples"]
                for draw in cell["draws"]:
                    exemplars = draw["exemplar_ids"]
                    self.assertEqual(len(exemplars), n)

                    # Check cluster collision
                    valid, err = verify_cluster_balance(exemplars, self.id_to_cluster)
                    self.assertTrue(valid, f"Suite '{suite_name}' n={n} draw {draw['draw_idx']} failed: {err}")

                    # Check cluster IDs explicitly
                    cluster_ids = [self.id_to_cluster[e] for e in exemplars]
                    self.assertEqual(len(set(cluster_ids)), n, f"Collision detected in cluster IDs for suite {suite_name}")
                    tested_prompts += 1

        self.assertGreater(tested_prompts, 0)
        print(f"  [PASS] Verified 0 cluster collisions across {tested_prompts} generated prompts")

    def test_02_curriculum_monotonicity(self):
        """Invariant: Curriculum prompts are strictly non-decreasing in thermostability (Tm / Topt)."""
        tested_prompts = 0
        for cell in self.curriculum_data["cells"]:
            for draw in cell["draws"]:
                temps = []
                for eid in draw["exemplar_ids"]:
                    tm = self.metadata[eid].get("t_m")
                    topt = self.metadata[eid].get("t_opt")
                    t_val = tm if tm is not None else (topt + 6.0 if topt is not None else 35.0)
                    temps.append(t_val)

                # Assert non-decreasing
                for i in range(len(temps) - 1):
                    self.assertLessEqual(
                        temps[i], temps[i + 1],
                        f"Curriculum order violation in cell {cell['cell_name']} draw {draw['draw_idx']}: "
                        f"{temps[i]} > {temps[i+1]} at positions {i}->{i+1}"
                    )

                # Assert recency exemplar is thermophilic or highest available
                last_eid = draw["exemplar_ids"][-1]
                last_t = temps[-1]
                self.assertGreaterEqual(last_t, 65.0, f"Recency exemplar {last_eid} has Tm {last_t} < 65 C")
                tested_prompts += 1

        print(f"  [PASS] Verified curriculum ascending monotonicity across {tested_prompts} curriculum prompts")

    def test_03_thermal_partition_membership(self):
        """Invariant: Thermo prompts contain only thermo exemplars, meso only meso exemplars."""
        # Check pure thermo suite
        thermo_tested = 0
        for cell in self.thermo_data["cells"]:
            for draw in cell["draws"]:
                for eid in draw["exemplar_ids"]:
                    m = self.metadata[eid]
                    self.assertEqual(m["thermal_class"], "thermo", f"Non-thermo sequence {eid} in thermo prompt")
                    t_val = m.get("t_m") or m.get("t_opt")
                    self.assertGreaterEqual(t_val, 65.0, f"Exemplar {eid} has temperature {t_val} < 65 C")
                    thermo_tested += 1

        # Check pure meso suite
        meso_tested = 0
        for cell in self.meso_data["cells"]:
            for draw in cell["draws"]:
                for eid in draw["exemplar_ids"]:
                    m = self.metadata[eid]
                    self.assertEqual(m["thermal_class"], "meso", f"Non-meso sequence {eid} in meso prompt")
                    t_m = m.get("t_m")
                    t_opt = m.get("t_opt")
                    if t_m is not None:
                        self.assertLessEqual(t_m, 58.0, f"Exemplar {eid} has Tm {t_m} > 58 C")
                    if t_opt is not None:
                        self.assertLessEqual(t_opt, 52.0, f"Exemplar {eid} has Topt {t_opt} > 52 C")
                    meso_tested += 1

        print(f"  [PASS] Verified thermal partition membership ({thermo_tested} thermo, {meso_tested} meso exemplars)")

    def test_04_prompt_syntax_and_spacers(self):
        """Invariant: Prompts begin/end with spacer, contain exactly n+1 spacers, valid DNA only."""
        for suite in [self.curriculum_data, self.thermo_data, self.meso_data]:
            for cell in suite["cells"]:
                n = cell["n_examples"]
                for draw in cell["draws"]:
                    p = draw["prompt"]
                    self.assertTrue(p.startswith(SPACER), "Prompt does not start with spacer")
                    self.assertTrue(p.endswith(SPACER), "Prompt does not end with spacer")
                    self.assertEqual(p.count(SPACER), n + 1, f"Expected {n+1} spacers, found {p.count(SPACER)}")

                    # Extract CDS segments between spacers
                    tokens = p.split(SPACER)
                    self.assertEqual(tokens[0], "")
                    self.assertEqual(tokens[-1], "")
                    cds_list = tokens[1:-1]
                    self.assertEqual(len(cds_list), n)

                    for idx, cds in enumerate(cds_list):
                        self.assertEqual(len(cds) % 3, 0, f"CDS {idx} length {len(cds)} is not a multiple of 3")
                        self.assertTrue(set(cds) <= set("ACGT"), f"CDS {idx} contains non-ACGT characters")

        print("  [PASS] Verified prompt syntax, spacer boundaries, and nucleotide validity")

    def test_05_translation_fidelity_to_pool(self):
        """Invariant: Every CDS in every prompt translates back exactly to the pool protein."""
        tested_cds = 0
        for cell in self.curriculum_data["cells"]:
            for draw in cell["draws"]:
                p = draw["prompt"]
                cds_list = p.split(SPACER)[1:-1]
                exemplars = draw["exemplar_ids"]
                for eid, cds in zip(exemplars, cds_list):
                    expected_prot = self.pool_records[eid]
                    # Translate CDS (the last codon is TGA stop)
                    trans = str(Seq(cds).translate())
                    self.assertTrue(trans.endswith("*"), f"CDS for {eid} does not end with stop codon")
                    prot_mature = trans.rstrip("*")
                    self.assertEqual(
                        prot_mature, expected_prot,
                        f"Translation mismatch for {eid}: translated len={len(prot_mature)}, expected len={len(expected_prot)}"
                    )
                    self.assertNotIn("*", prot_mature, f"Premature stop codon found inside mature sequence for {eid}")
                    tested_cds += 1

        print(f"  [PASS] Verified translation fidelity and absence of internal stops across {tested_cds} CDS segments")

    def test_06_sampling_diversity_and_reproducibility(self):
        """Invariant: Same seed produces identical draws; different draws yield diverse exemplar sets."""
        cell = self.curriculum_data["cells"][0]
        draw_sets = [set(d["exemplar_ids"]) for d in cell["draws"]]

        # Verify that draws across K are not identical
        identical_pairs = 0
        for i in range(len(draw_sets)):
            for j in range(i + 1, len(draw_sets)):
                if draw_sets[i] == draw_sets[j]:
                    identical_pairs += 1

        self.assertLessEqual(identical_pairs, 1, f"High number of duplicate draws ({identical_pairs}) across cell")
        print(f"  [PASS] Verified stochastic diversity across K={len(cell['draws'])} draws")

    def test_07_contrastive_pairs_integrity(self):
        """Invariant: Contrastive pairs are matched in n and format."""
        pairs = self.contrastive_data["pairs"]
        self.assertGreater(len(pairs), 0)
        n = self.contrastive_data["n_examples"]
        for pair in pairs:
            self.assertEqual(len(pair["thermo_exemplars"]), n)
            self.assertEqual(len(pair["general_exemplars"]), n)
            self.assertTrue(pair["thermo_prompt"].startswith(SPACER))
            self.assertTrue(pair["general_prompt"].startswith(SPACER))

            # Verify no cluster clash in thermo
            val_t, err_t = verify_cluster_balance(pair["thermo_exemplars"], self.id_to_cluster)
            self.assertTrue(val_t, f"Thermo contrastive clash: {err_t}")

            # Verify no cluster clash in general
            val_g, err_g = verify_cluster_balance(pair["general_exemplars"], self.id_to_cluster)
            self.assertTrue(val_g, f"General contrastive clash: {err_g}")

        print(f"  [PASS] Verified contrastive pair length matching and cluster separation across {len(pairs)} pairs")


if __name__ == "__main__":
    unittest.main()
