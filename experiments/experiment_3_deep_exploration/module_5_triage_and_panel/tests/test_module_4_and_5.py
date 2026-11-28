#!/usr/bin/env python3
"""Comprehensive Verification Test Suite for Experiment 3 Modules 4 & 5.

Tests:
1. Module 4: Sequence biophysical features, 3D cleft SASA via ShrakeRupley on synthetic/PDB,
   disulfide audit, 139-pool Ridge Tm calibration, and solubility score bounds.
2. Module 5: Stage 1 hard binary elimination gate & explicit drop reasons.
3. Module 5: Stage 2 Z-score composite ranking exact arithmetic.
4. Module 5: Controlled 48-target panel (12+12+8+6+4+6 = 48), zero duplicates in Arms A-D,
   exact Ser->Ala knockout verification in Panel E, and exact 6 reference standards in Panel F.
"""
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from Bio import SeqIO

TEST_DIR = pathlib.Path(__file__).resolve().parent
MOD5_DIR = TEST_DIR.parent
EXP3_DIR = MOD5_DIR.parent
MOD4_DIR = EXP3_DIR / "module_4_biophysical_proxies"
REPO_ROOT = EXP3_DIR.parent.parent

for p in (str(MOD4_DIR), str(MOD5_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from biophysical_proxies_utils import (
    audit_disulfide_topology,
    compute_cleft_interface_proxies,
    fit_pool_tm_calibrator,
    predict_ecoli_solubility,
    predict_thermostability,
)
from mod_04_biophysical_proxies import run_module_4
from mod_05_decoupled_triage import run_module_5
from triage_panel_utils import (
    apply_stage1_hard_gate,
    build_engineered_benchmarks,
    compute_zscore_fitness,
)


class TestModules4And5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs_dir = EXP3_DIR / "outputs"
        cls.data_dir = REPO_ROOT / "data"
        cls.fasta_path = cls.outputs_dir / "designs_exp3_screened_novel.faa"
        cls.grid_tsv = cls.outputs_dir / "generated_continuations_grid.tsv"
        cls.reranked_tsv = cls.outputs_dir / "contrastive_reranked_designs.tsv"
        cls.pool_fasta = cls.data_dir / "petase_pool.fasta"
        cls.pool_meta = cls.outputs_dir / "metadata_pool_annotated.tsv"

        assert cls.fasta_path.exists(), f"Missing {cls.fasta_path}"
        assert cls.pool_fasta.exists(), f"Missing {cls.pool_fasta}"
        assert cls.pool_meta.exists(), f"Missing {cls.pool_meta}"

    def test_01_pool_tm_calibrator_and_benchmarks(self):
        """Pool Tm Ridge calibrator reproduces thermo > meso trend and valid benchmarks."""
        cal = fit_pool_tm_calibrator(self.pool_fasta, self.pool_meta)
        self.assertIsNotNone(cal)
        self.assertEqual(cal["n_samples"], 50)
        self.assertLess(cal["train_mae"], 12.0)

        bms = build_engineered_benchmarks(self.pool_fasta)
        self.assertEqual(len(bms), 6)
        bm_ids = [b["id"] for b in bms]
        self.assertEqual(
            bm_ids,
            ["003_LCC", "003_LCC_ICCG", "004_PHL7", "002_TfCut2", "001_IsPETase", "001_IsPETase_FastPETase"],
        )
        # Verify ICCG differs from LCC by exactly 4 mutations and Fast-PETase from IsPETase by 5
        lcc = next(b["sequence"] for b in bms if b["id"] == "003_LCC")
        iccg = next(b["sequence"] for b in bms if b["id"] == "003_LCC_ICCG")
        self.assertEqual(sum(a != b for a, b in zip(lcc, iccg)), 4)

        isp = next(b["sequence"] for b in bms if b["id"] == "001_IsPETase")
        fast = next(b["sequence"] for b in bms if b["id"] == "001_IsPETase_FastPETase")
        self.assertEqual(sum(a != b for a, b in zip(isp, fast)), 5)

        # Benchmarks should pass solubility gate >= 0.45
        for bm in bms:
            sol = predict_ecoli_solubility(bm["sequence"], mean_plddt=95.0)
            self.assertGreaterEqual(sol["solubility_score"], 0.45, f"Benchmark {bm['id']} failed solubility")
            tm = predict_thermostability(bm["sequence"], cal, has_cterm_disulfide=(bm["id"] != "001_IsPETase"))
            self.assertTrue(35.0 <= tm["predicted_tm"] <= 90.0)

    def test_02_3d_cleft_sasa_and_disulfide_pdb(self):
        """3D ShrakeRupley cleft SASA & disulfide SG-SG scan work on a valid PDB file."""
        # Minimal 3-residue PDB (SER, TRP, CYS-CYS disulfide)
        pdb_text = (
            "ATOM      1  N   SER A 160       0.000   0.000   0.000  1.00 90.00           N\n"
            "ATOM      2  CA  SER A 160       1.458   0.000   0.000  1.00 90.00           C\n"
            "ATOM      3  C   SER A 160       2.000   1.400   0.000  1.00 90.00           C\n"
            "ATOM      4  O   SER A 160       1.200   2.300   0.000  1.00 90.00           O\n"
            "ATOM      5  CB  SER A 160       2.000  -0.800   1.200  1.00 90.00           C\n"
            "ATOM      6  OG  SER A 160       3.400  -0.800   1.200  1.00 90.00           O\n"
            "ATOM      7  N   TRP A 159       5.000   0.000   0.000  1.00 90.00           N\n"
            "ATOM      8  CA  TRP A 159       6.400   0.000   0.000  1.00 90.00           C\n"
            "ATOM      9  CB  TRP A 159       6.800   1.400   0.000  1.00 90.00           C\n"
            "ATOM     10  N   ARG A 161       5.000   3.000   0.000  1.00 90.00           N\n"
            "ATOM     11  CA  ARG A 161       6.400   3.000   0.000  1.00 90.00           C\n"
            "ATOM     12  SG  CYS A 274      20.000   0.000   0.000  1.00 90.00           S\n"
            "ATOM     13  SG  CYS A 290      22.050   0.000   0.000  1.00 90.00           S\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "test.pdb"
            p.write_text(pdb_text, encoding="utf-8")
            cleft = compute_cleft_interface_proxies(p, "A" * 295, ser_og_coord=[3.4, -0.8, 1.2], cleft_radius=8.0)
            self.assertGreater(cleft["cleft_hydrophobic_sasa"], 0.0)
            self.assertEqual(cleft["cleft_hp_res_count"], 1)
            self.assertEqual(cleft["net_charge_cleft"], 1.0)
            self.assertIn(cleft["cleft_sasa_method"], {"freesasa", "biopython_shrake_rupley"})

            ds = audit_disulfide_topology(p, "A" * 295)
            self.assertEqual(ds["disulfide_count"], 1)
            self.assertTrue(ds["has_cterm_disulfide"])

    def test_03_stage1_elimination_and_stage2_zscore_math(self):
        """Stage 1 eliminates failing designs cleanly and Stage 2 matches exact weighted Z-scores."""
        df = pd.DataFrame([
            {
                "id": "pass_1", "sequence": "M" + "A" * 100 + "GHSMG" + "D" + "H" + "A" * 150,
                "length": 258, "mean_plddt": 90.0, "solubility_score": 0.65,
                "tm_ref": 0.96, "2he_mhet_affinity": -6.5, "3pet_affinity": -7.5,
                "2he_mhet_nac_score": 0.9, "3pet_nac_score": 0.7,
                "2he_mhet_attack_dist": 3.1, "3pet_attack_dist": 3.3,
                "predicted_tm": 74.0, "cleft_hydrophobic_sasa": 620.0,
            },
            {
                "id": "pass_2", "sequence": "M" + "A" * 100 + "GHSMG" + "D" + "H" + "A" * 150,
                "length": 258, "mean_plddt": 84.0, "solubility_score": 0.55,
                "tm_ref": 0.90, "2he_mhet_affinity": -5.5, "3pet_affinity": -6.5,
                "2he_mhet_nac_score": 0.5, "3pet_nac_score": 0.5,
                "2he_mhet_attack_dist": 3.8, "3pet_attack_dist": 4.0,
                "predicted_tm": 68.0, "cleft_hydrophobic_sasa": 520.0,
            },
            {
                "id": "fail_low_plddt", "sequence": "M" + "A" * 100 + "GHSMG" + "D" + "H" + "A" * 150,
                "length": 258, "mean_plddt": 72.0, "solubility_score": 0.70,
                "tm_ref": 0.95, "2he_mhet_affinity": -7.0, "3pet_affinity": -7.0,
                "2he_mhet_nac_score": 0.8, "3pet_nac_score": 0.8,
                "2he_mhet_attack_dist": 3.0, "3pet_attack_dist": 3.0,
                "predicted_tm": 75.0, "cleft_hydrophobic_sasa": 600.0,
            },
            {
                "id": "fail_insoluble_short", "sequence": "M" * 120 + "GHSMG" + "D" + "H",
                "length": 127, "mean_plddt": 92.0, "solubility_score": 0.30,
                "tm_ref": 0.95, "2he_mhet_affinity": -6.0, "3pet_affinity": -6.0,
                "2he_mhet_nac_score": 0.7, "3pet_nac_score": 0.7,
                "2he_mhet_attack_dist": 3.0, "3pet_attack_dist": 3.0,
                "predicted_tm": 72.0, "cleft_hydrophobic_sasa": 550.0,
            },
        ])
        gated = apply_stage1_hard_gate(df)
        self.assertEqual(int(gated["passed_stage1_gate"].sum()), 2)
        reasons = dict(zip(gated["id"], gated["stage1_drop_reasons"]))
        self.assertEqual(reasons["fail_low_plddt"], "plddt<80")
        self.assertEqual(reasons["fail_insoluble_short"], "length_out_of_241_306;solubility<0.45")

        ranked = compute_zscore_fitness(gated)
        # With exactly 2 symmetric passers, Z-scores are +1.0 for pass_1 and -1.0 for pass_2
        row1 = ranked[ranked["id"] == "pass_1"].iloc[0]
        row2 = ranked[ranked["id"] == "pass_2"].iloc[0]
        row_fail = ranked[ranked["id"] == "fail_low_plddt"].iloc[0]
        self.assertAlmostEqual(float(row1["fitness_score"]), 1.0, places=4)
        self.assertAlmostEqual(float(row2["fitness_score"]), -1.0, places=4)
        self.assertTrue(pd.isna(row_fail["fitness_score"]))

    def test_04_end_to_end_140_candidates_and_48_panel(self):
        """End-to-end Module 4 + Module 5 run on all 140 novel Exp 3 designs produces exact 48-target panel."""
        with open(self.fasta_path, "r", encoding="utf-8") as fh:
            seqs = list(SeqIO.parse(fh, "fasta"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            # Create realistic synthetic Module 3 inputs for all 140 designs to test integration
            rng = np.random.default_rng(123)
            plddt_rows = []
            triad_rows = []
            dock_rows = []
            fs_lines = []
            for i, rec in enumerate(seqs):
                plddt = 76.0 if i == 0 else float(rng.uniform(83.0, 94.5))
                plddt_rows.append({"id": rec.id, "mean_plddt": round(plddt, 2), "passed_plddt80": plddt >= 80.0})
                triad_rows.append({
                    "id": rec.id,
                    "seq_triad_intact": True,
                    "triad_intact": i % 15 != 0,
                    "d_ser_his": 2.95,
                    "d_his_asp": 2.82,
                    "has_oxyanion_tyr": True,
                    "has_groove_w159": i % 2 == 0,
                    "has_groove_w185": True,
                    "active_site_complete": i % 15 != 0,
                })
                d2 = float(rng.uniform(2.7, 4.2))
                d3 = float(rng.uniform(2.8, 4.4))
                dock_rows.append({
                    "id": rec.id,
                    "2he_mhet_affinity": round(float(rng.uniform(-6.8, -4.6)), 2),
                    "2he_mhet_attack_dist": round(d2, 2),
                    "2he_mhet_nac_score": round(1.0 / (1.0 + max(0.0, d2 - 3.2)), 3),
                    "2he_mhet_is_nac": d2 <= 3.5,
                    "3pet_affinity": round(float(rng.uniform(-7.6, -5.1)), 2),
                    "3pet_attack_dist": round(d3, 2),
                    "3pet_nac_score": round(1.0 / (1.0 + max(0.0, d3 - 3.2)), 3),
                    "3pet_is_nac": d3 <= 3.5,
                })
                tm6 = round(float(rng.uniform(0.91, 0.98)), 4)
                tm4 = round(float(rng.uniform(0.90, 0.98)), 4)
                fs_lines.append(f"{rec.id}_relaxed.pdb\t6EQE.pdb\t0.55\t255\t100\t1\t1\t255\t1\t255\t1e-50\t400\t{tm6}\t1.2\t0.9")
                fs_lines.append(f"{rec.id}_relaxed.pdb\t4EB0.pdb\t0.60\t255\t90\t1\t1\t255\t1\t255\t1e-55\t420\t{tm4}\t1.1\t0.9")

            plddt_p = tmp_dir / "esmfold_plddt_summary.tsv"
            triad_p = tmp_dir / "catalytic_triad_geometry_report.tsv"
            dock_p = tmp_dir / "smina_flexible_docking_results.tsv"
            fs_p = tmp_dir / "foldseek_vs_references.m8"
            pd.DataFrame(plddt_rows).to_csv(plddt_p, sep="\t", index=False)
            pd.DataFrame(triad_rows).to_csv(triad_p, sep="\t", index=False)
            pd.DataFrame(dock_rows).to_csv(dock_p, sep="\t", index=False)
            fs_p.write_text("\n".join(fs_lines) + "\n", encoding="utf-8")

            df_m4 = run_module_4(
                fasta_path=self.fasta_path,
                reranked_tsv=self.reranked_tsv,
                grid_tsv=self.grid_tsv,
                pool_fasta=self.pool_fasta,
                pool_metadata_tsv=self.pool_meta,
                out_dir=tmp_dir,
                plddt_tsv=plddt_p,
                triad_tsv=triad_p,
            )
            self.assertEqual(len(df_m4), 140)

            summary_m5 = run_module_5(
                fasta_path=self.fasta_path,
                reranked_tsv=self.reranked_tsv,
                grid_tsv=self.grid_tsv,
                mod4_features_tsv=tmp_dir / "interfacial_and_stability_features.tsv",
                pool_fasta=self.pool_fasta,
                out_dir=tmp_dir,
                plddt_tsv=plddt_p,
                triad_tsv=triad_p,
                foldseek_m8=fs_p,
                docking_tsv=dock_p,
                seed=42,
            )
            self.assertEqual(summary_m5["total_candidates"], 140)
            self.assertEqual(summary_m5["panel_total"], 48)
            self.assertEqual(summary_m5["panel_arm_counts"], {
                "A_thermostable_lead": 12,
                "B_deep_novelty": 12,
                "C_ispetase_groove_rescue": 8,
                "D_random_control": 6,
                "E_catalytic_ko_S_to_A": 4,
                "F_natural_standard": 6,
            })

            panel_df = pd.read_csv(tmp_dir / "wetlab_panel_exp3_48targets.tsv", sep="\t")
            # Zero duplicate IDs in Arms A-D
            ad_ids = panel_df[panel_df["panel"].isin([
                "A_thermostable_lead", "B_deep_novelty", "C_ispetase_groove_rescue", "D_random_control"
            ])]["id"].tolist()
            self.assertEqual(len(ad_ids), 38)
            self.assertEqual(len(set(ad_ids)), 38)

            # Panel E knockouts differ from corresponding Panel A parents by exactly 1 S->A substitution
            pa_seqs = dict(zip(
                panel_df[panel_df["panel"] == "A_thermostable_lead"]["id"],
                panel_df[panel_df["panel"] == "A_thermostable_lead"]["sequence"],
            ))
            for _, ko_row in panel_df[panel_df["panel"] == "E_catalytic_ko_S_to_A"].iterrows():
                parent_id = ko_row["id"].split("_KO_")[0]
                p_seq = pa_seqs[parent_id]
                diffs = [(a, b) for a, b in zip(p_seq, ko_row["sequence"]) if a != b]
                self.assertEqual(diffs, [("S", "A")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
