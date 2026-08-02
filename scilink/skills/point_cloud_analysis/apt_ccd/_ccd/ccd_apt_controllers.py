"""
CCD APT Analysis Controllers

This module contains unified controllers that handle both single sample (n=1)
and batch (n>1) analysis identically. The key principle is:

    Single sample = Batch of 1

All controllers adapt their behavior based on state["is_single_sample"]
and state["num_samples"], but use the same code paths.
"""

import subprocess
import json
import logging
import io
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Dict
import numpy as np
import pandas as pd

from PIL import Image

from .ccd import detect_compositional_communities


# ============================================================================
# CCD ANALYSIS INSTRUCTIONS
# ============================================================================

CCD_PARAMETER_ESTIMATION_INSTRUCTIONS = """You are an expert assistant analyzing APT neighborhoods to determine optimal parameters for a subsequent analysis technique called Compositional Community Detection (CCD).

**How CCD Works:**
1.  **Ion Filtering:** Ions in the APT neighborhoods are first filtered to remove ion species that are not of interest or are not easily quantified by APT (e.g. O1, O1H1, etc.) to create feature space for each neighborhood consisting of their compositional distribution.
2.  **k-Means Clustering:** The neighborhoods are first clustered into k clusters based on their compositional similarity. This step identifies distinct clusters of neighborhoods that often correspond to different phases, defects, or structural motifs in the material.
3.  **KS Statistics:** Each cluster is characterized by the Kolmogorov-Smirnov (KS) statistic, which quantifies how much the compositional distribution of that cluster deviates from a random distribution. Clusters with high KS values are more likely to represent meaningful structural features.
4.  **Community Detection:** A community detection algorithm is applied to the clusters to identify groups of clusters that are compositionally similar, which can reveal higher-level structural organization in the material.

**Your Task:**
Based on the provided neighborhood data and its metadata, estimate the optimal values for three key parameters for this CCD analysis:

1.  **`k_values` (List of Integers):** The k values for the k-means clustering step.
    * **Guidance:** Choose a values that are appropriate for the compositional complexity of the neighborhoods. If the neighborhoods show a few distinct compositional motifs, a smaller k (e.g. 2-4) may be sufficient. If there are many different motifs or phases, a larger k (e.g. 5-10) may be needed to capture the diversity. Consider the trade-off between too few clusters (which may merge distinct features) and too many clusters (which may split meaningful features into multiple clusters). You can also suggest multiple k values to explore in the analysis.
    * **Constraints:** Suggest a list of integers.

2.  **`q` (Integer):** The percentaile threshold for selecting clusters based on their KS statistic. Only clusters with KS values above the q-th percentile will be considered significant and included in the community detection step.
    * **Guidance:** Choose a value between 0 and 100 that balances sensitivity and specificity. A lower q (e.g. 50) will include more clusters, which may capture more subtle features but also include more noise. A higher q (e.g. 90) will focus on the most distinct clusters, which may highlight the most important features but miss some relevant ones. Consider the overall distribution of KS values across clusters when choosing this threshold.
    * **Constraints:** Suggest an integer.

3.  **`ignore_ions` (List of Strings):** A list of ion species to ignore in the analysis. These are typically ions that are not reliably detected by APT or are not relevant to the material system being studied (e.g. O1, O1H1, etc.). Ignoring these ions can help reduce noise and improve the quality of the clustering. The list of ions present in the neighborhoods is provided in the metadata, and you should choose which ones to ignore based on their relevance and detectability.

4.  **`explanation` (String):** Provide a brief explanation for your choice of `k_values` and `q`, referencing specific features visible in the neighborhoods or general compositional complexity, ideally in the context of this specific material system.


**Output Format:**
Provide your response ONLY as a valid JSON object containing the keys "k_values", "q", "ignore_ions", and "explanation" with integer values. Do not include any other text, explanations, or markdown formatting.

"""


CCD_ANALYSIS_INSTRUCTIONS = '''You are an expert assistant analyzing APT neighborhoods using Compositional Community Detection (CCD).


You will receive:
1. Analysis statistics (communities trends, KS statistics)
2. Visualizations (KS statistics)

Your goal is to provide scientific interpretation and formulate precise claims for literature search.

## Required Output
Return a JSON object with:

```json
{
    "methodology_notes": "Brief description of the analysis and notable aspects of this dataset",
    
    "detailed_analysis": "Thorough analysis correlating CCD communities with KS statistics. Identify: precipitates, phases, defects, and other structural motifs.",
    
    "component_interpretations": [
        {
            "index": 1,
            "spectral_features": "What you see in the CCD results for this component (e.g. enriched in Ga, depleted in Fe and Cr, etc.)",
            "physical_meaning": "What physical motif this represents",
            "confidence": "high/medium/low"
        }
    ],
    
    "visualization_descriptions": [
        {
            "name": "exact_filename_without_extension",
            "description": "What this visualization shows and its significance"
        }
    ],
    
    "scientific_claims": [
        {
            "claim": "A single, focused scientific claim about a specific observation.",
            "scientific_impact": "Why this would be scientifically significant.",
            "has_anyone_question": "A question starting with 'Has anyone' - must be portable and understandable WITHOUT seeing the images. Do NOT use 'this', 'that', 'the observed'.",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }
    ]
}
```

## Guidelines for CCD Interpretation
- Positive KS values indicate enrichment of the associated ion species in that cluster compared to random distribution.
- Negative KS values indicate depletion of the associated ion species in that cluster compared to random distribution.
- KS values near zero indicate no significant difference from random distribution for that ion species in that cluster.
- Associations between specific ion species and clusters can suggest the presence of particular phases, defects, or structural motifs in the material. For example, a cluster enriched in a certain alloying element may correspond to a precipitate phase, while a cluster enriched in oxygen may indicate an oxide inclusion or surface contamination.
- Enrichment of Ga in a cluster may indicate FIB-induced damage if Ga is not a main component of the material. 
- Depletion of matrix elements in a cluster may suggest voids, pores, or other defects.

## Guidelines for Spatial Distribution Interpretation
The ``spatial_distribution`` block quantifies how communities are arranged in 3D space:

- **segregation_score** (0–1): 0 = communities randomly intermixed throughout the volume; 1 = each community forms a fully isolated spatial cluster. Values above ~0.3 suggest meaningful regional segregation.
- **neighbor_purity_per_community**: fraction of each neighborhood's 10 nearest neighbors that share the same community label. High values (>0.5) indicate spatially coherent domains; low values indicate diffuse intermixing.
- **community_centroids_normalized**: center-of-mass per community, normalized to the dataset bounding box [0,1]³. Well-separated centroids indicate communities occupying distinct sample regions.
- **community_spatial_extent_normalized**: [dx, dy, dz] fraction of the total bounding box spanned by each community. Small extent + high purity = localized feature (precipitate, inclusion, damage zone). Large extent = widespread or matrix phase.
- **inter_centroid_distances_normalized**: pairwise distances between community centroids. Large (>0.3) = regionally segregated; small (<0.1) = co-located (possibly layered or interpenetrating phases).

Use these to identify:
- **Localized precipitates/inclusions**: high purity, small extent, centroid offset from others
- **Matrix phase**: large extent, lower purity, centroid near dataset center
- **Interfacial/layered structures**: centroid offset primarily along one axis, moderate purity
- **Homogeneous distribution**: low segregation score, similar centroids, large extents

## Guidelines for Claims
- Generate 2-4 specific, testable claims
- Avoid overly specific numbers
- "has_anyone_question" must be self-contained
'''



# ============================================================================
# INITIAL CCD ANALYSIS CONTROLLER
# ============================================================================

class InitialCCDAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Analyzes the first sample with LLM-guided parameters.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if state.get("locked_params"):
            self.logger.info("📋 Preset parameters provided, skipping initial analysis.")
            return state
        
        is_single = state.get("is_single_sample", False)
        mode_str = "SINGLE SAMPLE" if is_single else f"BATCH ({state.get('num_samples', 1)} samples)"
        self.logger.info(f"\n\n🔬 --- INITIAL CCD ANALYSIS ({mode_str}) --- 🔬\n")
        
        # Get LLM parameter suggestions
        self.logger.info("🧠 LLM Step: Reasoning about CCD parameters...")

        first_frame: "pd.DataFrame" = state["first_frame"]
        system_info = state.get("system_info", {})

        # Build text metadata from the first neighborhood DataFrame
        ion_cols = sorted([c for c in first_frame.columns if c.startswith('p')])
        ion_names = [c[1:] for c in ion_cols]  # strip leading 'p'
        neighborhood_metadata = {
            "neighborhood_count": int(len(first_frame)),
            "ion_types": ion_names,
            "mean_density": float(first_frame["density"].mean()) if "density" in first_frame.columns else None,
            "std_density": float(first_frame["density"].std()) if "density" in first_frame.columns else None,
            "mean_composition": {
                ion: float(first_frame[col].mean())
                for ion, col in zip(ion_names, ion_cols)
                if col in first_frame.columns
            },
        }

        prompt_parts = [CCD_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append(
            f"\n\nNeighborhood Metadata:\n{json.dumps(neighborhood_metadata, indent=2)}"
        )
        if system_info:
            prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(system_info, indent=2)}")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            llm_params = json.loads(response.text)
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
            
            print("\n" + "="*60)
            print("🧠 LLM REASONING")
            print(f"   Explanation: {llm_params.get('explanation', 'N/A')}")
            print(f"   Params: k_values={llm_params.get('k_values')}, q={llm_params.get('q')}, ignore_ions={llm_params.get('ignore_ions')}")
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"❌ LLM parameter estimation failed: {e}")
            llm_params = {"k_values": [4,5,6], "q": 25, "ignore_ions": [], "explanation": "Using defaults"}
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
        
        # Run CCD analysis with the suggested parameters
        self.logger.info("🛠️ Running CCD...")

        k_values = llm_params.get("k_values", [4, 5, 6])
        q = llm_params.get("q", 25)
        ignore_ions = llm_params.get("ignore_ions", [])

        neighborhood_csv = state["data_paths"][0]

        try:
            out = detect_compositional_communities(
                neighborhood_csv,
                savedir=str(self.output_dir),
                k_values=k_values,
                q=q,
                ignore_ions=ignore_ions,
                n_repeats=5,
            )
            
            state["ccd_communities"] = out["community_count"]
            state["ccd_neighborhood_counts"] = out["community_neighborhood_counts"]
            state["ccd_compositions"] = out["community_compositions"]
            
            self.logger.info(f"✅ CCD complete. {out['community_count']} communities identified.")
            
        except Exception as e:
            self.logger.error(f"❌ CCD failed: {e}")
            state["ccd_communities"] = None
            state["ccd_neighborhood_counts"] = None
            state["ccd_compositions"] = None
        
        state["first_frame_results"] = {
            "n_communities": state.get("ccd_communities"),
            "neighborhood_counts": state.get("ccd_neighborhood_counts"),
            "compositions": state.get("ccd_compositions"),
            "llm_params": llm_params
        }
        
        return state
    


# ============================================================================
# HUMAN FEEDBACK REFINEMENT CONTROLLER
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step]
    Facilitates human-in-the-loop parameter refinement for CCD parameters.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
            return state
        
        if state.get("preset_params"):
            self.logger.info("Preset parameters provided. Skipping feedback loop.")
            state["locked_params"] = state.get("preset_params")
            return state
        
        is_single = state.get("is_single_sample", False)
        mode_str = "SINGLE SAMPLE" if is_single else "BATCH"
        self.logger.info(f"\n\n👤 --- HUMAN FEEDBACK REFINEMENT ({mode_str}) --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            self._display_analysis_for_review(state, iteration)
            feedback = self._collect_human_feedback(state)
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
                break

            elif feedback["action"] == "modify" and feedback.get("params"):
                state = self._rerun_analysis(state, feedback["params"])
        
        if iteration >= self.max_refinement_iterations:
            self.logger.warning(f"⚠️ Max iterations reached. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
        
        return state
    
    def _display_analysis_for_review(self, state: dict, iteration: int) -> None:
        llm_params = state.get("llm_params", {})
        is_single = state.get("is_single_sample", False)

        # Show KS stats heatmap if available
        ks_png = self.output_dir / "KS_stats.png"

        print("\n" + "=" * 80)
        mode_str = "SINGLE SAMPLE" if is_single else f"BATCH ({state.get('num_samples', 1)} samples)"
        print(f"🔬 CCD ANALYSIS REVIEW - {mode_str} - Iteration {iteration}")
        print("=" * 80)
        if ks_png.exists():
            print(f"\n📊 KS statistics heatmap: {ks_png}")
        n_communities = state.get("ccd_communities")
        if n_communities is not None:
            print(f"\n🔢 Communities identified: {n_communities}")
        print(f"\n⚙️ Parameters: k_values={llm_params.get('k_values', 'auto')}, "
              f"q={llm_params.get('q', 25)}, "
              f"ignore_ions={llm_params.get('ignore_ions', [])}")
        if not is_single:
            print(f"\n📦 Note: These parameters will be applied to all {state.get('num_samples', 1)} samples.")
        print("-" * 80)
    
    def _collect_human_feedback(self, state: dict) -> dict:
        """Collect human feedback on the CCD analysis parameters.

        Uses a single ``input()`` call so it maps directly to the
        Streamlit UI's "Accept as-is" / "Submit feedback" buttons:
        - Empty response  → accept current parameters
        - Non-empty text  → LLM converts natural-language suggestion to params
        """
        llm_params = state.get("llm_params", {})
        print(f"\nCurrent parameters: k_values={llm_params.get('k_values', 'auto')}, "
              f"q={llm_params.get('q', 25)}, "
              f"ignore_ions={llm_params.get('ignore_ions', [])}")
        print("Press Enter to accept, or describe what to change "
              "(e.g. 'use k=[4,5,6,7]', 'raise q to 50', 'ignore O1 and O1H1').")

        try:
            user_feedback = input("\n🤔 Your feedback (or press Enter to accept): ").strip()
        except (KeyboardInterrupt, EOFError):
            return {"action": "accept"}

        if not user_feedback:
            return {"action": "accept"}

        params = self._convert_feedback_to_params(user_feedback, llm_params)
        if params:
            return {"action": "modify", "params": params}
        return {"action": "accept"}

    def _convert_feedback_to_params(self, user_feedback: str, current_params: dict) -> dict:
        """Use LLM to convert natural language feedback to CCD parameters."""
        self.logger.info("   🧠 Converting feedback to parameters...")

        prompt = f"""Convert user feedback into CCD parameter adjustments.

**Current Parameters:**
{json.dumps(current_params, indent=2)}

**Available Parameters:**
- k_values (List of integers): k values for k-means clustering (e.g. [4,5,6])
- q (integer): Percentile threshold for KS statistic (0-100)
- ignore_ions (List of strings): Ion species to ignore (e.g. ["O1", "O1H1"])

**User Feedback:**
"{user_feedback}"

Return JSON with ONLY the parameters to change:
{{"k_values": [4,5,6], "q": 25, "ignore_ions": ["O1", "O1H1"]}}
"""

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict or not result_json:
                return None

            print(f"\n   ✅ Interpreted as: {json.dumps(result_json, indent=2)}")
            return result_json

        except Exception as e:
            self.logger.error(f"Error converting feedback: {e}")
            return None
    
    def _rerun_analysis(self, state: dict, new_params: dict) -> dict:
        self.logger.info("🔄 Re-running CCD with updated parameters...")

        llm_params = state.get("llm_params", {}).copy()
        llm_params.update(new_params)
        state["llm_params"] = llm_params
        state["current_params"] = llm_params

        k_values = llm_params.get("k_values", [4, 5, 6])
        q = llm_params.get("q", 25)
        ignore_ions = llm_params.get("ignore_ions", [])
        neighborhood_csv = state["data_paths"][0]

        try:
            out = detect_compositional_communities(
                neighborhood_csv,
                savedir=str(self.output_dir),
                k_values=k_values,
                q=q,
                ignore_ions=ignore_ions,
                n_repeats=5,
            )
            state["ccd_communities"] = out["community_count"]
            state["ccd_neighborhood_counts"] = out["community_neighborhood_counts"]
            state["ccd_compositions"] = out["community_compositions"]
            state["first_frame_results"] = {
                "n_communities": out["community_count"],
                "neighborhood_counts": out["community_neighborhood_counts"],
                "compositions": out["community_compositions"],
                "llm_params": llm_params,
            }
            self.logger.info(f"✅ Re-analysis complete. {out['community_count']} communities identified.")
        except Exception as e:
            self.logger.error(f"❌ Re-analysis failed: {e}")

        return state


# ============================================================================
# UNIFIED BATCH PROCESSING CONTROLLER
# ============================================================================

class UnifiedBatchProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL samples using the refined parameters.
    Single sample = batch of 1.
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state

        num_samples = state.get("num_samples", 1)
        is_single = state.get("is_single_sample", False)

        mode_str = "SINGLE SAMPLE" if is_single else f"BATCH ({num_samples} samples)"
        self.logger.info(f"\n\n🔄 --- PROCESSING: {mode_str} --- 🔄\n")

        # For single sample, reuse results already computed by InitialCCDAnalysisController
        if is_single and state.get("ccd_communities") is not None:
            self.logger.info("📋 Using results from initial analysis (single sample).")
            state["batch_results"] = [{
                "index": 0,
                "sample_path": state.get("data_path"),
                "sample_name": state.get("first_data_name", "sample"),
                "n_communities": state["ccd_communities"],
                "community_neighborhood_counts": state["ccd_neighborhood_counts"],
                "community_compositions": state["ccd_compositions"],
                "success": True,
                "error": None,
            }]
            return state

        # Batch (or single that needs initial run): merge all CSVs, tag by source
        locked_params = state.get("locked_params", state.get("current_params", {}))
        k_values = locked_params.get("k_values", [4, 5, 6])
        q = locked_params.get("q", 25)
        ignore_ions = locked_params.get("ignore_ions", [])

        data_paths = state.get("data_paths", [])
        self.logger.info(f"📦 Processing {num_samples} CSV(s) with: k_values={k_values}, q={q}")

        # Load, tag, and concatenate all neighborhood CSVs
        frames = []
        for csv_path in data_paths:
            try:
                df = pd.read_csv(csv_path)
                df["sample_source"] = Path(csv_path).stem
                frames.append(df)
                self.logger.info(f"   Loaded {len(df)} rows from {Path(csv_path).name}")
            except Exception as e:
                self.logger.error(f"   ❌ Failed to load {csv_path}: {e}")

        if not frames:
            state["error_dict"] = {"error": "No files loaded", "details": str(data_paths)}
            return state

        merged = pd.concat(frames, ignore_index=True)
        merged_csv_path = self.output_dir / "merged_neighborhoods.csv"
        merged.to_csv(merged_csv_path, index=False)
        self.logger.info(f"   Merged DataFrame: {len(merged)} total rows → {merged_csv_path}")

        # Run CCD on merged CSV
        try:
            out = detect_compositional_communities(
                str(merged_csv_path),
                savedir=str(self.output_dir),
                k_values=k_values,
                q=q,
                ignore_ions=ignore_ions,
                n_repeats=5,
            )
            state["ccd_communities"] = out["community_count"]
            state["ccd_neighborhood_counts"] = out["community_neighborhood_counts"]
            state["ccd_compositions"] = out["community_compositions"]
            self.logger.info(f"✅ CCD complete. {out['community_count']} communities identified.")
        except Exception as e:
            self.logger.error(f"❌ CCD failed: {e}")
            state["error_dict"] = {"error": "CCD analysis failed", "details": str(e)}
            return state

        # Re-read merged CSV with community labels to compute per-source stats
        try:
            merged_result = pd.read_csv(merged_csv_path)
            batch_results = []
            for idx, csv_path in enumerate(data_paths):
                source = Path(csv_path).stem
                source_df = merged_result[merged_result["sample_source"] == source]
                if "community" in source_df.columns:
                    counts = source_df["community"].value_counts().to_dict()
                else:
                    counts = {}
                batch_results.append({
                    "index": idx,
                    "sample_name": source,
                    "sample_path": csv_path,
                    "n_communities": out["community_count"],
                    "community_neighborhood_counts": counts,
                    "community_compositions": out["community_compositions"],
                    "success": True,
                    "error": None,
                })
        except Exception as e:
            self.logger.warning(f"   Per-source stats failed: {e}")
            batch_results = [{
                "index": idx,
                "sample_name": Path(p).stem,
                "sample_path": p,
                "n_communities": out["community_count"],
                "success": True,
                "error": None,
            } for idx, p in enumerate(data_paths)]

        state["batch_results"] = batch_results
        state["batch_params"] = {
            "k_values": k_values,
            "q": q,
            "ignore_ions": ignore_ions,
            "n_samples": num_samples,
        }

        return state


# ============================================================================
# CONDITIONAL CUSTOM ANALYSIS CONTROLLER
# ============================================================================

class ConditionalCustomAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates trend analysis script for n>=2, skipped for n=1.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        if state.get("is_single_sample", False) or state.get("num_samples", 1) < 2:
            self.logger.info("\n📊 Custom trend analysis skipped (single sample mode).\n")
            state["custom_analysis_results"] = {"success": True, "skipped": True, "reason": "Single sample"}
            return state
        
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        # Generate and execute script (simplified)
        script = self._fallback_script()
        script_path = self.output_dir / "analyze_results.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(['python', str(script_path)], capture_output=True, text=True, timeout=300, cwd=str(self.output_dir))
            success = result.returncode == 0
            if success:
                print(result.stdout)
        except Exception as e:
            success = False
        
        state["custom_analysis_results"] = {"success": success, "skipped": False, "script_path": str(script_path)}
        state["analysis_script_path"] = str(script_path)
        
        return state
    
    def _fallback_script(self) -> str:
        return f'''#!/usr/bin/env python3
"""Fallback trend analysis script for batch CCD results."""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("{self.output_dir}")

def main():
    merged_csv = OUTPUT_DIR / "merged_neighborhoods.csv"
    if not merged_csv.exists():
        print("merged_neighborhoods.csv not found — skipping trend analysis")
        return

    df = pd.read_csv(merged_csv)
    print(f"Loaded {{len(df)}} neighborhoods from merged CSV")

    trends = {{}}
    if "sample_source" in df.columns and "community" in df.columns:
        for source, grp in df.groupby("sample_source"):
            counts = grp["community"].value_counts().to_dict()
            trends[source] = {{"n_neighborhoods": len(grp), "community_counts": counts}}
    else:
        trends["note"] = "community column not present; CCD assigns communities per run"

    with open(OUTPUT_DIR / "trends.json", "w") as f:
        json.dump(trends, f, indent=2)
    print("Saved: trends.json")

if __name__ == "__main__":
    main()
'''


# ============================================================================
# UNIFIED SYNTHESIS CONTROLLER
# ============================================================================

class UnifiedSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict, store_fn: Callable = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self._store_analysis_images = store_fn or (lambda *a, **k: None)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_sample", False)
        
        if is_single:
            return self._synthesize_single(state)
        else:
            return self._synthesize_batch(state)
    
    def _compute_spatial_stats(self, output_dir: Path, k_neighbors: int = 10) -> Optional[dict]:
        """
        Compute spatial distribution statistics from the CCD .xyz output file.

        Reads the community-labelled .xyz file (columns: community x y z) and returns:
          - community_centroids_normalized: center-of-mass per community in [0,1]^3
          - community_spatial_extent_normalized: bounding-box fraction per community
          - neighbor_purity_per_community: fraction of k nearest neighbors sharing label
          - segregation_score: 0 = random mix, 1 = fully segregated
          - inter_centroid_distances_normalized: pairwise normalized centroid separations
        """
        from scipy.spatial import KDTree

        xyz_files = sorted(output_dir.glob("*community_clustering.xyz"))
        if not xyz_files:
            xyz_files = sorted(output_dir.glob("*.xyz"))
        if not xyz_files:
            self.logger.warning("   ⚠️  No .xyz community file found — skipping spatial analysis")
            return None

        xyz_path = xyz_files[-1]
        communities_list, coords_list = [], []
        with open(xyz_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    comm = int(float(parts[0]))
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    communities_list.append(comm)
                    coords_list.append([x, y, z])
                except ValueError:
                    continue  # skip header / comment lines

        if len(communities_list) < 10:
            self.logger.warning("   ⚠️  Too few points in .xyz file for spatial analysis")
            return None

        communities = np.array(communities_list)
        coords = np.array(coords_list, dtype=float)
        unique_comms = sorted(np.unique(communities).tolist())
        n_comms = len(unique_comms)

        # Normalise coordinates to [0, 1] within the dataset bounding box
        c_min = coords.min(axis=0)
        c_range = np.where(coords.max(axis=0) - c_min > 0, coords.max(axis=0) - c_min, 1.0)
        coords_n = (coords - c_min) / c_range

        # 1. Community centroids (normalised)
        centroids = {
            int(c): [round(float(v), 3) for v in coords_n[communities == c].mean(axis=0)]
            for c in unique_comms
        }

        # 2. Spatial extent per community (bounding box as fraction of total)
        extents = {}
        for c in unique_comms:
            pts = coords_n[communities == c]
            ext = (pts.max(axis=0) - pts.min(axis=0)).tolist() if len(pts) > 1 else [0.0, 0.0, 0.0]
            extents[int(c)] = [round(v, 3) for v in ext]

        # 3. Neighbour purity via KNN (sampled for speed)
        max_pts = 5000
        if len(communities) > max_pts:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(communities), max_pts, replace=False)
            s_coords, s_comms = coords_n[idx], communities[idx]
        else:
            s_coords, s_comms = coords_n, communities

        k = min(k_neighbors, len(s_coords) - 1)
        tree = KDTree(s_coords)
        _, nn_idx = tree.query(s_coords, k=k + 1)  # +1: first hit is self

        purity = {}
        for c in unique_comms:
            mask = s_comms == c
            if not np.any(mask):
                continue
            pur = float(np.mean([
                np.mean(s_comms[nn_idx[i, 1:]] == c)
                for i in np.where(mask)[0]
            ]))
            purity[int(c)] = round(pur, 3)

        random_purity = 1.0 / n_comms if n_comms > 1 else 1.0
        overall_purity = float(np.mean(list(purity.values()))) if purity else random_purity
        denom = 1.0 - random_purity
        segregation_score = round((overall_purity - random_purity) / denom, 3) if denom > 0 else 0.0

        # 4. Inter-centroid distances (normalised)
        centroid_arr = np.array([centroids[c] for c in unique_comms])
        icd = {}
        for i, ci in enumerate(unique_comms):
            for j, cj in enumerate(unique_comms):
                if i < j:
                    icd[f"{ci}-{cj}"] = round(float(np.linalg.norm(centroid_arr[i] - centroid_arr[j])), 3)

        self.logger.info(
            f"   🗺️  Spatial stats: segregation_score={segregation_score}, "
            f"n_comms={n_comms}, xyz={xyz_path.name}"
        )
        return {
            "n_points_analyzed": len(communities_list),
            "community_centroids_normalized": {str(k): v for k, v in centroids.items()},
            "community_spatial_extent_normalized": {str(k): v for k, v in extents.items()},
            "neighbor_purity_per_community": {str(k): v for k, v in purity.items()},
            "segregation_score": segregation_score,
            "inter_centroid_distances_normalized": icd,
            "interpretation_guide": (
                "segregation_score: 0=random mix, 1=fully segregated. "
                "neighbor_purity: fraction of k=10 nearest neighbours sharing the same community. "
                "Centroids and extents are normalised to the dataset bounding box [0,1]^3."
            ),
        }

    def _synthesize_single(self, state: dict) -> dict:
        """
        Synthesize findings for a single sample analysis.

        Sends CCD statistics and the KS-statistics heatmap to the LLM for
        scientific interpretation.
        """
        self.logger.info("\n\n🔬 --- SINGLE SAMPLE SYNTHESIS --- 🔬\n")

        prompt_parts = [CCD_ANALYSIS_INSTRUCTIONS]

        # System / sample information
        system_info = state.get("system_info", {})
        if system_info:
            prompt_parts.append(
                f"\n\n## System Information\n```json\n{json.dumps(system_info, indent=2)}\n```"
            )

        # CCD statistics
        ccd_stats = {
            "n_communities": state.get("ccd_communities"),
            "community_neighborhood_counts": {
                str(k): int(v)
                for k, v in (state.get("ccd_neighborhood_counts") or {}).items()
            },
            "community_mean_ks_statistics": [
                list(map(float, row))
                for row in (state.get("ccd_compositions") or [])
            ],
        }
        # Spatial distribution of communities
        spatial_stats = self._compute_spatial_stats(self.output_dir)
        if spatial_stats:
            ccd_stats["spatial_distribution"] = spatial_stats

        prompt_parts.append(
            f"\n\n## CCD Analysis Statistics\n```json\n{json.dumps(ccd_stats, indent=2)}\n```"
        )

        # KS statistics heatmap image (written by detect_compositional_communities)
        ks_png = self.output_dir / "KS_stats.png"
        if ks_png.exists():
            try:
                with open(ks_png, "rb") as f:
                    img_bytes = f.read()
                prompt_parts.append("\n\n## KS Statistics Heatmap\n")
                prompt_parts.append({"mime_type": "image/png", "data": img_bytes})
                self.logger.info("   📊 Added KS_stats.png to synthesis prompt")
            except Exception as e:
                self.logger.warning(f"   Could not load KS_stats.png: {e}")
        else:
            self.logger.warning("   ⚠️ KS_stats.png not found; excluding from prompt")

        # Analysis parameters used
        params = state.get("locked_params") or state.get("llm_params") or state.get("current_params", {})
        if params:
            prompt_parts.append(
                f"\n\n## Analysis Parameters\n```json\n{json.dumps(params, indent=2)}\n```"
            )

        n_images = sum(1 for p in prompt_parts if isinstance(p, dict) and p.get("mime_type"))
        self.logger.info(f"   📤 Sending {n_images} image(s) to LLM for synthesis")

        try:
            self.logger.info("   🧠 Calling LLM for synthesis...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            if error_dict:
                self.logger.error(f"   ❌ Synthesis parsing failed: {error_dict}")
                state["synthesis_result"] = self._fallback_single(state)
            else:
                state["synthesis_result"] = result_json
                state["result_json"] = result_json
                self.logger.info("   ✅ Single sample synthesis complete.")
        except Exception as e:
            self.logger.error(f"   ❌ Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_single(state)

        return state


    
    def _synthesize_batch(self, state: dict) -> dict:
        """
        Synthesize findings across multiple samples.
        """
        self.logger.info("\n\n🔬 --- BATCH SYNTHESIS --- 🔬\n")

        batch_results = state.get("batch_results", [])
        series_metadata = state.get("series_metadata", {})
        batch_params = state.get("batch_params", {})

        # Build per-sample summary
        stats_summary = [
            {
                "index": r["index"],
                "name": r["sample_name"],
                "n_communities": r.get("n_communities", 0),
                "community_neighborhood_counts": {
                    str(k): int(v)
                    for k, v in (r.get("community_neighborhood_counts") or {}).items()
                },
            }
            for r in batch_results
            if r.get("success")
        ]

        # Global CCD statistics (shared across merged run)
        ccd_stats = {
            "n_communities": state.get("ccd_communities"),
            "community_mean_ks_statistics": [
                list(map(float, row))
                for row in (state.get("ccd_compositions") or [])
            ],
        }

        prompt_parts = [CCD_ANALYSIS_INSTRUCTIONS]

        # Analysis parameters
        prompt_parts.append(
            f"\n\n**ANALYSIS PARAMETERS:**\n"
            f"- Samples analysed: {batch_params.get('n_samples', len(batch_results))}\n"
            f"- k-Means Values: {batch_params.get('k_values', 'auto')}\n"
            f"- Percentile q: {batch_params.get('q', 'auto')}\n"
            f"- Ignored ions: {batch_params.get('ignore_ions', [])}"
        )

        # Global CCD stats
        # Spatial distribution of communities
        spatial_stats = self._compute_spatial_stats(self.output_dir)
        if spatial_stats:
            ccd_stats["spatial_distribution"] = spatial_stats

        prompt_parts.append(
            f"\n\n**GLOBAL CCD STATISTICS:**\n```json\n{json.dumps(ccd_stats, indent=2)}\n```"
        )

        # Per-sample summary (condensed)
        prompt_parts.append(
            f"\n\n**PER-SAMPLE SUMMARY:**\n{json.dumps(stats_summary[:10], indent=2)}"
        )
        if len(stats_summary) > 10:
            prompt_parts.append(f"\n... and {len(stats_summary) - 10} more samples")

        # Series metadata
        if series_metadata:
            prompt_parts.append(
                f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}"
            )

        # KS statistics heatmap
        ks_png = self.output_dir / "KS_stats.png"
        if ks_png.exists():
            try:
                with open(ks_png, "rb") as f:
                    img_bytes = f.read()
                prompt_parts.append("\n\n**KS STATISTICS HEATMAP:**\n")
                prompt_parts.append({"mime_type": "image/png", "data": img_bytes})
                self.logger.info("   📊 Added KS_stats.png to batch synthesis prompt")
            except Exception as e:
                self.logger.warning(f"   Could not load KS_stats.png: {e}")

        prompt_parts.append(
            "\n\nBased on all the above, provide a comprehensive scientific synthesis as a JSON object. Output ONLY the JSON."
        )

        n_images = sum(1 for p in prompt_parts if isinstance(p, dict) and p.get("mime_type"))
        self.logger.info(f"   📤 Sending {n_images} image(s) to LLM for batch synthesis")

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = self._fallback_batch(state)
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Batch synthesis complete.")
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_batch(state)

        return state
    
    def _fallback_single(self, state: dict) -> dict:
        n_communities = state.get("ccd_communities") or 0
        return {
            "detailed_analysis": f"CCD identified {n_communities} compositional communities.",
            "scientific_claims": [{"claim": f"CCD identified {n_communities} compositional communities.", "scientific_impact": "Complex microstructure identified."}]
        }
    
    def _fallback_batch(self, state: dict) -> dict:
        """Fallback for batch synthesis."""
        n_samples = state.get("num_samples", 0)
        n_communities = state.get("ccd_communities") or 0
        return {
            "detailed_analysis": f"CCD analysis of {n_samples} samples identified {n_communities} compositional communities.",
            "scientific_claims": [{
                "claim": f"CCD analysis of {n_samples} APT samples reveals {n_communities} distinct compositional communities.",
                "scientific_impact": "CCD enables tracking of compositional structure across multiple APT datasets.",
                "has_anyone_question": "Has anyone used CCD to compare compositional communities across multiple APT datasets?",
                "keywords": ["APT", "CCD", "compositional communities", "microstructure"]
            }]
        }
        

class UnifiedReportGenerationController:
    """
    [📄 Report Step]
    Generates HTML report with embedded visualizations - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_sample", False)
        self.logger.info("\n\n📄 --- GENERATING REPORT --- 📄\n")
        
        try:
            if not is_single:
                self._generate_trend_visualizations(state)

            if is_single:
                self._generate_single_sample_report(state)
            else:
                self._generate_batch_report(state)
        except Exception as e:
            self.logger.error(f"   ❌ Report generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return state
    
    def _generate_trend_visualizations(self, state: dict) -> None:
        """Write trends.json summarising per-source community distributions."""
        batch_results = state.get("batch_results", [])
        if not batch_results:
            self.logger.warning("   No batch results available for trend analysis")
            return

        trends = {}
        for r in batch_results:
            name = r.get("sample_name", f"sample_{r.get('index', 0)}")
            trends[name] = {
                "n_communities": r.get("n_communities", 0),
                "community_neighborhood_counts": {
                    str(k): int(v)
                    for k, v in (r.get("community_neighborhood_counts") or {}).items()
                },
            }

        try:
            with open(self.output_dir / "trends.json", "w") as f:
                json.dump(trends, f, indent=2)
            self.logger.info("   📊 Generated trends.json")
            state["trend_data"] = trends
        except Exception as e:
            self.logger.error(f"   ❌ Trend file write failed: {e}")
    
    # =========================================================================
    # SINGLE SAMPLE REPORT
    # =========================================================================

    def _generate_single_sample_report(self, state: dict) -> None:
        """
        Generate HTML report for a single neighborhood CSV analysis.

        Structure:
            1. System Information
            2. Analysis Parameters
            3. CCD Statistics
            4. Visualizations (KS_stats.png)
            5. Scientific Analysis
            6. Scientific Claims
        """
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        result = batch_results[0] if batch_results else {}

        detailed_analysis = synthesis.get("detailed_analysis", "No analysis available.")
        scientific_claims = synthesis.get("scientific_claims", [])

        params = state.get("locked_params") or state.get("llm_params") or state.get("current_params", {})
        system_info = state.get("system_info", {})

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        sample_name = result.get("sample_name", state.get("first_data_name", "unknown"))
        n_communities = state.get("ccd_communities") or result.get("n_communities", 0)

        # KS stats plot as base64 for embedding
        ks_b64 = None
        ks_png = self.output_dir / "KS_stats.png"
        if ks_png.exists():
            try:
                with open(ks_png, "rb") as f:
                    ks_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                self.logger.warning(f"Could not load KS_stats.png for report: {e}")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CCD APT Analysis Report - {sample_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f4f9; line-height: 1.6; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ color: #2980b9; margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        .info-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .info-box p {{ margin: 8px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 25px; border-radius: 8px; border: 1px solid #eee; font-size: 0.95em; }}
        .claim-card {{ background: linear-gradient(135deg, #e8f6f3 0%, #d5f5e3 100%); border-left: 5px solid #1abc9c; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .claim-card strong {{ color: #16a085; }}
        .full-width-viz {{ width: 100%; margin: 20px 0; text-align: center; }}
        .full-width-viz img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .full-width-viz .caption {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; padding-top: 20px; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 CCD APT Analysis Report</h1>
    <p><em>Generated: {timestamp} | Sample: {sample_name} | Communities: {n_communities}</em></p>
"""

        # === 1. SYSTEM INFORMATION ===
        html += "\n    <h2>📋 System Information</h2>\n"
        if system_info and isinstance(system_info, dict):
            html += '    <div class="info-box">\n'
            for key, value in system_info.items():
                html += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html += "    </div>\n"
        else:
            html += '    <div class="info-box"><p>No system information provided.</p></div>\n'

        # === 2. ANALYSIS PARAMETERS ===
        if params:
            html += "\n    <h2>⚙️ Analysis Parameters</h2>\n    <div class=\"info-box\">\n"
            for k, v in params.items():
                html += f"        <p><strong>{k}:</strong> {v}</p>\n"
            html += "    </div>\n"

        # === 3. CCD STATISTICS ===
        html += "\n    <h2>📊 CCD Statistics</h2>\n    <div class=\"info-box\">\n"
        html += f"        <p><strong>Number of communities:</strong> {n_communities}</p>\n"
        nh_counts = state.get("ccd_neighborhood_counts") or result.get("community_neighborhood_counts")
        if nh_counts:
            for comm, cnt in nh_counts.items():
                html += f"        <p><strong>Community {comm}:</strong> {cnt} neighborhoods</p>\n"
        html += "    </div>\n"

        # === 4. VISUALIZATIONS ===
        html += "\n    <h2>📈 Visualizations</h2>\n"
        if ks_b64:
            html += f"""    <div class="full-width-viz">
        <img src="data:image/png;base64,{ks_b64}" alt="KS Statistics Heatmap">
        <p class="caption">
            Mean KS statistics per community. Positive values indicate enrichment; negative values indicate depletion relative to the bulk composition.
        </p>
    </div>
"""
        else:
            html += '    <div class="info-box"><p>KS statistics plot not available.</p></div>\n'

        # === 5. SCIENTIFIC ANALYSIS ===
        html += f'\n    <h2>🔍 Scientific Analysis</h2>\n    <div class="analysis-text">{detailed_analysis}</div>\n'

        # === 6. SCIENTIFIC CLAIMS ===
        html += "\n    <h2>💡 Scientific Claims</h2>\n"
        if scientific_claims:
            for i, claim in enumerate(scientific_claims, 1):
                keywords_str = ', '.join(claim.get('keywords', [])) or 'N/A'
                html += f"""    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br><br>
        <strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}<br><br>
        <strong>Literature Search Query:</strong> {claim.get('has_anyone_question', 'N/A')}<br><br>
        <strong>Keywords:</strong> {keywords_str}
    </div>
"""
        else:
            html += '    <div class="info-box"><p>No scientific claims generated.</p></div>\n'

        html += """    <div class="footer">Generated by CCD APT Analysis Agent</div>
</div>
</body>
</html>"""

        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        self.logger.info(f"   ✅ Generated report: {report_path}")
    
    # =========================================================================
    # BATCH REPORT
    # =========================================================================
    
    def _generate_batch_report(self, state: dict) -> None:
        """Generate HTML report for a batch (multi-CSV) CCD analysis."""
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        batch_params = state.get("batch_params", {})
        series_metadata = state.get("series_metadata", {})

        detailed_analysis = synthesis.get("detailed_analysis", "No synthesis available.")
        scientific_claims = synthesis.get("scientific_claims", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_samples = state.get("num_samples", len(batch_results))
        successful = sum(1 for r in batch_results if r.get("success"))
        n_communities = state.get("ccd_communities", 0)
        system_info = state.get("system_info", {})

        # KS stats heatmap
        ks_b64 = None
        ks_png = self.output_dir / "KS_stats.png"
        if ks_png.exists():
            try:
                with open(ks_png, "rb") as f:
                    ks_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                self.logger.warning(f"Could not load KS_stats.png for batch report: {e}")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CCD APT Batch Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f4f4f9; line-height: 1.6; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        h2 {{ color: #2980b9; margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        .info-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .info-box p {{ margin: 8px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 25px; border-radius: 8px; border: 1px solid #eee; }}
        .claim-card {{ background: linear-gradient(135deg, #e8f6f3 0%, #d5f5e3 100%); border-left: 5px solid #1abc9c; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin: 25px 0; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center; }}
        .image-card img {{ max-width: 100%; border-radius: 4px; margin-bottom: 10px; }}
        .image-card .caption {{ color: #666; font-size: 0.9em; }}
        .full-width-viz {{ width: 100%; margin: 25px 0; text-align: center; }}
        .full-width-viz img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .full-width-viz .caption {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; padding-top: 20px; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 CCD APT Batch Analysis Report</h1>
    <p><em>Generated: {timestamp} | Samples: {successful}/{num_samples} | Communities: {n_communities}</em></p>
"""

        # === 1. SYSTEM INFORMATION ===
        html += "\n    <h2>📋 System Information</h2>\n"
        if system_info and isinstance(system_info, dict):
            html += '    <div class="info-box">\n'
            for key, value in system_info.items():
                html += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html += "    </div>\n"
        else:
            html += '    <div class="info-box"><p>No system information provided.</p></div>\n'

        # === 2. ANALYSIS PARAMETERS ===
        if batch_params:
            html += "\n    <h2>⚙️ Analysis Parameters</h2>\n    <div class=\"info-box\">\n"
            for k, v in batch_params.items():
                html += f"        <p><strong>{k}:</strong> {v}</p>\n"
            html += "    </div>\n"

        # === 3. PER-SAMPLE SUMMARY ===
        html += "\n    <h2>📦 Per-Sample Summary</h2>\n    <div class=\"info-box\">\n"
        for r in batch_results:
            status = "✅" if r.get("success") else "❌"
            html += f"        <p>{status} <strong>{r.get('sample_name', 'unknown')}:</strong> {r.get('n_communities', 'N/A')} communities</p>\n"
        html += "    </div>\n"

        # === 4. VISUALIZATIONS ===
        html += "\n    <h2>📊 Visualizations</h2>\n"
        if ks_b64:
            html += f"""    <div class="full-width-viz">
        <img src="data:image/png;base64,{ks_b64}" alt="KS Statistics Heatmap">
        <p class="caption">Mean KS statistics per community across the merged dataset. Positive values indicate enrichment; negative indicate depletion.</p>
    </div>
"""
        else:
            html += '    <div class="info-box"><p>KS statistics plot not available.</p></div>\n'

        # === 5. SCIENTIFIC ANALYSIS ===
        html += f'\n    <h2>🔍 Scientific Analysis</h2>\n    <div class="analysis-text">{detailed_analysis}</div>\n'

        # === 6. SCIENTIFIC CLAIMS ===
        html += "\n    <h2>💡 Scientific Claims</h2>\n"
        if scientific_claims:
            for i, claim in enumerate(scientific_claims, 1):
                keywords_str = ', '.join(claim.get('keywords', [])) or 'N/A'
                html += f"""    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br><br>
        <strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}<br><br>
        <strong>Literature Search Query:</strong> {claim.get('has_anyone_question', 'N/A')}<br><br>
        <strong>Keywords:</strong> {keywords_str}
    </div>
"""
        else:
            html += '    <div class="info-box"><p>No scientific claims generated.</p></div>\n'

        html += """    <div class="footer">Generated by CCD APT Analysis Agent</div>
</div>
</body>
</html>"""

        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        self.logger.info(f"   ✅ Generated batch report: {report_path}")