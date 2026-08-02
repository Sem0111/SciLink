"""
CCD APT Analysis Agent
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple, Union

from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent, AnalysisInput
from scilink.agents.exp_agents.human_feedback import SimpleFeedbackMixin
from scilink.agents.exp_agents._deprecation import normalize_params

from .ccd_apt_pipelines import create_unified_apt_pipeline

APT_CLAIMS_INSTRUCTIONS = """You are an expert system specialized in analyzing atom probe tomography (APT) data of materials.
You will receive a pre-generated APT neighborhoods file and potentially additional images derived from Compostional Community Detection (CCD).
These derived images show compositional communities (representing dominant compositional domains). 

Your goal is to extract key information from these images and formulate a set of precise scientific claims that can be used to search existing literature.

**Important Note on Formulation:** When formulating claims, focus on specific, testable observations that could be compared against existing research. Use precise scientific terminology, and avoid ambiguous statements. Make each claim distinct and focused on a single phenomenon or observation.

You MUST output a valid JSON object containing two keys: "detailed_analysis" and "scientific_claims".

1.  **detailed_analysis**: (String) Provide a thorough text analysis of the APT data. Explicitly correlate features
    in the original data with patterns observed in the CCD communities, if available.
    Identify features like:
    * Line defects (dislocations, grain boundaries)
    * Extended defects (stacking faults, phase boundaries)
    * Interfacial regions with distinct composition
    * Local chemical composition differences 
    * Concentration gradients
    * Heterostructure interfaces

2.  **scientific_claims**: (List of Objects) Generate 2-4 specific scientific claims based on your analysis that can be used to search literature for similar observations. Each object must have the following keys:
    * **claim**: (String) A single, focused scientific claim written as a complete sentence about a specific observation from the APT data.
    * **scientific_impact**: (String) A brief explanation of why this claim would be scientifically significant if confirmed through literature search or further experimentation.
    * **has_anyone_question**: (String) A direct question starting with "Has anyone" that reformulates the claim as a research question.
    * **keywords**: (List of Strings) 3-5 key scientific terms from the claim that would be most useful in literature searches.

Focus on formulating claims that are specific enough to be meaningfully compared against literature but general enough to have a reasonable chance of finding matches. 
Avoid using **overly specific** numbers from the analysis.
Your question **must be portable** and understandable without seeing the image or having access to the detailed analysis. **DO NOT** use words like "this," "that," "the observed pattern," or "the specific signature." 
Ensure the final output is ONLY the JSON object and nothing else.
"""



ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS = """You are an expert in atomic-resolution characterization analyzing comprehensive experimental results to recommend optimal follow-up measurements.

You will receive:
1. Detailed atomistic analysis results with atomic-scale insights
2. Generated scientific claims from the analysis
3. Analysis images showing:
   - Intensity histogram: Distribution of atomic intensities (different species/environments)
   - Intensity-based clustering: Atoms colored by intensity groups (often different atomic species)
   - Local environment clustering: Atoms colored by their structural neighborhood (defects, interfaces, etc.)
   - Nearest-neighbor distance maps: Color-coded atomic positions showing local strain and lattice variations
   - These reveal atomic species, defects, grain boundaries, interfaces, and local structural environments
4. Optional novelty assessment results from literature review
5. Current experimental parameters and context

Your goal is to recommend the most scientifically valuable follow-up measurements to maximize research impact.

**Recommendation Categories:**
1. **Spatial Refinement**: Higher resolution or different orientations for atomic-scale features
2. **Chemical Analysis**: Atomic-scale spectroscopic techniques (EELS, EDS, etc.)
3. **Dynamic Studies**: In-situ measurements of atomic processes
4. **Computational Correlative**: DFT validation measurements for specific structures
5. **Statistical Sampling**: Sampling across different atomic environments or conditions

**For each recommendation, provide:**
- Specific measurement parameters (resolution, voltage, acquisition time, etc.)
- Scientific justification linked to current findings
- Expected information gain and impact
- Priority level (1=highest, 5=lowest)

You MUST output a valid JSON object with two keys: "analysis_integration" and "measurement_recommendations".

1. **analysis_integration**: (String) How you integrated atomistic findings and novelty assessment (if available) to inform recommendations.

2. **measurement_recommendations**: (List of Objects) 2-5 specific measurements, each with:
   * **category**: (String) One of the five categories above
   * **description**: (String) Detailed measurement description with specific parameters
   * **target_regions**: (String) Specific atomic features or regions to target
   * **scientific_justification**: (String) Why this measurement provides valuable insights
   * **expected_outcomes**: (String) Specific information to be gained
   * **priority**: (Integer) 1-5 priority ranking
   * **parameters**: (Object) Specific measurement parameters

Focus on actionable recommendations that maximize scientific insight while being technically feasible.
"""

class CCDAPTAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    CCD-APT Analysis Agent based on sliding FFT and NMF decomposition.
    Utilizes LLMs to select appropriate FFT/NMF parameters and interpret results.
    
    Example:
        agent = CCDAPTAnalysisAgent(api_key="...")
        
        # Single image
        result = agent.analyze("neighborhoods.csv")
        
        # Multiple images
        result = agent.analyze(["neighbors1.csv", "neighbors2.csv"])
        
        # Numpy stack
        result = agent.analyze(my_stack)
        
        # Get measurement recommendations
        recommendations = agent.recommend_measurements(analysis_result=result)
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        # Deprecated params
        google_api_key: str | None = None,
        local_model: str = None,
        # Agent specific params
        ccd_settings: dict | None = None,
        enable_human_feedback: bool = False,
        output_dir: str = "apt_analysis_output"
    ):
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, 
            source="CCDAPTAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "apt"

        # Resolve output directory
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories
        self.viz_dir = self.output_dir / "ccd_apt_visualizations"
        self.data_dir = self.output_dir / "analysis_output"
        self.scripts_dir = self.output_dir / "scripts"
        
        # Create directories
        for d in [self.viz_dir, self.data_dir, self.scripts_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Prepare settings
        self.settings = ccd_settings if ccd_settings else {}
        self.settings.setdefault('CCD_ENABLED', True)
        self.settings.setdefault('enable_human_feedback', enable_human_feedback)
        self.settings.setdefault('max_feedback_iterations', 3)
        self.settings.setdefault('max_script_corrections', 3)
        self.settings.setdefault('save_visualizations', True)
        self.settings['visualization_dir'] = str(self.viz_dir)
        self.settings['output_dir'] = str(self.data_dir)
        
        self._recommendation_agent = None
        
        if self.settings.get('CCD_ENABLED', True):
            self.logger.info(f"CCDAPTAnalysisAgent initialized. Outputs: {self.output_dir}")
        else:
            self.logger.warning("CCDAPTAnalysisAgent initialized, but 'CCD_ENABLED' is False.")
    
    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the APT agent."""
        return {
            "pipeline_type": "ccd_apt_unified",
            "analysis_results": [],
            "batch_mode": False,
            "locked_params": None,
            "ccd_communities": None,
            "ccd_neighborhood_counts": None,
            "ccd_compositions": None,
        }
    
    def _initialize_ccd_params(self) -> dict:
        """Get initial CCD parameters from settings."""
        return {
            "k_values": self.settings.get('k_values', [4,5,6]), 
            "ignore_ions": self.settings.get('ignore_ions', []),
            "n_repeats": self.settings.get('n_repeats', 5),
            "q": self.settings.get('q', 25),
        }
    
    def analyze(
        self,
        data: AnalysisInput,
        system_info: Optional[Union[Dict[str, Any], str]] = None,
        # APT-specific options
        series_metadata: Optional[dict] = None,
        preset_params: Optional[dict] = None,
        feedback_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified analysis method — handles single neighborhood CSV files and batches identically.

        Single-sample analysis is internally converted to a batch of 1.

        Args:
            data: Input neighborhood data. Can be:
                - str: Path to a single neighborhoods .csv file
                - List[str]: Multiple .csv paths (all merged into one CCD run)
                - pd.DataFrame: Pre-loaded neighborhood DataFrame (written to a
                  temporary CSV inside ``output_dir`` before processing)
            system_info: System/sample information dict or path to a JSON file.
                May include a ``"series"`` key with series metadata that will be
                extracted automatically.
            series_metadata: Optional metadata describing the experimental variable
                that differs across samples. Can also live in ``system_info["series"]``.
                Expected structure::

                    {
                        "variable": "temperature",  # independent variable name
                        "values": [300, 350, 400],   # one value per CSV, in file order
                        "unit": "K"                  # unit
                    }
            preset_params: If provided, skip LLM parameter estimation and use these
                CCD parameters directly::

                    {"k_values": [4, 5, 6], "q": 25, "ignore_ions": ["O1", "O1H1"]}

            feedback_callback: Optional function for custom feedback collection.

        Returns:
            Dict with ``status``, ``detailed_analysis``, ``scientific_claims``,
            ``summary``, ``output_directory``, plus APT-specific fields
            (``n_communities``, ``community_compositions``).

        Examples:
            # Single neighborhoods CSV
            result = agent.analyze("sample_neighborhoods.csv")

            # Multiple CSVs (merged CCD run, per-source stats preserved)
            result = agent.analyze(["sample1_neighborhoods.csv", "sample2_neighborhoods.csv"])

            # Pre-loaded DataFrame
            result = agent.analyze(my_neighborhood_df)

            # With preset parameters (skip LLM estimation)
            result = agent.analyze(
                "sample_neighborhoods.csv",
                preset_params={"k_values": [4, 5, 6], "q": 25, "ignore_ions": ["O1"]}
            )
        """
        # -------------------------------------------------------------------- #
        # Handle pd.DataFrame input: write to a temp CSV inside output_dir    #
        # -------------------------------------------------------------------- #
        if isinstance(data, pd.DataFrame):
            tmp_csv = self.data_dir / "_input_neighborhoods.csv"
            data.to_csv(tmp_csv, index=False)
            self.logger.info(f"DataFrame input received — written to {tmp_csv}")
            data = str(tmp_csv)

        # -------------------------------------------------------------------- #
        # Reject raw numpy arrays (not meaningful for APT CSV neighborhood data)
        # -------------------------------------------------------------------- #
        if isinstance(data, np.ndarray):
            return {
                "status": "error",
                "error": {
                    "error": "Unsupported input type",
                    "details": (
                        "CCDAPTAnalysisAgent requires a .csv neighborhoods file path or "
                        "a pd.DataFrame. Raw numpy arrays are not supported."
                    ),
                },
                "output_directory": str(self.output_dir),
            }

        # Parse input (str or List[str] only at this point)
        data_path, data_paths, data_array, error = self._parse_data_input(data)

        # Neighborhood
        nm_per_neighborhood = 1 
        
        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir)
            }
        
        # Convert single image to batch of 1
        if data_path is not None:
            data_paths = [data_path]
            self.logger.info(f"Single image mode: treating as batch of 1")
        
        # Set input type and count
        num_samples = len(data_paths)
        input_type = "file_paths"
        
        is_single_sample = (num_samples == 1)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 APT CCD ANALYSIS - {num_samples} CSV file{'s' if num_samples > 1 else ''}")
        self.logger.info(f"{'='*80}\n")
        
        # Load and preprocess first image for initial analysis
        try:
            first_sample = pd.read_csv(data_paths[0])
            first_sample_name = Path(data_paths[0]).stem
        except Exception as e:
            return {
                "status": "error",
                "error": {"error": "Failed to load neighborhoods", "details": str(e)},
                "output_directory": str(self.output_dir)
            }
        
              
        # Build initial state dict
        _sys_info = self._handle_system_info(system_info)
        _sys_info, series_metadata = self._extract_series_metadata(_sys_info, series_metadata)
        state = {
            # Input data
            "data_paths": data_paths,
            "data_array": data_array,
            "input_type": input_type,
            "num_samples": num_samples,
            "is_single_sample": is_single_sample,

            # For series compatibility
            "series_data": data_array if data_array is not None else None,
            "n_frames": num_samples,
            "first_frame": first_sample,

            # System info
            "system_info": _sys_info,
            "series_metadata": series_metadata or {},
            
            # First sample path/name
            "data_path": data_paths[0] if data_paths else first_sample_name,
            "first_data_name": first_sample_name,

            # Spatial calibration
            "nm_per_neighborhood": nm_per_neighborhood,

            # Settings and params
            "settings": self.settings,
            "enable_human_feedback": self.settings.get('enable_human_feedback', False) and preset_params is None,
            "current_params": preset_params or self._initialize_ccd_params(),
            "preset_params": preset_params,
            "feedback_callback": feedback_callback,

            # CCD results placeholders
            "ccd_communities": None,
            "ccd_neighborhood_counts": None,
            "ccd_compositions": None,
            "llm_params": None,
            "error_dict": None,
        }
        
        # If preset params provided, inject them
        if preset_params:
            state["locked_params"] = preset_params
            state["first_frame_results"] = {"llm_params": preset_params}
        
        # Create and Execute Unified Pipeline
        pipeline = create_unified_apt_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            preset_params=preset_params
        )
        
        # Execute pipeline steps
        for i, controller in enumerate(pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 STEP {i}: {step_name}\n")
            
            try:
                state = controller.execute(state)
                
                # Check for errors
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {step_name}: {state['error_dict']}")
                    break
                
                # Check for cancellation
                if state.get("batch_cancelled"):
                    self.logger.info("Analysis cancelled by user.")
                    return {
                        "status": "cancelled",
                        "first_frame_results": state.get("first_frame_results"),
                        "output_directory": str(self.output_dir)
                    }
                    
            except Exception as e:
                self.logger.error(f"Pipeline step {step_name} raised exception: {e}")
                state["error_dict"] = {"error": f"Pipeline step failed: {step_name}", "details": str(e)}
                break
        
        if state.get("error_dict"):
            return {
                "status": "error",
                "error": state["error_dict"],
                "output_directory": str(self.output_dir)
            }
        
        # Compile final results
        final_results = self._compile_results(state)

        # Store KS stats PNG so SimpleFeedbackMixin can pass it to the LLM during refinement
        ks_png = self.output_dir / "KS_stats.png"
        if ks_png.exists():
            with open(ks_png, "rb") as f:
                ks_bytes = f.read()
            self._store_analysis_images([{"label": "KS Statistics Heatmap", "data": ks_bytes}])

        # Save final results JSON
        final_path = self.output_dir / "analysis_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Results saved to: {final_path}")
        self.logger.info(f"{'='*80}\n")
        
        # Log action
        self._log_action(
            action="analyze",
            input_ctx={
                "num_samples": num_samples,
                "input_type": input_type,
                "series_metadata": series_metadata
            },
            result=final_results.get("summary"),
            rationale="APT CCD analysis completed."
        )
        
        return final_results
    
    def _compile_results(self, state: dict) -> Dict[str, Any]:
        """Compile pipeline state into a consistent output structure."""
        is_single = state.get("is_single_sample", False)
        num_samples = state.get("num_samples", 1)
        batch_results = state.get("batch_results", [])

        results = {
            "status": "success",
            "summary": {
                "total_samples": num_samples,
                "successful": (
                    sum(1 for r in batch_results if r.get("success", False))
                    if batch_results
                    else (1 if not state.get("error_dict") else 0)
                ),
                "input_type": state.get("input_type"),
                "parameters_used": state.get("locked_params", state.get("current_params", {})),
                "is_single_sample": is_single,
            },
            "output_directory": str(self.output_dir),
        }

        synthesis = state.get("synthesis_result", {})

        # CCD-specific fields present for both single and batch
        results["n_communities"] = (
            state.get("ccd_communities")
            or (batch_results[0].get("n_communities", 0) if batch_results else 0)
        )
        results["community_compositions"] = state.get("ccd_compositions")
        results["community_neighborhood_counts"] = state.get("ccd_neighborhood_counts")

        if is_single:
            result_json = state.get("result_json", {})
            results["detailed_analysis"] = (
                synthesis.get("detailed_analysis")
                or result_json.get("detailed_analysis")
                or "Analysis complete."
            )
            results["scientific_claims"] = (
                synthesis.get("scientific_claims")
                or result_json.get("scientific_claims")
                or []
            )
        else:
            results["num_samples_processed"] = state.get("num_samples", 0)
            results["individual_results"] = batch_results
            results["custom_analysis"] = state.get("custom_analysis_results", {})
            results["detailed_analysis"] = synthesis.get("detailed_analysis", "")
            results["scientific_claims"] = synthesis.get("scientific_claims", [])
            results["synthesis"] = synthesis
            results["locked_params"] = state.get("locked_params")
            results["first_frame_results"] = state.get("first_frame_results")

        return results
    
    # =========================================================================
    # CONVENIENCE / BACKWARD-COMPATIBLE METHODS
    # =========================================================================

    def analyze_csv(
        self,
        csv_path: str,
        system_info: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze a single neighborhoods CSV file. Delegates to :meth:`analyze`."""
        return self.analyze(csv_path, system_info=system_info, **kwargs)

    def analyze_from_dataframe(
        self,
        df: "pd.DataFrame",
        system_info: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze a pre-loaded neighborhoods DataFrame. Delegates to :meth:`analyze`."""
        return self.analyze(df, system_info=system_info, **kwargs)

    def analyze_with_preset(
        self,
        data: Union[str, List[str], "pd.DataFrame"],
        k_values: List[int],
        q: int = 25,
        ignore_ions: Optional[List[str]] = None,
        system_info: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Run CCD analysis with explicitly specified parameters, skipping LLM estimation.

        Args:
            data: Path(s) to neighborhoods CSV(s) or a DataFrame.
            k_values: List of community counts to evaluate (e.g. ``[4, 5, 6]``).
            q: KS-test *q* percentile cutoff (default 25).
            ignore_ions: Ion labels to exclude (e.g. ``["O1", "O1H1"]``).
            system_info: Optional system/sample metadata.
        """
        preset = {"k_values": k_values, "q": q, "ignore_ions": ignore_ions or []}
        return self.analyze(data, system_info=system_info, preset_params=preset)
    
    def _get_claims_instruction_prompt(self) -> str:
        return APT_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS