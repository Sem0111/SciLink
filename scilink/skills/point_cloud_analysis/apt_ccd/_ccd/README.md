# Compositional Community Detection


## SciLink Agent Plugin
This agent was developed for use with SciLink analyze. It is designed to identify compositionally distinct regions in a reconstructed APT point cloud.

### Use 

Reconstructed APT data (.pos or .apt file along with a .rrng file) must first be converted into a .csv containing overlapping spherical neighborhoods with 1 nm radius. The preprocessed data file can be passed directly into a SciLink analyze session along with the agent plugin.

```
# Preprocess neighborhoods 
python apt_preprocessing.py --data 'reconstructed_sample.pos' --rrng 'range_file.RRNG' --savedir './raw_data/'

# Set API key
export SCILINK_API_KEY="your-ai-incubator-api-key-here"

# Run SciLink session
scilink analyze --base-url "https://ai-incubator-api.pnnl.gov" --model claude-sonnet-4-6-project --metadata sample_metadata.json --agents apt_agent_plugin.py --data ./raw_data/reconstructed_sample.csv
```

## GitHub Copilot Skill

The CCD algorithm is also implemented as a GitHub Copilot skill for use in any Python environment. The ./github/skills/apt-ccd/SKILL.md file provides a detailed overview of the CCD process.

### Use

Examples of skill use in the GitHub Copilot chat:
```
/apt-ccd Identify compositionally distinct regions in the APT dataset at sample.pos using the range file range.rrng. Generate a report summarizing the identified regions and their compositions.
```



## Requirements 
- scikit-learn
- python-louvain
- apav

## Contact
This agent/tool is maintained by Jenna Pope (jenna.pope@pnnl.gov).

## Reference
If you use this in your work please cite:
```
@article{bilbrey2025compositional,
  title={Compositional Community Detection: Automated Identification of Chemical Segregation in Atom Probe Tomography Data},
  author={Bilbrey, Jenna A and Doty, Christina and Wirth, Mark G and Tong, Mengkong and Royer, Jacqueline and Senor, David J and Devaraj, Arun},
  journal={Microscopy and Microanalysis},
  volume={31},
  number={3},
  pages={ozaf036},
  year={2025},
  publisher={Oxford University Press US}
}
```



