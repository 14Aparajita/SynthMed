"""
Build an extended clinical knowledge base for SynthMed RAG.
Writes data/knowledge_base/extended_kb.jsonl with 60 documents.
"""

import json
from pathlib import Path

OUT = Path("data/knowledge_base/extended_kb.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


# 60 curated passages on diabetic retinopathy and its clinical management.
# These are paraphrased/condensed clinical facts, safe to distribute.
DOCS = [
    "Diabetic retinopathy is the leading cause of preventable blindness in working-age adults worldwide.",
    "The prevalence of diabetic retinopathy increases with the duration of diabetes, reaching 80% after 20 years.",
    "Mild non-proliferative DR (NPDR) is defined by the presence of at least one microaneurysm.",
    "Moderate NPDR shows more than just microaneurysms but less than severe NPDR.",
    "Severe NPDR is defined by the 4-2-1 rule: hemorrhages in 4 quadrants, venous beading in 2, IRMA in 1.",
    "Proliferative DR is characterized by neovascularization of the disc or elsewhere.",
    "High-risk proliferative DR requires prompt treatment with pan-retinal photocoagulation.",
    "Microaneurysms are saccular outpouchings of retinal capillaries, typically 15-60 microns.",
    "Dot and blot hemorrhages occur in the inner nuclear and outer plexiform layers.",
    "Flame hemorrhages occur in the retinal nerve fiber layer and follow the nerve fiber pattern.",
    "Hard exudates are yellow-white deposits of lipoproteins in the outer plexiform layer.",
    "Circinate exudate patterns often surround clusters of leaking microaneurysms.",
    "Cotton wool spots represent focal retinal ischemia in the nerve fiber layer.",
    "Venous beading is irregular caliber change in retinal veins, indicating severe ischemia.",
    "Intraretinal microvascular abnormalities (IRMA) are dilated capillaries in areas of ischemia.",
    "Neovascularization of the disc (NVD) carries higher risk than neovascularization elsewhere (NVE).",
    "Vitreous hemorrhage is a common presentation of advanced proliferative DR.",
    "Tractional retinal detachment is a late complication of proliferative DR with fibrous proliferation.",
    "Diabetic macular edema (DME) is retinal thickening within 500 microns of the fovea center.",
    "Clinically significant macular edema (CSME) is diagnosed when thickening threatens the fovea.",
    "Optical coherence tomography (OCT) is the standard for detecting and monitoring DME.",
    "Anti-VEGF injections are first-line treatment for center-involving DME.",
    "Laser photocoagulation remains useful for non-center-involving DME and proliferative DR.",
    "Intravitreal corticosteroids are an option for refractory DME.",
    "Vitrectomy is indicated for non-clearing vitreous hemorrhage and tractional detachment.",
    "Annual screening is recommended for all patients with type 2 diabetes.",
    "Type 1 diabetic patients should be screened within 5 years of diagnosis.",
    "Pregnant diabetic patients require screening before conception and each trimester.",
    "Non-mydriatic fundus photography is a validated screening modality.",
    "Mydriatic photography provides superior image quality compared to non-mydriatic.",
    "Adequate fundus image requires at least a 45-degree field of view.",
    "Image quality scores below 0.5 often compromise diagnostic accuracy.",
    "Graders should be masked to patient identity and clinical history.",
    "The International Clinical Diabetic Retinopathy scale defines 5 severity levels.",
    "Grade 0 corresponds to no apparent retinopathy.",
    "Grade 1 corresponds to mild non-proliferative DR.",
    "Grade 2 corresponds to moderate non-proliferative DR.",
    "Grade 3 corresponds to severe non-proliferative DR.",
    "Grade 4 corresponds to proliferative diabetic retinopathy.",
    "Deep learning models have achieved expert-level DR grading performance.",
    "Public DR datasets include EyePACS, APTOS 2019, Messidor, and DDR.",
    "The APTOS 2019 dataset contains approximately 3,662 labeled retinal images.",
    "Class imbalance is a major challenge in DR datasets.",
    "Data augmentation improves DR classifier robustness to appearance variation.",
    "Geometric augmentation includes rotation, flip, and translation.",
    "Photometric augmentation includes brightness, contrast, and color jitter.",
    "Synthetic data generation can address class imbalance in medical imaging.",
    "Generative adversarial networks (GANs) have been widely applied to fundus synthesis.",
    "Diffusion models have recently surpassed GANs in image fidelity.",
    "Denoising diffusion probabilistic models iteratively denoise Gaussian noise.",
    "Latent diffusion models operate in a compressed latent space for efficiency.",
    "Conditional diffusion allows generation conditioned on class or text.",
    "Class-conditional diffusion improves label-image alignment over unconditional.",
    "Schema-enforced JSON generation constrains LLM outputs to a target structure.",
    "Grammar-constrained decoding reduces post-hoc repair requirements.",
    "Retrieval-augmented generation grounds LLM outputs in external knowledge.",
    "Small knowledge bases (under 20 documents) limit RAG grounding effectiveness.",
    "Distributional fidelity metrics quantify the gap between real and synthetic data.",
    "Jensen-Shannon divergence measures distributional distance between two samples.",
    "Frechet Inception Distance is a standard metric for image generation quality.",
    "Expected calibration error quantifies a classifier's confidence reliability.",
    "Per-class performance is essential in imbalanced medical classification.",
]


def main():
    with open(OUT, "w") as f:
        for i, text in enumerate(DOCS):
            f.write(json.dumps({"id": f"DR{i:03d}", "text": text}) + "\n")
    print(f"Wrote {len(DOCS)} documents to {OUT}")


if __name__ == "__main__":
    main()