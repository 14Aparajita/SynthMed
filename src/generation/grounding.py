import json
import logging
from typing import Dict, Any, List, Optional
from ..retrieval.rag_fusion import RAGFusion
from .metadata_generator import MetadataGenerator

logger = logging.getLogger("synthmed.generation")

class GroundedGenerator:
    """
    Knowledge-grounded generation pipeline.
    Combines RAG retrieval with metadata generation for grounded synthetic data.
    """
    
    def __init__(
        self,
        metadata_generator: MetadataGenerator,
        rag_fusion: RAGFusion,
    ):
        self.metadata_generator = metadata_generator
        self.rag_fusion = rag_fusion
    
    def generate_grounded(
        self,
        dr_grade: int,
        num_records: int = 1,
        use_grounding: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate metadata records grounded in retrieved knowledge.
        
        Args:
            dr_grade: DR severity grade (0-4)
            num_records: Number of records to generate
            use_grounding: Whether to use RAG grounding
        
        Returns:
            List of generated metadata records
        """
        records = []
        
        for _ in range(num_records):
            context = None
            grounding_score = 0.0
            
            if use_grounding:
                # Build query from DR grade
                query = self._build_retrieval_query(dr_grade)
                
                # Retrieve relevant documents
                retrieved = self.rag_fusion.retrieve(query)
                retrieved_docs = [doc for doc, _, _ in retrieved]
                context = "\n".join(retrieved_docs[:3])
            
            # Generate metadata with context
            generated = self.metadata_generator.generate_structured(
                dr_grade=dr_grade,
                context=context,
                num_records=1
            )
            
            if generated:
                record = generated[0]
                
                # Compute grounding score if applicable
                if use_grounding and retrieved_docs:
                    record_text = json.dumps(record)
                    grounding_score = self.rag_fusion.compute_grounding_score(
                        record_text, retrieved_docs
                    )
                
                record["_grounding_score"] = grounding_score
                record["_grounding_enabled"] = use_grounding
                records.append(record)
        
        return records
    
    def _build_retrieval_query(self, dr_grade: int) -> str:
        """Build retrieval query based on DR grade."""
        grade_queries = {
            0: "normal retinal examination findings healthy retina",
            1: "mild non-proliferative diabetic retinopathy microaneurysms only",
            2: "moderate diabetic retinopathy microaneurysms hemorrhages exudates",
            3: "severe non-proliferative diabetic retinopathy venous beading intraretinal hemorrhages",
            4: "proliferative diabetic retinopathy neovascularization vitreous hemorrhage"
        }
        
        return f"Clinical findings for {grade_queries.get(dr_grade, 'diabetic retinopathy')}"