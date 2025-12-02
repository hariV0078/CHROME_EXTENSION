"""
Unit tests for the deterministic job-resume matching system.
"""

import unittest
import logging
from matching import match_job_resume
from matching.scoring_engine import (
    calculate_skills_score,
    calculate_experience_score,
    calculate_responsibility_score,
    calculate_education_score,
    calculate_seniority_score
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


# Sample data
SAMPLE_JOB_DESCRIPTION = """
Data Annotator - AI/ML Team

Expert Global Solutions Pvt. Ltd is seeking a Data Annotator for our AI/ML car platform projects.

Requirements:
- Experience: 0 to 1 Year
- Qualification: B.E. / M.Tech (CSE/ETC/EE/Mechanical/Instrumental) / MCA / MSc
- Skills: Python, Machine Learning, Data Analysis
- Understanding of AI/ML concepts and data curation
- Ability to clean and organize data for machine learning models

Responsibilities:
- Annotate data for machine learning models
- Clean and curate datasets
- Quality check labeled data
- Work with annotation tools
- Collaborate with ML engineers

Location: Coimbatore, Tamil Nadu
Training: 2 weeks project training in Aurangabad
"""

SAMPLE_RESUME = """
Hari Varadhan N R
Email: hshri511@gmail.com | Phone: +919597103099

EDUCATION
Sri Krishna College of Engineering and Technology
B.E. in Computer Science and Engineering (Expected: April 2027)

SKILLS
Programming: Python, C++, Java
ML/AI: TensorFlow, Keras, PyTorch, OpenCV, Computer Vision
Data: Pandas, NumPy, Tableau, MySQL
Tools: Git, Nvidia Jetson, N8n

EXPERIENCE (1.42 years total)

AI Research Intern | Arivara AI | Nov-Dec 2024
- Built AI-powered classroom tutor using GPT-4o-mini
- Implemented hybrid retrieval with Weaviate, Neo4j, Redis
- Developed RAG+LLM model for document eligibility

AI/ML Practitioner | Various Projects | 2023-2024
- Crop disease detection using TensorFlow/Keras
- Deepfake detection with DenseNet and ViT
- Voice sales assistant with N8n workflows
- Computer vision and OCR projects

PROJECTS
- Hybrid retrieval system with vector DB and knowledge graphs
- CNN models for image classification
- Data augmentation pipelines
- Automation workflows for CRM

CERTIFICATIONS
- Supervised Machine Learning (DeepLearning.AI, Coursera)

INTERESTS
AI research, Machine Learning, Computer Vision, Hackathons
"""


class TestScoringComponents(unittest.TestCase):
    """Test individual scoring components."""
    
    def test_skills_score_perfect_match(self):
        """Test skills scoring with perfect match."""
        score = calculate_skills_score(
            ["Python", "Machine Learning", "SQL"],
            ["Docker", "AWS"],
            ["Python", "Machine Learning", "SQL", "Docker", "AWS"]
        )
        self.assertEqual(score, 100.0)
    
    def test_skills_score_partial_match(self):
        """Test skills scoring with partial match."""
        score = calculate_skills_score(
            ["Python", "Java", "SQL"],  # Required
            ["Docker"],  # Preferred
            ["Python", "SQL"]  # Candidate has 2/3 required, 0/1 preferred
        )
        # Expected: (0.70 * 66.67) + (0.30 * 0) = 46.67
        self.assertAlmostEqual(score, 46.67, places=1)
    
    def test_experience_score_meets_requirement(self):
        """Test experience scoring when candidate meets requirement."""
        score = calculate_experience_score(1.0, 1.5)
        self.assertEqual(score, 100.0)  # Capped at 100
    
    def test_experience_score_below_requirement(self):
        """Test experience scoring when candidate is below requirement."""
        score = calculate_experience_score(2.0, 1.0)
        # Expected: (1.0 / 2.0) * 100 * 0.7 = 35.0
        self.assertEqual(score, 35.0)
    
    def test_education_score_exact_match(self):
        """Test education scoring with exact match."""
        score = calculate_education_score(
            "Bachelor", "Computer Science",
            "Bachelor", "Computer Science"
        )
        self.assertEqual(score, 100.0)
    
    def test_education_score_related_field(self):
        """Test education scoring with related field."""
        score = calculate_education_score(
            "Bachelor", "Computer Science",
            "Bachelor", "Software Engineering"
        )
        # Expected: 100 * 0.85 = 85.0
        self.assertEqual(score, 85.0)
    
    def test_seniority_score_exact_match(self):
        """Test seniority scoring with exact match."""
        score = calculate_seniority_score("Mid", "Mid")
        self.assertEqual(score, 100.0)
    
    def test_seniority_score_one_above(self):
        """Test seniority scoring when candidate is one level above."""
        score = calculate_seniority_score("Mid", "Senior")
        self.assertEqual(score, 90.0)
    
    def test_seniority_score_one_below(self):
        """Test seniority scoring when candidate is one level below."""
        score = calculate_seniority_score("Mid", "Entry")
        self.assertEqual(score, 60.0)


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end matching with sample data."""
    
    def test_sample_job_resume_match(self):
        """Test matching with sample job and resume."""
        # This test requires API key and makes actual LLM calls
        # Skip if no API key available
        import os
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set")
        
        result = match_job_resume(SAMPLE_JOB_DESCRIPTION, SAMPLE_RESUME)
        
        # Assertions
        self.assertIn("match_percentage", result)
        self.assertIn("breakdown", result)
        self.assertIn("extracted_data", result)
        
        # Match percentage should be between 0 and 100
        self.assertGreaterEqual(result["match_percentage"], 0)
        self.assertLessEqual(result["match_percentage"], 100)
        
        # Breakdown should have all components
        breakdown = result["breakdown"]
        self.assertIn("skills", breakdown)
        self.assertIn("experience", breakdown)
        self.assertIn("responsibilities", breakdown)
        self.assertIn("education", breakdown)
        self.assertIn("seniority", breakdown)
        
        # For this sample data, we expect a good match (60-80%)
        # Hari has strong ML/Python skills relevant to data annotation
        self.assertGreater(result["match_percentage"], 60,
                          "Expected match > 60% for ML candidate with data annotation role")
        
        print(f"\n{'='*60}")
        print(f"SAMPLE MATCH RESULT")
        print(f"{'='*60}")
        print(f"Match Percentage: {result['match_percentage']}%")
        print(f"\nBreakdown:")
        for component, score in breakdown.items():
            print(f"  {component.capitalize()}: {score}%")
        print(f"{'='*60}\n")


class TestDeterminism(unittest.TestCase):
    """Test that scoring is deterministic."""
    
    def test_skills_score_determinism(self):
        """Test that skills scoring is deterministic."""
        # Same inputs should produce same outputs
        score1 = calculate_skills_score(
            ["Python", "Java"],
            ["Docker"],
            ["Python"]
        )
        score2 = calculate_skills_score(
            ["Python", "Java"],
            ["Docker"],
            ["Python"]
        )
        self.assertEqual(score1, score2)
    
    def test_experience_score_determinism(self):
        """Test that experience scoring is deterministic."""
        score1 = calculate_experience_score(2.0, 1.5)
        score2 = calculate_experience_score(2.0, 1.5)
        self.assertEqual(score1, score2)


if __name__ == "__main__":
    unittest.main()

