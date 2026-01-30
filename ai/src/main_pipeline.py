"""
WCAG-VLM Evaluation Framework - Main Pipeline
Automated Web Accessibility Assessment using Vision-Language Models

Author: [Research Team]
Institution: University of Salamanca
License: Apache 2.0
"""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

import numpy as np
import pandas as pd
from PIL import Image
import base64
import io

# Vision-Language Models (Lazy loaded)
# from transformers import (
#     AutoProcessor, 
#     AutoModelForVision2Seq,
#     LlavaNextProcessor,
#     LlavaNextForConditionalGeneration
# )

# Async web scraping
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
import pyppeteer

# Metrics & evaluation
# from sklearn.metrics import (
#     precision_score, recall_score, f1_score,
#     confusion_matrix, roc_auc_score, cohen_kappa_score
# )

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES & CONFIGURATION
# ============================================================================

@dataclass
class WCAGCriterion:
    """Represents a single WCAG 2.2 AA criterion for evaluation."""
    
    criterion_id: str  # e.g., "1.4.3"
    name: str
    description: str
    level: str  # "A", "AA", "AAA"
    category: str  # "perceivable", "operable", "understandable", "robust"
    automation_capability: str  # "full", "partial", "manual_only"
    labels: Dict[str, str]  # Possible violation categories
    prompt_template: str


class WCAGEvaluationConfig:
    """Configuration for the entire evaluation pipeline."""
    
    # Model configuration
    MODELS = {
        "llava-7b": {
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "processor_id": "llava-hf/llava-1.5-7b-hf",
            "quantization": None,
            "device": "cuda:0"
        },
        "llava-13b": {
            "model_id": "llava-hf/llava-1.5-13b-hf",
            "processor_id": "llava-hf/llava-1.5-13b-hf",
            "quantization": None,
            "device": "cuda:0"
        },
        "qwen-vl": {
            "model_id": "Qwen/Qwen-VL-Plus",
            "processor_id": "Qwen/Qwen-VL-Plus",
            "quantization": "int8",
            "device": "cuda:1"
        }
    }
    
    # Evaluation settings
    WCAG_CRITERIA_FOCUS = ["1.4.3", "1.1.1", "3.3.2", "2.4.10", "2.1.1", 
                           "4.1.2", "1.3.1", "2.4.4"]  # 8 focus criteria
    
    BATCH_SIZE = 8
    MAX_WORKERS = 4  # Async web scraping workers
    TIMEOUT_SECONDS = 30
    
    # Paths
    DATA_DIR = Path("data")
    SCREENSHOTS_DIR = DATA_DIR / "screenshots"
    RESULTS_DIR = DATA_DIR / "results"
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    
    # Prompts
    # LLaVA-1.5 specific template: USER: <image>\n<prompt>\nASSISTANT:
    WCAG_EVALUATION_TEMPLATE = """USER: <image>
    Analyze this website screenshot for accessibility issues according to WCAG 2.2 AA.
    Focus specifically on criterion {criterion_id}: {criterion_name}.
    
    Examine the image carefully for:
    {examination_focus}
    
    Respond with a JSON object containing:
    {{
        "criterion_id": "{criterion_id}",
        "status": "PASS" | "FAIL" | "UNCLEAR",
        "severity": "Critical" | "Major" | "Minor" | "N/A",
        "issue_descriptions": [string list of specific issues found],
        "affected_elements": [string list of HTML elements],
        "evidence_snippets": [string list of specific evidence],
        "suggested_fixes": [string list of concrete fixes],
        "confidence_score": 0.0-1.0,
        "reasoning": "string explanation of decision",
        "related_criteria": [string list of related WCAG criteria]
    }}
    
    Be precise and evidence-based. Only report issues you can clearly identify from the visual.
    ASSISTANT:
    """


# ============================================================================
# WCAG CRITERIA DEFINITIONS
# ============================================================================

WCAG_CRITERIA_DEFINITIONS = {
    "1.4.3": {
        "name": "Contrast (Minimum)",
        "description": "Text and images of text have a contrast ratio of at least 4.5:1",
        "level": "AA",
        "category": "perceivable",
        "automation": "full",
        "examination_focus": [
            "- Measure contrast ratio between text and background",
            "- Check for adequate color contrast (4.5:1 for normal, 3:1 for large text)",
            "- Look for low-contrast text elements",
            "- Examine buttons, links, and interactive elements"
        ]
    },
    "1.1.1": {
        "name": "Non-text Content",
        "description": "All images have meaningful alternative text",
        "level": "A",
        "category": "perceivable",
        "automation": "partial",
        "examination_focus": [
            "- Identify all images on the page",
            "- Verify alt text presence for informative images",
            "- Check if alt text is descriptive and meaningful",
            "- Verify decorative images are properly marked"
        ]
    },
    # ... continue for other 6 criteria
}


# ============================================================================
# WEB SCRAPING MODULE
# ============================================================================

class AccessibilityWebScraper:
    """Captures website screenshots and DOM structure for accessibility evaluation."""
    
    def __init__(self, headless=True, timeout=30):
        self.headless = headless
        self.timeout = timeout
        self.session = None
    
    async def capture_website(self, url: str) -> Optional[Dict]:
        """
        Capture complete website state.
        
        Returns:
        {
            "url": str,
            "timestamp": ISO8601,
            "screenshot_b64": base64 PNG,
            "screenshot_path": Path,
            "dom_html": str,
            "page_metadata": {...},
            "images": [...],
            "forms": [...],
            "headings": [...],
            "links": [...]
        }
        """
        try:
            # Use Pyppeteer for async headless browser
            browser = await pyppeteer.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )
            page = await browser.newPage()
            
            # Navigation
            await page.goto(url, {'waitUntil': 'networkidle2', 'timeout': self.timeout * 1000})
            
            # Get full page height for scrolling
            viewport = await page.evaluate('() => ({width: window.innerWidth, height: document.body.scrollHeight})')
            await page.setViewport({'width': viewport['width'], 'height': viewport['height']})
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot({'fullPage': True})
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Get DOM
            dom_html = await page.content()
            
            # Extract metadata
            page_metadata = await page.evaluate('''
                () => ({
                    title: document.title,
                    lang: document.documentElement.lang || "unknown",
                    viewport: document.querySelector('meta[name="viewport"]')?.content,
                    hasLandmark: !!document.querySelector('main, [role="main"]')
                })
            ''')
            
            # Extract images
            images = await page.evaluate('''
                () => Array.from(document.querySelectorAll('img')).map(img => ({
                    src: img.src,
                    alt: img.alt,
                    role: img.getAttribute('role'),
                    ariaLabel: img.getAttribute('aria-label')
                }))
            ''')
            
            # Extract forms
            forms = await page.evaluate('''
                () => Array.from(document.querySelectorAll('form')).map(form => ({
                    name: form.name,
                    inputs: Array.from(form.querySelectorAll('input, select, textarea')).map(inp => ({
                        name: inp.name,
                        type: inp.type,
                        label: form.querySelector(`label[for="${inp.id}"]`)?.textContent
                    }))
                }))
            ''')
            
            # Extract headings
            headings = await page.evaluate('''
                () => Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                    level: parseInt(h.tagName[1]),
                    text: h.textContent
                }))
            ''')
            
            await browser.close()
            
            return {
                "url": url,
                "timestamp": datetime.isoformat(datetime.now()),
                "screenshot_b64": screenshot_b64,
                "dom_html": dom_html,
                "page_metadata": page_metadata,
                "images": images,
                "forms": forms,
                "headings": headings,
                "http_status": 200
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    async def batch_capture(self, url_list: List[str], max_workers: int = 4):
        """Parallel website capture with progress tracking."""
        semaphore = asyncio.Semaphore(max_workers)
        
        async def bounded_capture(url):
            async with semaphore:
                logger.info(f"Capturing {url}")
                return await self.capture_website(url)
        
        tasks = [bounded_capture(url) for url in url_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None]


# ============================================================================
# VLM INFERENCE MODULE
# ============================================================================

class VLMEvaluator:
    """Performs WCAG evaluation using Vision-Language Models."""
    
    def __init__(self, model_name: str = "llava-7b"):
        if torch is None:
            raise ImportError("PyTorch is required for VLMEvaluator but not installed.")
            
        from transformers import (
            AutoProcessor, 
            AutoModelForVision2Seq
        )
        self.model_name = model_name
        config = WCAGEvaluationConfig.MODELS[model_name]
        
        # Use Auto classes for automatic architecture detection
        self.processor = AutoProcessor.from_pretrained(config["processor_id"])
        self.model = AutoModelForVision2Seq.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.device = config["device"]
        self.model.to(self.device)
    
    def evaluate_page(self, screenshot_b64: str, criteria: List[str]) -> Dict:
        """
        Evaluate a single page for multiple WCAG criteria.
        
        Args:
            screenshot_b64: Base64-encoded screenshot
            criteria: List of criterion IDs (e.g., ["1.4.3", "1.1.1"])
        
        Returns:
            {
                "page_evaluations": {...},
                "macro_scores": {...},
                "inference_time": float,
                "model": str
            }
        """
        import time
        start_time = time.time()
        
        # Decode screenshot
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(image_data))
        
        evaluations = {}
        
        for criterion_id in criteria:
            if criterion_id not in WCAG_CRITERIA_DEFINITIONS:
                logger.warning(f"Unknown criterion: {criterion_id}")
                continue
            
            criterion = WCAG_CRITERIA_DEFINITIONS[criterion_id]
            
            # Build prompt
            prompt = WCAGEvaluationConfig.WCAG_EVALUATION_TEMPLATE.format(
                criterion_id=criterion_id,
                criterion_name=criterion["name"],
                examination_focus="\n".join(criterion["examination_focus"])
            )
            
            # Inference
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Parse JSON response
            try:
                # 0. Split off the prompt (Model often repeats it)
                if "ASSISTANT:" in response_text:
                    clean_text = response_text.split("ASSISTANT:")[-1]
                else:
                    clean_text = response_text
                
                # 1. Clean Markdown Code Blocks
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0]
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0]
                
                # 2. Global Escape Cleanup (Fix for _ being escaped as \_)
                clean_text = clean_text.replace(r'\_', '_')
                
                # 3. Extract JSON using Regex
                import re
                json_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                else:
                    logger.warning(f"No JSON found in response for {criterion_id}: {response_text[:100]}...")
                    result = {"error": "No JSON found", "raw_response": response_text}
                        
            except json.JSONDecodeError as e:
                logger.warning(f"JSON Decode Error for {criterion_id}: {str(e)}")
                logger.warning(f"RAW RESPONSE that failed: {response_text}")  # DEBUG LOG
                result = {"error": "JSON decode error", "raw_response": response_text}
            
            evaluations[criterion_id] = result
        
        elapsed = time.time() - start_time
        
        return {
            "page_evaluations": evaluations,
            "inference_time": elapsed,
            "model": self.model_name,
            "num_criteria": len(evaluations)
        }


# ============================================================================
# METRICS & EVALUATION MODULE
# ============================================================================

class MetricsCalculator:
    """Computes evaluation metrics: precision, recall, F1, etc."""
    
    @staticmethod
    def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, cohen_kappa_score
        )
        """
        Compute standard classification metrics.
        
        Args:
            predictions: Predicted labels (0=PASS, 1=FAIL)
            ground_truth: Expert labels (0=PASS, 1=FAIL)
        
        Returns:
            Dictionary with precision, recall, F1, specificity, MCC
        """
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions, labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews Correlation Coefficient (good for imbalanced data)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "mcc": mcc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
    
    @staticmethod
    def compute_criterion_metrics(results_df: pd.DataFrame, criterion_id: str) -> Dict:
        """Compute metrics for a single criterion across all pages."""
        subset = results_df[results_df['criterion_id'] == criterion_id]
        
        predictions = subset['model_prediction'].values
        ground_truth = subset['ground_truth'].values
        
        return MetricsCalculator.compute_metrics(predictions, ground_truth)


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

class WCAGEvaluationPipeline:
    """Orchestrates the full evaluation workflow."""
    
    def __init__(self, output_dir: Path = Path("results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scraper = AccessibilityWebScraper()
        self.evaluators = {}  # Cache for VLM evaluators
    
    async def run_full_evaluation(self, url_list: List[str], models: List[str] = None,
                                   criteria: List[str] = None):
        """
        Execute complete evaluation pipeline:
        1. Web scraping
        2. VLM inference (multiple models)
        3. Metrics computation
        4. Comparative analysis
        5. Result reporting
        """
        if models is None:
            models = ["llava-7b"]  # Default to smallest model
        if criteria is None:
            criteria = WCAGEvaluationConfig.WCAG_CRITERIA_FOCUS
        
        logger.info(f"Starting evaluation: {len(url_list)} URLs × {len(models)} models × {len(criteria)} criteria")
        
        # Stage 1: Web Scraping
        logger.info("Stage 1: Web Scraping")
        scraped_pages = await self.scraper.batch_capture(url_list)
        logger.info(f"Successfully scraped {len(scraped_pages)} pages")
        
        # Stage 2: VLM Inference
        logger.info("Stage 2: VLM Inference")
        results_data = []
        
        # Create a checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        for i, page in enumerate(scraped_pages):
            for model_name in models:
                if model_name not in self.evaluators:
                    self.evaluators[model_name] = VLMEvaluator(model_name)
                
                evaluator = self.evaluators[model_name]
                try:
                    eval_result = evaluator.evaluate_page(page['screenshot_b64'], criteria)
                    
                    results_data.append({
                        "url": page['url'],
                        "model": model_name,
                        "evaluation": eval_result,
                        "timestamp": datetime.isoformat(datetime.now())
                    })
                except Exception as e:
                    logger.error(f"Failed to evaluate {page['url']}: {e}")
                    continue

            # Incremental Checkpointing every 10 pages
            if (i + 1) % 10 == 0:
                intermediate_df = pd.DataFrame(results_data)
                ckpt_path = checkpoint_dir / f"checkpoint_batch_{i+1}.parquet"
                intermediate_df.to_parquet(ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path} ({len(results_data)} records)")
        
        # Stage 3: Save results
        results_df = pd.DataFrame(results_data)
        output_file = self.output_dir / f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        results_df.to_parquet(output_file)
        logger.info(f"Results saved to {output_file}")
        
        return results_df


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Example usage of the evaluation pipeline."""
    
    # Sample URLs for testing
    test_urls = [
        "https://example.com",
        "https://wikipedia.org",
        "https://github.com"
    ]
    
    pipeline = WCAGEvaluationPipeline(output_dir=Path("wcag_results"))
    results = await pipeline.run_full_evaluation(
        url_list=test_urls,
        models=["llava-7b"],  # Start with smallest model for testing
        criteria=["1.4.3", "1.1.1", "3.3.2"]  # Subset for demo
    )
    
    logger.info("Evaluation complete!")
    logger.info(results.head())


if __name__ == "__main__":
    asyncio.run(main())
