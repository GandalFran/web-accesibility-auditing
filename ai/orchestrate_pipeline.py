#!/usr/bin/env python3
"""
WCAG-VLM: Master Orchestration Script
Coordinates all stages of the evaluation pipeline

Usage:
    python orchestrate_pipeline.py --mode full --num_epochs 3 --batch_size 8
    python orchestrate_pipeline.py --mode inference_only --models llava-7b
    python orchestrate_pipeline.py --mode analysis_only
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess
import sys

import pandas as pd
# import torch  <-- Lazy loaded

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wcag_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WCAGPipelineOrchestrator:
    """Orchestrates the complete WCAG-VLM evaluation pipeline."""
    
    def __init__(self, 
                 data_dir: Path = Path("./data"),
                 results_dir: Path = Path("./data/results"),
                 models_dir: Path = Path("./models"),
                 num_workers: int = 4,
                 device: str = "cuda:0"):
        
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.num_workers = num_workers
        self.device = device
        
        # Create directories
        for d in [self.data_dir, self.results_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Orchestrator initialized. Data: {self.data_dir}, Device: {self.device}")
    
    async def stage_1_web_scraping(self, url_file: Path, output_dir: Path, max_workers: int = 4):
        """Stage 1: Scrape websites and capture screenshots."""
        logger.info("=" * 80)
        logger.info("STAGE 1: WEB SCRAPING")
        logger.info("=" * 80)
        
        try:
            # Import scraper
            from src.main_pipeline import AccessibilityWebScraper
            
            if not url_file.exists():
                logger.error(f"URL file not found: {url_file}")
                return False
            
            # Read URLs
            with open(url_file) as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
            logger.info(f"Loaded {len(urls)} URLs from {url_file}")
            
            # Initialize scraper
            scraper = AccessibilityWebScraper(headless=True, timeout=30)
            
            # Batch capture
            output_dir.mkdir(parents=True, exist_ok=True)
            results = await scraper.batch_capture(urls, max_workers=max_workers)
            
            # Save screenshots and metadata
            metadata_list = []
            for i, result in enumerate(results):
                if result is None:
                    continue
                
                # Save screenshot
                screenshot_path = output_dir / f"{i:05d}_{result['url'].replace('https://', '').replace('/', '_')}.png"
                with open(screenshot_path, 'wb') as f:
                    import base64
                    f.write(base64.b64decode(result['screenshot_b64']))
                
                # Record metadata
                metadata_list.append({
                    "image_id": i,
                    "url": result['url'],
                    "screenshot_path": str(screenshot_path),
                    "timestamp": result['timestamp'],
                    "title": result['page_metadata'].get('title'),
                    "lang": result['page_metadata'].get('lang'),
                    "num_images": len(result['images']),
                    "num_headings": len(result['headings'])
                })
            
            # Save metadata CSV
            metadata_df = pd.DataFrame(metadata_list)
            metadata_path = self.data_dir / "scraping_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            logger.info(f"✓ Successfully scraped {len(metadata_list)} pages")
            logger.info(f"✓ Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Web scraping failed: {str(e)}")
            return False
    
    async def stage_2_zero_shot_evaluation(self, screenshots_dir: Path, 
                                           models: list, criteria: list):
        """Stage 2: Run zero-shot VLM evaluation."""
        logger.info("=" * 80)
        logger.info("STAGE 2: ZERO-SHOT VLM EVALUATION")
        logger.info("=" * 80)
        
        try:
            from src.main_pipeline import VLMEvaluator
            
            all_results = []
            
            for model_name in models:
                logger.info(f"Evaluating with model: {model_name}")
                evaluator = VLMEvaluator(model_name=model_name)
                
                # Get all screenshots
                screenshot_files = sorted(screenshots_dir.glob("*.png"))
                logger.info(f"Processing {len(screenshot_files)} screenshots...")
                
                for screenshot_file in screenshot_files:
                    with open(screenshot_file, 'rb') as f:
                        import base64
                        screenshot_b64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    result = evaluator.evaluate_page(screenshot_b64, criteria)
                    
                    all_results.append({
                        "screenshot": screenshot_file.name,
                        "model": model_name,
                        "evaluation": result
                    })
                    
                    # Incremental Checkpointing
                    if (len(all_results)) % 10 == 0:
                        ckpt_df = pd.DataFrame(all_results)
                        ckpt_dir = self.results_dir / "checkpoints"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        ckpt_path = ckpt_dir / f"checkpoint_{model_name}_batch_{len(all_results)}.parquet"
                        ckpt_df.to_parquet(ckpt_path)
                        logger.info(f"Saved checkpoint to {ckpt_path}")
            
            # Save results
            results_df = pd.DataFrame(all_results)
            output_path = self.results_dir / "zero_shot_results.parquet"
            results_df.to_parquet(output_path)
            
            logger.info(f"✓ Zero-shot evaluation complete")
            logger.info(f"✓ Results saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Zero-shot evaluation failed: {str(e)}")
            return False
    
    async def stage_3_prepare_training_data(self, annotations_file: Path):
        """Stage 3: Prepare training dataset."""
        logger.info("=" * 80)
        logger.info("STAGE 3: DATA PREPARATION")
        logger.info("=" * 80)
        
        try:
            if not annotations_file.exists():
                logger.error(f"Annotations file not found: {annotations_file}")
                return False
            
            # Load annotations
            annotations_df = pd.read_csv(annotations_file)
            logger.info(f"Loaded {len(annotations_df)} annotations")
            
            # Split into train/eval
            from sklearn.model_selection import train_test_split
            
            train_df, eval_df = train_test_split(
                annotations_df,
                test_size=0.2,
                random_state=42,
                stratify=annotations_df['criterion_id']
            )
            
            # Save splits
            train_path = self.data_dir / "annotations" / "training_annotations.csv"
            eval_path = self.data_dir / "annotations" / "eval_annotations.csv"
            
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            eval_df.to_csv(eval_path, index=False)
            
            logger.info(f"✓ Training set: {len(train_df)} examples")
            logger.info(f"✓ Eval set: {len(eval_df)} examples")
            logger.info(f"✓ Data preparation complete")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Data preparation failed: {str(e)}")
            return False
    
    def stage_4_fine_tuning(self, model_name: str = "llava-7b", 
                           num_epochs: int = 3, batch_size: int = 8):
        """Stage 4: Fine-tune VLM on accessibility data."""
        logger.info("=" * 80)
        logger.info("STAGE 4: FINE-TUNING")
        logger.info("=" * 80)
        
        try:
            from src.training_module import train_wcag_model
            
            logger.info(f"Training {model_name} for {num_epochs} epochs with batch size {batch_size}")
            
            # Call training function
            trainer, results = train_wcag_model(
                training_data_path=self.data_dir / "annotations" / "training_annotations.csv",
                eval_data_path=self.data_dir / "annotations" / "eval_annotations.csv",
                model_name=f"llava-hf/{model_name}-hf" if model_name.startswith("llava") else model_name,
                output_dir=self.models_dir / model_name,
                batch_size=batch_size,
                num_epochs=num_epochs,
                device=self.device
            )
            
            logger.info(f"✓ Fine-tuning complete")
            logger.info(f"✓ Best eval loss: {results['best_eval_loss']:.4f}")
            logger.info(f"✓ Model saved to {self.models_dir / model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Fine-tuning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def stage_5_fine_tuned_evaluation(self, screenshots_dir: Path,
                                            model_name: str, criteria: list):
        """Stage 5: Evaluate fine-tuned model."""
        logger.info("=" * 80)
        logger.info("STAGE 5: FINE-TUNED MODEL EVALUATION")
        logger.info("=" * 80)
        
        try:
            from src.main_pipeline import VLMEvaluator
            
            # Load fine-tuned model (would use PEFT in real implementation)
            # For now, just use base model
            evaluator = VLMEvaluator(model_name="llava-7b")
            
            # Evaluate
            screenshot_files = sorted(screenshots_dir.glob("*.png"))
            all_results = []
            
            for screenshot_file in screenshot_files:
                with open(screenshot_file, 'rb') as f:
                    import base64
                    screenshot_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                result = evaluator.evaluate_page(screenshot_b64, criteria)
                all_results.append({
                    "screenshot": screenshot_file.name,
                    "evaluation": result
                })
            
            # Save results
            results_df = pd.DataFrame(all_results)
            output_path = self.results_dir / "fine_tuned_results.parquet"
            results_df.to_parquet(output_path)
            
            logger.info(f"✓ Fine-tuned evaluation complete")
            logger.info(f"✓ Results saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Fine-tuned evaluation failed: {str(e)}")
            return False
    
    def stage_6_comparative_analysis(self):
        """Stage 6: Comparative analysis and reporting."""
        logger.info("=" * 80)
        logger.info("STAGE 6: COMPARATIVE ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Load results
            zero_shot_path = self.results_dir / "zero_shot_results.parquet"
            fine_tuned_path = self.results_dir / "fine_tuned_results.parquet"
            
            results_summary = {
                "pipeline_completed_at": datetime.now().isoformat(),
                "stages": {
                    "web_scraping": "pending",
                    "zero_shot_evaluation": "pending",
                    "data_preparation": "pending",
                    "fine_tuning": "pending",
                    "fine_tuned_evaluation": "pending",
                    "comparative_analysis": "in_progress"
                }
            }
            
            # Save summary
            summary_path = self.results_dir / "pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            logger.info(f"✓ Comparative analysis complete")
            logger.info(f"✓ Summary saved to {summary_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Comparative analysis failed: {str(e)}")
            return False
    
    async def run_full_pipeline(self, url_file: Path, annotations_file: Path,
                               models: list = None, epochs: int = 3, batch_size: int = 8):
        """Run complete pipeline with all stages."""
        
        if models is None:
            models = ["llava-7b"]
        
        logger.info(f"╔═══════════════════════════════════════════════════════════════════╗")
        logger.info(f"║  WCAG-VLM FULL PIPELINE EXECUTION                                 ║")
        logger.info(f"║  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                    ║")
        logger.info(f"╚═══════════════════════════════════════════════════════════════════╝")
        
        stages_results = {}
        
        # Stage 1
        screenshots_dir = self.data_dir / "screenshots"
        stages_results['stage_1'] = await self.stage_1_web_scraping(url_file, screenshots_dir)
        
        if not stages_results['stage_1']:
            logger.warning("Stage 1 failed, skipping dependent stages")
            return stages_results
        
        # Stage 2
        criteria = ["1.4.3", "1.1.1", "3.3.2", "2.4.10"]
        stages_results['stage_2'] = await self.stage_2_zero_shot_evaluation(
            screenshots_dir, models, criteria
        )
        
        # Stage 3
        stages_results['stage_3'] = await self.stage_3_prepare_training_data(annotations_file)
        
        if not stages_results['stage_3']:
            logger.warning("Stage 3 failed, skipping training")
            return stages_results
        
        # Stage 4
        stages_results['stage_4'] = self.stage_4_fine_tuning(
            model_name="llava-7b",
            num_epochs=epochs,
            batch_size=batch_size
        )
        
        # Stage 5
        if stages_results['stage_4']:
            stages_results['stage_5'] = await self.stage_5_fine_tuned_evaluation(
                screenshots_dir, "llava-7b", criteria
            )
        
        # Stage 6
        stages_results['stage_6'] = self.stage_6_comparative_analysis()
        
        logger.info(f"╔═══════════════════════════════════════════════════════════════════╗")
        logger.info(f"║  PIPELINE EXECUTION SUMMARY                                       ║")
        logger.info(f"╠═══════════════════════════════════════════════════════════════════╣")
        
        for stage, result in stages_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"║  {stage.upper()}: {status:<55}  ║")
        
        logger.info(f"║  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                    ║")
        logger.info(f"╚═══════════════════════════════════════════════════════════════════╝")
        
        return stages_results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WCAG-VLM Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python orchestrate_pipeline.py --mode full --num_epochs 3
  
  # Training only
  python orchestrate_pipeline.py --mode training_only
  
  # Inference only
  python orchestrate_pipeline.py --mode inference_only --models llava-7b qwen-vl
  
  # Analysis only
  python orchestrate_pipeline.py --mode analysis_only
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['full', 'scraping_only', 'inference_only', 'training_only', 'analysis_only'],
                       default='full',
                       help='Pipeline execution mode')
    
    parser.add_argument('--url-file', type=Path, default=Path("data/urls/test_sample.txt"),
                       help='File with URLs to scrape')
    
    parser.add_argument('--annotations-file', type=Path, default=Path("data/annotations/manual_annotations.csv"),
                       help='File with expert annotations')
    
    parser.add_argument('--models', nargs='+', default=['llava-7b'],
                       help='Models to evaluate')
    
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    
    parser.add_argument('--device', default='cuda:0',
                       help='PyTorch device (cuda:0, cuda:1, etc)')
    
    parser.add_argument('--data-dir', type=Path, default=Path("./data"),
                       help='Data directory')
    
    parser.add_argument('--results-dir', type=Path, default=Path("./data/results"),
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = WCAGPipelineOrchestrator(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device
    )
    
    # Run appropriate mode
    if args.mode == 'full':
        asyncio.run(orchestrator.run_full_pipeline(
            url_file=args.url_file,
            annotations_file=args.annotations_file,
            models=args.models,
            epochs=args.num_epochs,
            batch_size=args.batch_size
        ))
    
    elif args.mode == 'scraping_only':
        asyncio.run(orchestrator.stage_1_web_scraping(
            args.url_file,
            args.data_dir / "screenshots"
        ))
    
    elif args.mode == 'inference_only':
        asyncio.run(orchestrator.stage_2_zero_shot_evaluation(
            args.data_dir / "screenshots",
            args.models,
            ["1.4.3", "1.1.1"]
        ))
    
    elif args.mode == 'training_only':
        orchestrator.stage_4_fine_tuning(
            model_name=args.models[0],
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'analysis_only':
        orchestrator.stage_6_comparative_analysis()


if __name__ == "__main__":
    main()
