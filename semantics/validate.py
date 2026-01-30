"""
SHACL Validation Script for Web Accessibility Audit Artifacts
==============================================================

This script validates RDF audit instances against the SHACL constraint
profile (C1-C12) defined in shapes.ttl.

Requirements:
    pip install pyshacl rdflib

Usage:
    python validate.py examples/run1.ttl
    python validate.py examples/invalid-run.ttl --verbose
    python validate.py examples/*.ttl --summary
"""

import sys
import argparse
from pathlib import Path
from pyshacl import validate
from rdflib import Graph

def load_shapes():
    """Load SHACL shapes and ontology."""
    shapes_graph = Graph()
    shapes_path = Path(__file__).parent / "ontology" / "shapes.ttl"
    vocab_path = Path(__file__).parent / "ontology" / "vocab.ttl"
    
    shapes_graph.parse(shapes_path, format="turtle")
    shapes_graph.parse(vocab_path, format="turtle")
    
    return shapes_graph

def validate_file(data_file, shapes_graph, verbose=False):
    """Validate a single RDF file against SHACL shapes."""
    data_graph = Graph()
    
    try:
        data_graph.parse(data_file, format="turtle")
    except Exception as e:
        print(f"[ERROR] Error parsing {data_file}: {e}")
        return False, str(e)
    
    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference='rdfs',
        abort_on_first=False,
        meta_shacl=False,
        debug=verbose
    )
    
    return conforms, results_text

def count_violations(results_text):
    """Count constraint violations in results."""
    lines = results_text.split('\n')
    violation_count = 0
    
    for line in lines:
        if 'violation' in line.lower():
            violation_count += 1
    
    return violation_count

def main():
    parser = argparse.ArgumentParser(
        description="Validate accessibility audit RDF against SHACL constraints"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="RDF files to validate (Turtle format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation output"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only"
    )
    
    args = parser.parse_args()
    
    # Load SHACL shapes
    print("Loading SHACL shapes...")
    shapes_graph = load_shapes()
    print(f"[OK] Loaded {len(shapes_graph)} triples from shapes.ttl\n")
    
    # Validate each file
    results = []
    for file_path in args.files:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"Validating {file_path.name}...")
        conforms, results_text = validate_file(file_path, shapes_graph, args.verbose)
        
        if conforms:
            print(f"✅ VALID - {file_path.name} conforms to all constraints\n")
            results.append((file_path.name, True, 0))
        else:
            violation_count = count_violations(results_text)
            print(f"[FAIL] INVALID - {file_path.name} has {violation_count} constraint violations")
            
            if args.verbose:
                print("\nValidation Report:")
                print(results_text)
            
            if not args.summary:
                # Print first few violations
                lines = results_text.split('\n')
                violation_lines = [l for l in lines if 'violation' in l.lower()][:5]
                for vline in violation_lines:
                    print(f"  • {vline.strip()}")
                
                if len(violation_lines) > 5:
                    print(f"  ... and {len(violation_lines) - 5} more violations")
            
            print()
            results.append((file_path.name, False, violation_count))
    
    # Summary
    if args.summary or len(args.files) > 1:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        valid_count = sum(1 for _, valid, _ in results if valid)
        invalid_count = len(results) - valid_count
        total_violations = sum(count for _, valid, count in results if not valid)
        
        print(f"Total files validated: {len(results)}")
        print(f"Valid: {valid_count}")
        print(f"Invalid: {invalid_count}")
        print(f"Total constraint violations: {total_violations}")
        print()
        
        # Per-file summary
        for filename, valid, count in results:
            status = "[OK] VALID" if valid else f"[FAIL] INVALID ({count} violations)"
            print(f"  {filename:30s} {status}")
    
    # Exit code
    all_valid = all(valid for _, valid, _ in results)
    sys.exit(0 if all_valid else 1)

if __name__ == "__main__":
    main()
