#!/usr/bin/env python3
"""
Hugging Face Space Deployment Optimizer
Pre-deploy sanity checks and optimization for HF Spaces deployment.
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import shutil
from typing import List, Dict, Any


class DeploymentOptimizer:
    """Optimizes the codebase for Hugging Face Spaces deployment"""
    
    def __init__(self, root_path: Path = None):
        self.root_path = root_path or Path.cwd()
        self.issues = []
        self.warnings = []
        
    def log_issue(self, message: str, severity: str = "ERROR"):
        """Log deployment issue"""
        if severity == "ERROR":
            self.issues.append(message)
        else:
            self.warnings.append(message)
        print(f"[{severity}] {message}")
    
    def check_required_files(self) -> bool:
        """Check for required files in deployment"""
        required_files = [
            "app.py",
            "requirements.txt",
            "utils/errors.py",
            "utils/results_manager.py",
            "utils/confidence.py", 
            "utils/multifile.py",
            "utils/preprocessing.py"
        ]
        
        all_present = True
        for file_path in required_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.log_issue(f"Required file missing: {file_path}")
                all_present = False
            else:
                print(f"✅ Found: {file_path}")
        
        return all_present
    
    def check_model_weights(self) -> bool:
        """Check for model weight files"""
        model_dirs = [
            self.root_path / "outputs",
            self.root_path / "model_weights"
        ]
        
        expected_models = [
            "figure2_model.pth",
            "resnet_model.pth"
        ]
        
        found_models = []
        for model_dir in model_dirs:
            if model_dir.exists():
                for model_file in expected_models:
                    model_path = model_dir / model_file
                    if model_path.exists():
                        found_models.append(str(model_path))
                        print(f"✅ Found model: {model_path}")
        
        if not found_models:
            self.log_issue("No model weight files found. App will use random initialization.", "WARNING")
            return False
        
        return True
    
    def check_sample_data(self) -> bool:
        """Check for sample data files"""
        sample_dir = self.root_path / "sample_data"
        if not sample_dir.exists():
            self.log_issue("Sample data directory missing", "WARNING")
            return False
        
        txt_files = list(sample_dir.glob("*.txt"))
        if not txt_files:
            self.log_issue("No sample .txt files found in sample_data/", "WARNING")
            return False
        
        print(f"✅ Found {len(txt_files)} sample data files")
        return True
    
    def validate_requirements(self) -> bool:
        """Validate and optimize requirements.txt"""
        req_file = self.root_path / "requirements.txt"
        if not req_file.exists():
            self.log_issue("requirements.txt missing")
            return False
        
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Essential packages for the app
        essential_packages = {
            "streamlit", "torch", "torchvision", "numpy", "pandas", 
            "scikit-learn", "scipy", "matplotlib", "altair"
        }
        
        found_packages = set()
        for req in requirements:
            package = req.split('>=')[0].split('==')[0].split('<')[0]
            found_packages.add(package)
        
        missing_essential = essential_packages - found_packages
        if missing_essential:
            for pkg in missing_essential:
                self.log_issue(f"Essential package missing from requirements.txt: {pkg}")
            return False
        
        # Check for potentially problematic packages
        problematic = {
            "tensorflow", "jupyter", "notebook", "jupyter-lab", 
            "opencv-python", "cv2"  # Heavy packages that might not be needed
        }
        
        found_problematic = found_packages.intersection(problematic)
        if found_problematic:
            for pkg in found_problematic:
                self.log_issue(f"Potentially heavy package found: {pkg}. Consider if necessary.", "WARNING")
        
        print(f"✅ Requirements validation passed ({len(requirements)} packages)")
        return True
    
    def optimize_requirements(self) -> bool:
        """Create optimized requirements.txt for HF Spaces"""
        optimized_requirements = [
            "streamlit>=1.49.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "altair>=4.2.0",
        ]
        
        output_file = self.root_path / "requirements_optimized.txt"
        with open(output_file, 'w') as f:
            f.write("# Optimized requirements for HF Spaces deployment\n")
            f.write("# Generated by deployment_optimizer.py\n\n")
            for req in optimized_requirements:
                f.write(f"{req}\n")
        
        print(f"✅ Created optimized requirements: {output_file}")
        return True
    
    def check_app_structure(self) -> bool:
        """Check app.py structure for HF Spaces compatibility"""
        app_file = self.root_path / "app.py"
        if not app_file.exists():
            self.log_issue("app.py not found")
            return False
        
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            "import streamlit as st",
            "from utils.multifile import",
            "from utils.results_manager import", 
            "from utils.confidence import"
        ]
        
        for import_line in required_imports:
            if import_line not in content:
                self.log_issue(f"Missing import in app.py: {import_line}")
                return False
        
        # Check for problematic patterns
        if "streamlit run" in content:
            self.log_issue("Found 'streamlit run' in app.py - remove for HF Spaces", "WARNING")
        
        if "__name__ == '__main__'" in content and "main()" in content:
            print("✅ App structure looks good for HF Spaces")
        else:
            self.log_issue("App should call main() function at the end", "WARNING")
        
        return True
    
    def create_hf_space_config(self) -> bool:
        """Create or update HF Spaces configuration"""
        # Create app.py header comment if needed
        app_file = self.root_path / "app.py"
        with open(app_file, 'r') as f:
            content = f.read()
        
        hf_header = '''"""
AI-Driven Polymer Aging Prediction and Classification
Hugging Face Spaces Deployment

This Streamlit app provides an interface for classifying polymer degradation
using deep learning models trained on Raman spectroscopy data.

Features:
- Single file and batch upload processing
- Multiple CNN model architectures
- Results export to CSV/JSON
- Enhanced confidence visualization
- Session-wide results management
"""

'''
        
        if "Hugging Face Spaces Deployment" not in content:
            # Backup original
            shutil.copy2(app_file, self.root_path / "app.py.backup")
            
            # Add header
            with open(app_file, 'w') as f:
                f.write(hf_header + content)
            
            print("✅ Added HF Spaces header to app.py")
        
        # Create .gitignore for HF Spaces
        gitignore_content = """
# HF Spaces specific
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.pytest_cache/
.coverage
*.cover
.hypothesis/

# Model weights (too large for git)
*.pth
*.pt
*.pkl

# Temporary files
*.tmp
*.bak
*.backup

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
        
        gitignore_file = self.root_path / ".gitignore"
        if not gitignore_file.exists():
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            print("✅ Created .gitignore for HF Spaces")
        
        return True
    
    def run_sanity_checks(self) -> bool:
        """Run all sanity checks"""
        print("🚀 Running HF Spaces deployment sanity checks...\n")
        
        checks = [
            ("Required Files", self.check_required_files),
            ("App Structure", self.check_app_structure),
            ("Requirements", self.validate_requirements),
            ("Model Weights", self.check_model_weights),
            ("Sample Data", self.check_sample_data),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.log_issue(f"Check failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def optimize_for_deployment(self) -> bool:
        """Run optimization steps"""
        print("\n🔧 Running deployment optimizations...\n")
        
        optimizations = [
            ("Requirements Optimization", self.optimize_requirements),
            ("HF Spaces Configuration", self.create_hf_space_config),
        ]
        
        all_passed = True
        for opt_name, opt_func in optimizations:
            print(f"\n⚙️ {opt_name}:")
            try:
                result = opt_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.log_issue(f"Optimization failed: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deployment readiness report"""
        report = {
            "timestamp": str(subprocess.check_output(['date'], text=True).strip()),
            "issues": self.issues,
            "warnings": self.warnings,
            "ready_for_deployment": len(self.issues) == 0
        }
        
        report_file = self.root_path / "deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Deployment report saved to: {report_file}")
        return report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HF Spaces Deployment Optimizer")
    parser.add_argument("--root", type=Path, default=Path.cwd(), 
                       help="Root directory of the project")
    parser.add_argument("--optimize", action="store_true",
                       help="Run optimization steps")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate report, don't run checks")
    
    args = parser.parse_args()
    
    optimizer = DeploymentOptimizer(args.root)
    
    if args.report_only:
        report = optimizer.generate_report()
        return 0 if report["ready_for_deployment"] else 1
    
    # Run sanity checks
    checks_passed = optimizer.run_sanity_checks()
    
    # Run optimizations if requested
    if args.optimize:
        opt_passed = optimizer.optimize_for_deployment()
        checks_passed = checks_passed and opt_passed
    
    # Generate report
    report = optimizer.generate_report()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("="*60)
    
    if report["ready_for_deployment"]:
        print("✅ READY FOR DEPLOYMENT")
        print("All checks passed!")
    else:
        print("❌ NOT READY FOR DEPLOYMENT")
        print(f"Issues found: {len(optimizer.issues)}")
        for issue in optimizer.issues:
            print(f"  • {issue}")
    
    if optimizer.warnings:
        print(f"\n⚠️ Warnings: {len(optimizer.warnings)}")
        for warning in optimizer.warnings:
            print(f"  • {warning}")
    
    return 0 if report["ready_for_deployment"] else 1


if __name__ == "__main__":
    sys.exit(main())