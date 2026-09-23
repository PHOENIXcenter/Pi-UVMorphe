#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepUV Mass Spectrometry Data Analysis Main Program
Used for prediction and fragment matching analysis
"""

import os
import sys
import pickle
import argparse
import torch
import pandas as pd
from pathlib import Path
import csv
# Add project root directory to path (if running directly)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DeepUVModel
from pipeline import process_single_file, load_model, process_single_deepuv_file
from utils import load_standard_scaler
from config import MODEL_CONFIG, ensure_directories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pi-Morphe Mass Spectrometry Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Basic usage
  python run.py -m msalign_file.msalign -j spectrum0.js -s "PHSHPALTPEQ..."

  # Specify modifications
  python run.py -m file.msalign -j spectrum0.js -s "SEQ..." -mod acetylated

  # Save results to a specified directory (includes visual spectrum maps)
  python run.py -m file.msalign -j spectrum0.js -s "SEQ..." -o ./results --save_csv
        """
    )

    parser.add_argument(
        '-m', '--msalign',
        type=str,
        required=True,
        help='Path to MSALIGN file (.msalign)'
    )

    parser.add_argument(
        '-j', '--js',
        type=str,
        required=True,
        help='Path to spectrum0.js file'
    )

    parser.add_argument(
        '-s', '--sequence',
        type=str,
        required=True,
        help='Protein sequence'
    )

    parser.add_argument(
        '-mod', '--modification',
        type=str,
        default=None,
        choices=['n_acetylated', 'n_deaminated', 'n_methylated', 'n_dimethylated',
                 'n_formylated', 'c_amided', 'c_methylated', 'c_dehydrated'],
        help='Modification type (Default: None)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='Output directory (Default: ./output)'
    )

    parser.add_argument(
        '--scaler',
        type=str,
        default='scaler0424.pkl',
        help='Path to StandardScaler file (Default: scaler0424.pkl)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='best_model.pth',
        help='Path to model weights file (Default: best_model.pth)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Computing device (Default: Auto-detect)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (Default: 64)'
    )

    parser.add_argument(
        '--ppm',
        type=int,
        default=5,
        help='PPM tolerance (Default: 5)'
    )

    parser.add_argument(
        '--no_save_images',
        action='store_true',
        help='Do not save intermediate image files'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save results as CSV files'
    )

    parser.add_argument(
        '--no_plot_maps',
        action='store_true',
        help='Do not generate fragmentation maps'
    )

    parser.add_argument(
        '--no_plot_abundance',
        action='store_true',
        help='Do not generate abundance bar plots'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed processing logs'
    )
    parser.add_argument('--custom_mod_name', type=str, default=None)
    parser.add_argument('--custom_mod_mass', type=float, default=None)
    parser.add_argument('--custom_mod_formula', type=str, default=None)
    parser.add_argument('--custom_mod_position', type=str, default='N-term', choices=['N-term', 'C-term'])
    return parser.parse_args()


def main_with_args(args):
    """
    Run the main program using an argument object (for interactive calls).

    Parameters:
    -----------
    args : argparse.Namespace
        Namespace object containing all arguments.

    Returns:
    --------
    tuple
        (result_df, matches_df, reliability)
    """
    # Set device
    if hasattr(args, 'device') and args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory
    temp_dir = output_dir / 'temp_envelopes'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Check if files exist
    if not os.path.exists(args.msalign):
        raise FileNotFoundError(f"MSALIGN file not found: {args.msalign}")

    if not os.path.exists(args.js):
        raise FileNotFoundError(f"spectrum0.js file not found: {args.js}")

    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Scaler file not found: {args.scaler}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Generate filename suffix
    file_suffix = os.path.splitext(os.path.basename(args.msalign))[0]

    print("=" * 60)
    print("Pi-Morphe Mass Spectrometry Data Analysis")
    print("=" * 60)
    print(f"MSALIGN file: {args.msalign}")
    print(f"spectrum0.js: {args.js}")
    print(f"Output directory: {output_dir}")
    print(f"Modification type: {getattr(args, 'modification', None) or 'None'}")
    print(f"PPM tolerance: {getattr(args, 'ppm', 5)}")
    print(f"Generate fragmentation maps: {'No' if getattr(args, 'no_plot_maps', False) else 'Yes'}")
    print(f"Generate abundance bar plots: {'No' if getattr(args, 'no_plot_abundance', False) else 'Yes'}")
    print("=" * 60)

    try:
        # -------------------------------------------------
        # Step 1: Load scaler and model
        # -------------------------------------------------
        print("\n[1/3] Loading Model and Scaler...")
        scaler = load_standard_scaler(args.scaler)

        model = DeepUVModel(**MODEL_CONFIG).to(device)
        model = load_model(model, args.model, device)
        model.eval()
        print("✓ Model successfully loaded")

        # -------------------------------------------------
        # Step 2: Prediction
        # -------------------------------------------------
        print("\n[2/3] Running Pi-Morphe prediction...")
        result_df = process_single_file(
            msalign_file=args.msalign,
            spectrum_js_file=args.js,
            standard_scaler=scaler,
            model=model,
            device=device,
            output_envelope_dir=str(temp_dir),
            save_images=True,
            return_df=True
        )
        print(f"✓ Prediction finished. Total records: {len(result_df)}")

        # Save prediction results
        # if getattr(args, 'save_csv', False):
        #     pred_path = output_dir / 'predictions.csv'
        #     result_df.to_csv(pred_path, index=False)
        #     print(f"✓ Predictions saved: {pred_path}")

        # -------------------------------------------------
        # Step 3: Fragment Matching
        # -------------------------------------------------
        print("\n[3/3] Analyzing fragment matching...")

        # Prepare visualization output directory
        plot_dir = output_dir / 'plots'

        matches_df, reliability = process_single_deepuv_file(
            seq=args.sequence,
            final_df=result_df,
            msalign_path=args.msalign,
            ppm=getattr(args, 'ppm', 5),
            modification=getattr(args, 'modification', None),
            custom_mod_name=getattr(args, 'custom_mod_name', None),
            custom_mod_mass=getattr(args, 'custom_mod_mass', None),
            custom_mod_formula=getattr(args, 'custom_mod_formula', None),
            custom_mod_position=getattr(args, 'custom_mod_position', 'N-term'),
            output_dir=str(plot_dir),
            plot_maps=not getattr(args, 'no_plot_maps', False),
            plot_abundance=not getattr(args, 'no_plot_abundance', False),
            filename_suffix=file_suffix
        )

        # Save matching results
        if matches_df is not None and getattr(args, 'save_csv', False):
            matches_path = output_dir / 'matches.csv'
            matches_df.to_csv(matches_path, index=False, quoting=csv.QUOTE_ALL)
            print(f"✓ Matching results saved: {matches_path}")

        # Save reliability score
        if reliability is not None and getattr(args, 'save_csv', False):
            rel_path = output_dir / 'reliability.json'
            import json
            with open(rel_path, 'w') as f:
                json.dump(reliability, f, indent=2, default=str)
            print(f"✓ Reliability score saved: {rel_path}")

        # -------------------------------------------------
        # Summary of results
        # -------------------------------------------------
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Predicted records: {len(result_df)}")
        print(f"  Matched records: {len(matches_df) if matches_df is not None else 0}")

        if reliability:
            print(f"  IRS Score: {reliability.get('IRS', 0):.4f}")
            print(f"  Terminal cleavage coverage: {reliability.get('terminal_cleavage_coverage', 0):.2%}")
            print(f"  Internal cleavage coverage: {reliability.get('internal_cleavage_coverage', 0):.2%}")
            print(f"  Total coverage: {reliability.get('total_cleavage_coverage', 0):.2%}")

        if not getattr(args, 'no_plot_maps', False) or not getattr(args, 'no_plot_abundance', False):
            print(f"\n  Maps saved in: {plot_dir}")
            if not getattr(args, 'no_plot_maps', False):
                print(f"    - terminal_fragment_cleavage_map.html")
                #print(f"    - internal_fragment_cleavage_map.html")
                print(f"    - internal_fragment_support_map.html")
            if not getattr(args, 'no_plot_abundance', False):
                print(f"    - bar_plot_terminal_{file_suffix}.html")
                print(f"    - bar_plot_internal_{file_suffix}.html")

        print("=" * 60)
        print("✓ Analysis complete!")

        return matches_df, reliability

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main execution point for command line usage."""
    args = parse_args()
    return main_with_args(args)


def interactive_mode():
    """
    Interactive execution mode (designed for Jupyter Notebooks or interactive Python environments).

    Example usage:
        from run import interactive_mode
        result_df, matches_df, reliability = interactive_mode()
    """
    print("=" * 60)
    print("Pi-Morphe Interactive Mode")
    print("=" * 60)

    msalign_path = input("MSALIGN file path: ").strip()
    js_path = input("spectrum0.js file path: ").strip()
    sequence = input("Protein sequence: ").strip()
    modification = input(
        "Modification type (Leave blank to skip, optional: n_acetylated/n_deaminated/n_methylated/n_dimethylated/n_formylated/c_amided/c_methylated/c_dehydrated): ").strip() or None

    if modification and modification not in ['n_acetylated', 'n_deaminated', 'n_methylated', 'n_dimethylated',
                                             'n_formylated', 'c_amided', 'c_methylated', 'c_dehydrated']:
        print(f"⚠ Unknown modification type: {modification}, falling back to None")
        modification = None

    output_dir = input("Output directory (Default: ./output): ").strip() or './output'
    save_csv = input("Save CSV? (y/n, Default: y): ").strip().lower() != 'n'
    plot_maps = input("Generate fragmentation maps? (y/n, Default: y): ").strip().lower() != 'n'
    plot_abundance = input("Generate abundance bar plots? (y/n, Default: y): ").strip().lower() != 'n'

    # Build argument namespace object
    args = argparse.Namespace(
        msalign=msalign_path,
        js=js_path,
        sequence=sequence,
        modification=modification,
        output=output_dir,
        scaler='scaler0424.pkl',
        model='best_model.pth',
        device=None,
        batch_size=64,
        ppm=5,
        no_save_images=False,
        save_csv=save_csv,
        no_plot_maps=not plot_maps,
        no_plot_abundance=not plot_abundance,
        verbose=True
    )

    return main_with_args(args)


def batch_process(
        input_dir: str,
        output_dir: str,
        protein_seq_map: dict,
        scaler_path: str = 'scaler0424.pkl',
        model_path: str = 'best_model.pth',
        modification: str = None,
        ppm: int = 5,
        save_csv: bool = True,
        plot_maps: bool = True,
        plot_abundance: bool = True
):
    """
    Batch process multiple CSV files (for cases where predictions already exist,
    running only fragment matching and visualization).

    Parameters:
    -----------
    input_dir : str
        Directory containing input CSV files.
    output_dir : str
        Output directory.
    protein_seq_map : dict
        A dictionary mapping filename keywords to protein sequences.
        Example: {"Myo": "GLSDGEW...", "aldolase": "PHSHPAL..."}
    scaler_path : str
        Path to Scaler file.
    model_path : str
        Path to Model file.
    modification : str
        Modification type.
    ppm : int
        PPM tolerance.
    save_csv : bool
        Whether to save CSV results.
    plot_maps : bool
        Whether to draw fragmentation maps.
    plot_abundance : bool
        Whether to draw abundance bar plots.
    """
    import glob

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir_path / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_results = []

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"\n{'=' * 50}")
        print(f"Processing: {file_name}")
        print(f"{'=' * 50}")

        try:
            # Map protein sequence based on filename keywords
            seq = None
            matched_key = None
            for key, seq_val in protein_seq_map.items():
                if key in file_name:
                    seq = seq_val
                    matched_key = key
                    break

            if seq is None:
                print(f"  ⚠ No matching protein sequence found. Skipping file: {file_name}")
                continue

            print(f"  ✓ Matched protein keyword: {matched_key}")

            # Read CSV
            df = pd.read_csv(csv_file)
            print(f"  ✓ Successfully read data: {len(df)} rows")

            # Check for 'pred_label' column, create it with default value if missing
            if 'pred_label' not in df.columns:
                print(f"  ⚠ Column 'pred_label' not found. Using default value (all 1s)")
                df['pred_label'] = 1

            # Generate filename suffix
            file_suffix = os.path.splitext(file_name)[0]

            # Perform fragment matching analysis
            print("  Analyzing fragment matching...")
            matches_df, reliability = process_single_deepuv_file(
                seq=seq,
                final_df=df,
                msalign_path=csv_file,  # Using CSV path here as identifier reference
                ppm=ppm,
                modification=modification,
                output_dir=str(plot_dir),
                plot_maps=plot_maps,
                plot_abundance=plot_abundance,
                filename_suffix=file_suffix
            )

            # Save matching results
            if save_csv and matches_df is not None:
                matches_path = output_dir_path / f'matches_{file_suffix}.csv'
                matches_df.to_csv(matches_path, index=False, quoting=csv.QUOTE_ALL)
                print(f"  ✓ Fragment matches saved: {matches_path}")

            if save_csv and reliability is not None:
                import json
                rel_path = output_dir_path / f'reliability_{file_suffix}.json'
                with open(rel_path, 'w') as f:
                    json.dump(reliability, f, indent=2, default=str)
                print(f"  ✓ Reliability score saved: {rel_path}")

            all_results.append({
                'file': file_name,
                'protein': matched_key,
                'matches_count': len(matches_df) if matches_df is not None else 0,
                'IRS': reliability.get('IRS', 0) if reliability else 0
            })

            print(f"  ✅ Finished processing successfully")

        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

    # Print processing summary
    print("\n" + "=" * 60)
    print("Batch Processing Complete!")
    print("=" * 60)
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
    else:
        print("No files were successfully processed.")

    return all_results


if __name__ == "__main__":
    main()