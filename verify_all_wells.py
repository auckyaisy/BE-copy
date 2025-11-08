#!/usr/bin/env python3
"""
Comprehensive verification script untuk memastikan SEMUA sumur 
memiliki hasil yang 100% exact match dengan reference
"""

import pandas as pd
from pathlib import Path
import sys

def compare_well(well_name, ref_base, output_base):
    """Compare a single well's output with reference"""
    
    results = {
        'well': well_name,
        '3hour_exists': False,
        '3hour_match': False,
        '3hour_rows_ref': 0,
        '3hour_rows_new': 0,
        '3hour_status_match': 0,
        '3hour_status_mismatch': 0,
        '30min_exists': False,
        '30min_match': False,
        '30min_rows_ref': 0,
        '30min_rows_new': 0,
        'status': 'UNKNOWN',
        'errors': []
    }
    
    ref_dir = ref_base / well_name
    output_dir = output_base / well_name
    
    if not output_dir.exists():
        results['status'] = 'SKIP'
        results['errors'].append('Output directory not found')
        return results
    
    # Check 3-hour results
    ref_3h_file = ref_dir / "result_df_3 jam.csv"
    new_3h_file = output_dir / "result_df_3 jam.csv"
    
    if ref_3h_file.exists() and new_3h_file.exists():
        try:
            results['3hour_exists'] = True
            ref_3h = pd.read_csv(ref_3h_file)
            new_3h = pd.read_csv(new_3h_file)
            
            results['3hour_rows_ref'] = len(ref_3h)
            results['3hour_rows_new'] = len(new_3h)
            
            # Convert timestamps
            ref_3h['Window_Start_Time'] = pd.to_datetime(ref_3h['Window_Start_Time'])
            new_3h['Window_Start_Time'] = pd.to_datetime(new_3h['Window_Start_Time'])
            
            # Merge and compare
            merged = ref_3h.merge(
                new_3h, 
                on='Window_Start_Time', 
                how='outer',
                suffixes=('_ref', '_new'),
                indicator=True
            )
            
            # Count matches
            both = merged[merged['_merge'] == 'both']
            if len(both) > 0:
                matches = sum(both['Dominant Status_ref'] == both['Dominant Status_new'])
                results['3hour_status_match'] = matches
                results['3hour_status_mismatch'] = len(both) - matches
                
                if len(ref_3h) == len(new_3h) and results['3hour_status_mismatch'] == 0:
                    results['3hour_match'] = True
            
        except Exception as e:
            results['errors'].append(f"3hour error: {str(e)}")
    
    # Check 30-min results
    ref_30_file = ref_dir / "prediction_results_30menit.csv"
    new_30_file = output_dir / f"{well_name}_failure_prediction_30min.csv"
    
    if ref_30_file.exists() and new_30_file.exists():
        try:
            results['30min_exists'] = True
            ref_30 = pd.read_csv(ref_30_file)
            new_30 = pd.read_csv(new_30_file)
            
            results['30min_rows_ref'] = len(ref_30)
            results['30min_rows_new'] = len(new_30)
            
            # Convert timestamps
            ref_30['Window_Start_Time'] = pd.to_datetime(ref_30['Window_Start_Time'])
            new_30['Window_Start_Time'] = pd.to_datetime(new_30['Window_Start_Time'])
            
            # Compare Status columns
            merged = ref_30.merge(
                new_30,
                on='Window_Start_Time',
                how='outer',
                suffixes=('_ref', '_new'),
                indicator=True
            )
            
            both = merged[merged['_merge'] == 'both']
            if len(both) > 0:
                # Check if Status columns match
                status_matches = sum(both['Status_ref'] == both['Status_new'])
                if len(ref_30) == len(new_30) and status_matches == len(both):
                    results['30min_match'] = True
            
        except Exception as e:
            results['errors'].append(f"30min error: {str(e)}")
    
    # Determine overall status
    if results['3hour_exists'] and results['30min_exists']:
        if results['3hour_match'] and results['30min_match']:
            results['status'] = '✅ PERFECT'
        elif results['3hour_match'] or results['30min_match']:
            results['status'] = '⚠️ PARTIAL'
        else:
            results['status'] = '❌ MISMATCH'
    elif results['3hour_exists'] or results['30min_exists']:
        if results['3hour_match'] or results['30min_match']:
            results['status'] = '⚠️ PARTIAL'
        else:
            results['status'] = '❌ MISMATCH'
    else:
        results['status'] = '❌ NO_DATA'
    
    return results


def main():
    base_dir = Path(__file__).parent
    ref_base = base_dir / "Test Web" / "Hasil Bacaan Notebook"
    output_base = base_dir / "test_output"
    
    # Get all wells
    wells = [d.name for d in ref_base.iterdir() if d.is_dir() and d.name.startswith("SKW-")]
    wells = sorted(wells)
    
    print("\n" + "="*100)
    print("COMPREHENSIVE VERIFICATION: ALL WELLS")
    print("="*100)
    
    all_results = []
    
    for well in wells:
        print(f"\n{'='*100}")
        print(f"Verifying: {well}")
        print(f"{'='*100}")
        
        result = compare_well(well, ref_base, output_base)
        all_results.append(result)
        
        # Print details
        print(f"Status: {result['status']}")
        
        if result['3hour_exists']:
            match_str = "✅" if result['3hour_match'] else "❌"
            print(f"  3-Hour: {match_str}")
            print(f"    Reference rows: {result['3hour_rows_ref']}")
            print(f"    New output rows: {result['3hour_rows_new']}")
            if result['3hour_status_match'] > 0 or result['3hour_status_mismatch'] > 0:
                total = result['3hour_status_match'] + result['3hour_status_mismatch']
                pct = 100 * result['3hour_status_match'] / total if total > 0 else 0
                print(f"    Status matches: {result['3hour_status_match']}/{total} ({pct:.1f}%)")
                if result['3hour_status_mismatch'] > 0:
                    print(f"    ⚠️ Mismatches: {result['3hour_status_mismatch']}")
        
        if result['30min_exists']:
            match_str = "✅" if result['30min_match'] else "❌"
            print(f"  30-Min: {match_str}")
            print(f"    Reference rows: {result['30min_rows_ref']}")
            print(f"    New output rows: {result['30min_rows_new']}")
        
        if result['errors']:
            print(f"  Errors:")
            for err in result['errors']:
                print(f"    - {err}")
    
    # Summary
    print("\n\n" + "="*100)
    print("SUMMARY: ALL WELLS VERIFICATION")
    print("="*100)
    
    perfect = sum(1 for r in all_results if r['status'] == '✅ PERFECT')
    partial = sum(1 for r in all_results if r['status'] == '⚠️ PARTIAL')
    mismatch = sum(1 for r in all_results if '❌' in r['status'])
    
    print(f"\nTotal Wells: {len(all_results)}")
    print(f"✅ Perfect Match: {perfect}")
    print(f"⚠️ Partial Match: {partial}")
    print(f"❌ Mismatch/Issues: {mismatch}")
    
    print(f"\n{'Well':<12} {'Status':<15} {'3-Hour':<12} {'30-Min':<12} {'Notes'}")
    print("-"*100)
    
    for r in all_results:
        well = r['well']
        status = r['status']
        
        # 3-hour status
        if r['3hour_exists']:
            if r['3hour_match']:
                hour3 = f"✅ {r['3hour_rows_ref']}"
            else:
                hour3 = f"❌ {r['3hour_status_match']}/{r['3hour_rows_ref']}"
        else:
            hour3 = "N/A"
        
        # 30-min status
        if r['30min_exists']:
            if r['30min_match']:
                min30 = f"✅ {r['30min_rows_ref']}"
            else:
                min30 = f"❌ {r['30min_rows_ref']}"
        else:
            min30 = "N/A"
        
        notes = ", ".join(r['errors']) if r['errors'] else ""
        
        print(f"{well:<12} {status:<15} {hour3:<12} {min30:<12} {notes}")
    
    # Final verdict
    print("\n" + "="*100)
    if perfect == len(all_results):
        print("🎉 ALL WELLS: 100% PERFECT MATCH!")
    elif perfect + partial == len(all_results):
        print(f"⚠️ {perfect}/{len(all_results)} wells perfect, {partial} partial matches")
    else:
        print(f"⚠️ ATTENTION: {mismatch} wells have issues that need review")
    print("="*100 + "\n")
    
    # Save detailed report
    df_report = pd.DataFrame(all_results)
    report_file = output_base / "verification_report_all_wells.csv"
    df_report.to_csv(report_file, index=False)
    print(f"📄 Detailed report saved to: {report_file}\n")
    
    return 0 if perfect == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
