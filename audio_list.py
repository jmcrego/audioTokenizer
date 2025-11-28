import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import soundfile as sf

def get_audio_duration(filepath):
    """Get duration of an audio file in seconds."""
    try:
        info = sf.info(filepath)
        return filepath, info.duration
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return filepath, None

def find_audio_files_by_lang(base_path, langs, max_workers=8):
    """
    Find audio files by language and compute their durations efficiently.
    
    Args:
        base_path (str): Base path with LANG placeholder
        langs (str or list): Comma-separated string or list of language codes
        max_workers (int): Number of parallel workers
        use_full_path (bool): If True, use full path as key; if False, use filename only
    
    Returns:
        dict: Nested dictionary {lang: {path/filename: duration}}
    """
    if isinstance(langs, str):
        langs = [lang.strip() for lang in langs.split(',')]
    
    results = {}
    
    for lang in langs:
        print(f"\nProcessing language: {lang}")
        lang_path = Path(base_path.replace('LANG', lang))
        
        if not lang_path.exists():
            print(f"Warning: Path does not exist for language {lang}: {lang_path}")
            results[lang] = {}
            continue
        
        # Find all .mp3 files
        mp3_files = list(lang_path.rglob('*.mp3'))
        print(f"Found {len(mp3_files)} files")
        
        if not mp3_files:
            results[lang] = {}
            continue
        
        # Compute durations in parallel
        # lang_durations = {}
        # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #     futures = {executor.submit(get_audio_duration, str(f)): f for f in mp3_files}            
        #     for future in tqdm(as_completed(futures), total=len(mp3_files), desc=f"Computing durations for {lang}"):
        #         filepath, duration = future.result()
        #         if duration is not None:
        #             key = filepath if use_full_path else Path(filepath).name
        #             lang_durations[key] = duration

        lang_durations = {}
        for filepath in tqdm(mp3_files, total=len(mp3_files), desc=f"Durations for {lang}", unit=" file"):
            duration = get_audio_duration(filepath)
            if duration is not None:
                key = filepath
                lang_durations[key] = duration

        results[lang] = lang_durations
        total_duration = sum(lang_durations.values())
        print(f"Total duration for {lang}: {total_duration:.2f}s ({total_duration/3600:.2f}h)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language and compute durations")
    parser.add_argument("--base-path", type=str, required=True, help="Base path with LANG placeholder")
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()
    
    print("Starting audio file search and duration computation...")
    audio_files = find_audio_files_by_lang(
        args.base_path, 
        args.langs, 
        max_workers=args.workers,
    )
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(audio_files, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nSummary:")
    total_files = 0
    total_hours = 0
    for lang, files in audio_files.items():
        num_files = len(files)
        total_duration = sum(files.values()) if files else 0
        total_files += num_files
        total_hours += total_duration / 3600
        print(f"  {lang}: {num_files:6d} files, {total_duration/3600:8.2f}h")
    
    print(f"\n  Total: {total_files:6d} files, {total_hours:8.2f}h")