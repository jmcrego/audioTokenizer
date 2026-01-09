# coding=utf-8
import os
import argparse
import tarfile
import urllib.request
from pathlib import Path

# Supported language pairs for CoVoST 2
XX_EN_LANGUAGES = ["fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa", "et", "mn", "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta", "ja", "id", "cy"]
EN_XX_LANGUAGES = ["de", "tr", "fa", "sv-SE", "mn", "zh-CN", "cy", "ca", "sl", "et", "id", "ar", "ta", "lv", "ja"]

# Template for CoVoST 2 TSV download URL
COVOST_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/covost/covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"

def get_supported_lang_pairs():
    """Return a list of all supported language pairs."""
    pairs = []
    for lang in EN_XX_LANGUAGES:
        pairs.append(f"en_{lang}")
    for lang in XX_EN_LANGUAGES:
        pairs.append(f"{lang}_en")
    return pairs

def download_covost_tsv(src_lang, tgt_lang, output_dir):
    """
    Download and extract the CoVoST 2 TSV file for a specific language pair.

    Args:
        src_lang (str): Source language.
        tgt_lang (str): Target language.
        output_dir (str): Directory to save the TSV file.
    """
    url = COVOST_URL_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang)
    tsv_filename = f"covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    tsv_path = os.path.join(output_dir, tsv_filename)

    print(f"Downloading {url}...")

    # Add a custom user agent to mimic a browser
    req = urllib.request.Request(
        url,
        data=None,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})

    with urllib.request.urlopen(req) as response, open(tsv_path, 'wb') as out_file:
        out_file.write(response.read())

    print(f"Extracting {tsv_path}...")
    with tarfile.open(tsv_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    os.remove(tsv_path)
    print(f"TSV file saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download CoVoST 2 TSV files for a specific language pair.")
    parser.add_argument("--lang_pair", type=str, required=True, choices=get_supported_lang_pairs(), help="Language pair, e.g., 'fr_en' or 'en_es'.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the downloaded TSV files.")
    parser.add_argument("--cv_audio_path", type=str, default=None, help="Path to the Common Voice audio files (e.g., 'clips/'). Optional, for validation or further processing.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    src_lang, tgt_lang = args.lang_pair.split("_")
    download_covost_tsv(src_lang, tgt_lang, args.output_dir)

    if args.cv_audio_path:
        print(f"Common Voice audio path set to: {args.cv_audio_path}")

if __name__ == "__main__":
    main()
