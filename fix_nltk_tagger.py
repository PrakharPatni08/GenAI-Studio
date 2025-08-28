import shutil
import nltk
from pathlib import Path

try:
    tagger_path = Path(nltk.data.find("taggers/averaged_perceptron_tagger").path)
    tagger_eng_path = tagger_path.parent / "averaged_perceptron_tagger_eng"

    if not tagger_eng_path.exists():
        shutil.copytree(tagger_path, tagger_eng_path)
        print(f"✅ Created alias: {tagger_eng_path}")
    else:
        print("ℹ️ averaged_perceptron_tagger_eng already exists.")
except LookupError:
    print("❌ averaged_perceptron_tagger not found. Run: nltk.download('averaged_perceptron_tagger')")
