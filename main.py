import os
import csv
import openai
import dotenv
import json
import google.cloud.texttospeech as tts
from tqdm import tqdm
import argparse

dotenv.load_dotenv()

TTS_CLIENT = tts.TextToSpeechClient()

def synthesize_audio(text, filename):
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(
        language_code="cmn-CN",
        name="cmn-CN-Chirp3-HD-Achernar"
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    response = TTS_CLIENT.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(filename, "wb") as out:
        out.write(response.audio_content)


def generate_fields_from_hanzi(hanzi):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a Mandarin language assistant. For the word: {hanzi}, provide the following:
1. English meaning
2. Pinyin
3. Example sentence using the word
4. English translation of the sentence
5. Sentence with the word replaced by a blank (cloze format). Make sure the cloze deletion has low ambiguity and the missing word is clearly {hanzi}, not another word.

Respond in JSON like this:
{{
  "english": "...",
  "pinyin": "...",
  "sentence": "...",
  "translation": "...",
  "cloze": "..."
}}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "json_object"}}
    )

    text = response.output_text
    try:
        data = json.loads(text)
    except Exception:
        raise ValueError("Could not parse GPT output")

    return {
        "Hanzi": hanzi,
        "English": data["english"],
        "Pinyin": data["pinyin"],
        "Sentence": data["sentence"].strip(),
        "Sentence (Translation)": data["translation"].strip(),
        "Sentence (Cloze)": data["cloze"].strip()
    }



def process_word(hanzi, output_writer):
    try:
        fields = generate_fields_from_hanzi(hanzi)

        os.makedirs('output/audio', exist_ok=True)
        audio_word_path = f"{'output/audio/'}{hanzi}_word.mp3"
        audio_sent_path = f"{'output/audio/'}{hanzi}_sentence.mp3"

        synthesize_audio(fields["Hanzi"], audio_word_path)
        synthesize_audio(fields["Sentence"], audio_sent_path)

        fields["Audio (Word)"] = f"[sound:{os.path.basename(audio_word_path)}]"
        fields["Audio (Sentence)"] = f"[sound:{os.path.basename(audio_sent_path)}]"

        output_writer.writerow(fields)
    except Exception as e:
        print(f"Error processing {hanzi}: {e}")

def save_progress(writer, rows, output_file):
    """Save the current progress to the CSV file"""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Hanzi", "English", "Pinyin", "Sentence",
            "Sentence (Translation)", "Sentence (Cloze)",
            "Audio (Word)", "Audio (Sentence)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process Hanzi words and generate Anki cards.')
    parser.add_argument('input_file', nargs='?', help='Input file containing Hanzi words (one per line)')
    args = parser.parse_args()

    output_file = "output/notes.csv"
    os.makedirs("output", exist_ok=True)

    fieldnames = [
        "Hanzi", "English", "Pinyin", "Sentence",
        "Sentence (Translation)", "Sentence (Cloze)",
        "Audio (Word)", "Audio (Sentence)"
    ]
    
    # Initialize CSV file with headers
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    processed_rows = []
    failed_words = []
    row_count = 0

    # Get words from input file or interactive mode
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' not found.")
            return
        with open(args.input_file, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        print("No input file provided. Entering interactive mode.")
        print("Enter Hanzi words one at a time. Press Enter twice to finish.")
        words = []
        while True:
            word = input("Enter a Hanzi word (or press Enter to finish): ").strip()
            if not word:
                break
            words.append(word)
        
        if not words:
            print("No words entered. Exiting.")
            return

    # Remove duplicates while preserving order
    original_count = len(words)
    seen = set()
    words = [x for x in words if not (x in seen or seen.add(x))]
    
    if original_count > len(words):
        print(f"\nRemoved {original_count - len(words)} duplicate words.")

    print(f"\nProcessing {len(words)} unique words...")
    for hanzi in tqdm(words, desc="Processing words", unit="word"):
        try:
            fields = generate_fields_from_hanzi(hanzi)

            os.makedirs('output/audio', exist_ok=True)
            audio_word_path = f"{'output/audio/'}{hanzi}_word.mp3"
            audio_sent_path = f"{'output/audio/'}{hanzi}_sentence.mp3"

            synthesize_audio(fields["Hanzi"], audio_word_path)
            synthesize_audio(fields["Sentence"], audio_sent_path)

            fields["Audio (Word)"] = f"[sound:{os.path.basename(audio_word_path)}]"
            fields["Audio (Sentence)"] = f"[sound:{os.path.basename(audio_sent_path)}]"

            processed_rows.append(fields)
            row_count += 1

            # Save progress every 10 rows
            if row_count % 10 == 0:
                tqdm.write(f"Saving progress after {row_count} words...")
                save_progress(csv.DictWriter(open(output_file, "w", newline="", encoding="utf-8"), fieldnames=fieldnames), processed_rows, output_file)

        except Exception as e:
            tqdm.write(f"Error processing {hanzi}: {e}")
            failed_words.append(hanzi)
            continue

    # Final save of all processed rows
    if processed_rows:
        save_progress(csv.DictWriter(open(output_file, "w", newline="", encoding="utf-8"), fieldnames=fieldnames), processed_rows, output_file)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_rows)} words")
    if failed_words:
        print(f"Failed to process: {len(failed_words)} words")
        print("Failed words:", ", ".join(failed_words))

if __name__ == "__main__":
    main()