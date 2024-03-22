from transformers import MarianMTModel, MarianTokenizer

def translate_arabic_to_english(text):
    # Load pre-trained model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    tokenized_text = tokenizer.encode(text, return_tensors="pt")

    # Translate the tokenized text
    translated_text = model.generate(tokenized_text, max_length=1000, num_beams=4, early_stopping=True)

    # Decode the translated text
    translated_text_decoded = tokenizer.decode(translated_text[0], skip_special_tokens=True)

    return translated_text_decoded

# Example usage:
arabic_text = input
english_translation = translate_arabic_to_english(arabic_text)
print(english_translation)
