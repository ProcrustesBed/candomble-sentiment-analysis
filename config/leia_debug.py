from leia import SentimentIntensityAnalyzer as LeiaAnalyzer

def test_leia_lexicon_expansion():
    """
    Vollständiger Test ob LeIAs Lexikon erweiterbar ist.
    Kombiniert Emoji-Preprocessing mit Lexikon-Updates.
    """
    
    # LeIA initialisieren
    leia = LeiaAnalyzer()
    
    print("=== LeIA Lexikon-Erweiterungstest ===\n")
    
    # 1. Baseline testen
    test_text = "Axé ❤️🙏🔥"
    baseline_score = leia.polarity_scores(test_text)['compound']
    print(f"Baseline Score für '{test_text}': {baseline_score}")
    
    # 2. Emoji-zu-Portugiesisch Mapping
    emoji_to_portuguese = {
        '❤️': 'amor',
        '💖': 'amor', 
        '💕': 'carinho',
        '😍': 'apaixonado',
        '😊': 'feliz',
        '🥰': 'apaixonado',
        '🙏': 'oracao',
        '🙌': 'celebracao',
        '✨': 'brilhante',
        '🌟': 'estrela',
        '🎉': 'festa',
        '🎊': 'celebracao',
        '🏹': 'oxossi',
        '⚡': 'xango',
        '🌊': 'iemanja',
        '🔥': 'fogo',
        '🐍': 'serpente',
        '🕊️': 'pomba',
        '😢': 'triste',
        '😡': 'raiva',
        '💔': 'coracao_partido'
    }
    
    # 3. Portugiesische Wörter mit Sentiment-Werten
    portuguese_sentiment = {
        'amor': 2.5,
        'carinho': 2.0,
        'apaixonado': 2.8,
        'feliz': 2.3,
        'oracao': 1.5,
        'celebracao': 2.0,
        'brilhante': 1.8,
        'estrela': 1.2,
        'festa': 2.2,
        'oxossi': 3.0,
        'xango': 3.0,
        'iemanja': 3.0,
        'fogo': 2.0,  # Überschreibt LeIAs -1.4
        'serpente': 1.5,
        'pomba': 2.5,
        'triste': -2.1,
        'raiva': -2.8,
        'coracao_partido': -3.0
    }
    
    # 4. Yoruba/Candomblé Begriffe
    religious_terms = {
        'axe': 2.5,
        'ase': 2.5,
        'oxala': 3.0,
        'iemanja': 3.0,
        'oxossi': 3.0,
        'oxum': 3.0,
        'iansa': 3.0,
        'ogum': 3.0,
        'exu': 2.0,
        'terreiro': 2.0,
        'orixa': 2.5,
        'ebo': 1.5,
        'oferenda': 2.0,
        'mojuba': 2.0
    }
    
    # 5. Versuche Lexikon zu erweitern
    print("\n=== Versuche Lexikon-Erweiterung ===")
    
    original_size = len(leia.lexicon)
    print(f"Ursprüngliche Lexikon-Größe: {original_size}")
    
    # Teste verschiedene Update-Methoden
    all_updates = {**portuguese_sentiment, **religious_terms}
    
    success_count = 0
    for term, value in all_updates.items():
        try:
            # Methode 1: Direkte Zuweisung
            leia.lexicon[term] = value
            
            # Sofort testen ob es funktioniert
            if term in leia.lexicon and leia.lexicon[term] == value:
                success_count += 1
            else:
                print(f"FEHLER: {term} konnte nicht hinzugefügt werden")
                
        except Exception as e:
            print(f"FEHLER bei {term}: {e}")
    
    new_size = len(leia.lexicon)
    print(f"Neue Lexikon-Größe: {new_size}")
    print(f"Erfolgreich hinzugefügt: {success_count}/{len(all_updates)} Begriffe")
    
    # 6. Emoji-Preprocessing Funktion
    def preprocess_emojis(text):
        processed = text
        for emoji, portuguese in emoji_to_portuguese.items():
            processed = processed.replace(emoji, f' {portuguese} ')
        return processed
    
    # 7. Vollständiger Test mit Preprocessing
    processed_text = preprocess_emojis(test_text)
    print(f"\nOriginal: '{test_text}'")
    print(f"Verarbeitet: '{processed_text}'")
    
    new_score = leia.polarity_scores(processed_text)['compound']
    print(f"Neuer Score: {new_score}")
    print(f"Verbesserung: {new_score - baseline_score:+.4f}")
    
    # 8. Teste verschiedene Beispiele
    test_cases = [
        "Axé para todos ❤️",
        "🙏 Oxóssi protetor 🏹",
        "Terreiro lindo ✨🌟",
        "😢 Muito triste",
        "Festa no terreiro 🎉🎊",
        "❤️🏹"
    ]
    
    print("\n=== Test verschiedener Beispiele ===")
    for case in test_cases:
        original = leia.polarity_scores(case)['compound']
        processed = preprocess_emojis(case)
        enhanced = leia.polarity_scores(processed)['compound']
        print(f"'{case}':")
        print(f"  Original: {original:.4f}")
        print(f"  Verarbeitet: '{processed}' -> {enhanced:.4f}")
        print(f"  Verbesserung: {enhanced - original:+.4f}\n")
    
    # 9. Fazit
    if success_count > 0 and new_score > baseline_score:
        print("✅ SUCCESS: LeIA-Lexikon ist erweiterbar und Emoji-Preprocessing funktioniert!")
        return True
    else:
        print("❌ FAILURE: LeIA-Lexikon ist nicht erweiterbar oder Preprocessing funktioniert nicht.")
        return False

# Test ausführen
if __name__ == "__main__":
    test_leia_lexicon_expansion()