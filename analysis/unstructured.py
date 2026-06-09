import numpy as np
import textstat
from textblob import TextBlob
from collections import Counter
import cv2
from sklearn.cluster import KMeans

def compute_text_health(text: str):
    """Compute unstructured health score for text based on readability and stop-word density."""
    if not text.strip():
        return {"total_score": 0, "grade": "F", "readability": 0, "complexity": "Empty"}
    
    # Flesch Reading Ease (0-100, higher is easier/better)
    flesch_score = textstat.flesch_reading_ease(text)
    # Clamp between 0 and 100
    readability_score = max(0, min(100, flesch_score))
    
    # Calculate stop word density (proxy for natural language vs gibberish/code)
    words = text.lower().split()
    if not words:
        return {"total_score": 0, "grade": "F", "readability": 0, "complexity": "Empty"}
        
    common_stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "was"}
    stop_count = sum(1 for w in words if w in common_stop_words)
    stop_density = stop_count / len(words)
    
    # Good natural text usually has ~20-40% stop words.
    # If it's < 5% or > 60%, it's likely a list of keywords or gibberish
    if 0.15 <= stop_density <= 0.45:
        density_score = 100
    elif stop_density < 0.15:
        density_score = max(0, (stop_density / 0.15) * 100)
    else:
        density_score = max(0, 100 - ((stop_density - 0.45) * 200))
        
    total_score = (readability_score * 0.6) + (density_score * 0.4)
    total_score = round(max(0, min(100, total_score)), 1)
    
    grade = "A" if total_score >= 90 else "B" if total_score >= 80 else "C" if total_score >= 70 else "D" if total_score >= 60 else "F"
    
    return {
        "total_score": total_score,
        "grade": grade,
        "readability_raw": round(flesch_score, 1),
        "stop_density": round(stop_density * 100, 1),
        "complexity_level": textstat.text_standard(text)
    }

def compute_image_health(image_bytes: bytes):
    """Compute unstructured health score for images based on blurriness and contrast."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"total_score": 0, "grade": "F", "blur_score": 0, "contrast_score": 0}
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blurriness (Variance of Laplacian)
    # Higher variance = sharper image. Threshold is often around 100.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100, (laplacian_var / 500) * 100) # Cap at 100
    
    # Contrast (Standard Deviation of pixel intensities)
    # Higher std = higher contrast
    contrast = gray.std()
    contrast_score = min(100, (contrast / 80) * 100)
    
    total_score = round((blur_score * 0.5) + (contrast_score * 0.5), 1)
    grade = "A" if total_score >= 90 else "B" if total_score >= 80 else "C" if total_score >= 70 else "D" if total_score >= 60 else "F"
    
    return {
        "total_score": total_score,
        "grade": grade,
        "blurriness": round(laplacian_var, 1),
        "contrast": round(contrast, 1),
    }

def extract_ngrams_and_lemmas(text: str):
    """Extract lemmatized word frequencies, bigrams, and trigrams."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.util import ngrams
        from nltk.tokenize import word_tokenize
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and clean
        tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        
        # Lemmatize non-stopwords
        lemmas = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        
        # Frequencies
        top_words = Counter(lemmas).most_common(20)
        
        # N-Grams
        bigrams = Counter([' '.join(bg) for bg in ngrams(lemmas, 2)]).most_common(15)
        trigrams = Counter([' '.join(tg) for tg in ngrams(lemmas, 3)]).most_common(10)
        
        return {
            "top_words": top_words,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "sentiment": TextBlob(text).sentiment
        }
    except Exception as e:
        # Fallback if NLTK fails
        return {"error": str(e)}

def extract_dominant_colors(image_bytes: bytes, k=5):
    """Extract dominant colors from an image using KMeans, returning hex codes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return []
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize for speed
    h, w = img.shape[:2]
    if h > 200 or w > 200:
        scale = 200 / max(h, w)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        
    pixels = img.reshape((-1, 3))
    
    # Filter out pure white/black background pixels (e.g. padding)
    mask = ~((pixels.sum(axis=1) > 750) | (pixels.sum(axis=1) < 15))
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < k:
        filtered_pixels = pixels # fallback
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(filtered_pixels)
    
    colors = kmeans.cluster_centers_
    # Convert to hex
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors]
    return hex_colors

def run_ocr(image_bytes: bytes):
    """Extract text from image using EasyOCR."""
    try:
        import easyocr
        # Use English model. Set gpu=False if torch isn't configured for CUDA properly
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        result = reader.readtext(image_bytes, detail=0) # returns list of strings
        return " ".join(result)
    except Exception as e:
        return f"OCR Failed: {str(e)}"
