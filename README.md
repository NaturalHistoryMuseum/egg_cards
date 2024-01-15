# Egg Card Data Extraction

__Aim__: Extract text from museum egg cards.

Basic outline of pipeline:
1. Google Vision to find all text on egg card.
2. Approximate boxes around each category section, based on Google Vision textboxes.
3. Associate each textbox to a category box.
4. Dictionary output of Google Vision's OCR result, per category.

