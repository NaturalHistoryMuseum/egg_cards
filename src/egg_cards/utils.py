def camelcase(words):  
    converted = "".join(word[0].upper() + word[1:].lower() for word in words.split())  
    return converted[0].lower() + converted[1:]  