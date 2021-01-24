s = "Salut salut "
def slice(text, n):
    text = text.replace(" ", "")
    return '\n'.join([ text[i:i+n] for i in range(0, len(text), n)  ])

print(slice(s, 3))