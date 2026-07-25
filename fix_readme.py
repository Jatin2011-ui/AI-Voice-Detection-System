with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()
# Remove BOM and any stray BOM artifacts
content = content.replace('\ufeff', '')
with open('README.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print('BOM removed')
