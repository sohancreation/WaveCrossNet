import fitz

doc = fitz.open('WaveCrossNet_Publication.pdf')
print('Number of pages:', len(doc))
for i in range(len(doc)):
    print(f'\n=== Page {i+1} ===')
    blocks = doc[i].get_text('blocks')
    for b in blocks:
        col = '1' if b[0] < 300 else '2'
        text = b[4].strip().replace('\n', ' ')[:50]
        text = text.encode('ascii', 'replace').decode('ascii')
        print(f'  Col {col} ({b[0]:.1f}, {b[1]:.1f} -> {b[2]:.1f}, {b[3]:.1f}): {text}')
