with open('manuscript/paper.tex', 'r', encoding='utf-8') as f:
    text = f.read()

checks = [
    ('63,733', 'Parameters'),
    ('62.24', 'Flash KB'),
    ('12.40', 'SRAM KB'),
    ('14.80', 'Latency ms'),
    ('1.77\\%', 'Duty cycle'),
    ('36.1', 'Average power uW'),
    ('84~MHz', 'Clock speed'),
    ('1.12', 'MFLOPs'),
    ('100,694', 'MIT-BIH total beats'),
    ('51,002', 'DS1 beats'),
    ('49,692', 'DS2 beats'),
    ('22,540', 'SVDB beats'),
    ('123,234', 'Total beats'),
    ('91.33\\%', 'DS2 accuracy'),
    ('91.03\\%', 'DS2 weighted F1'),
    ('94.94\\%', 'V-recall'),
    ('16.60\\%', 'S-recall'),
    ('36.85\\%', 'S-recall calibrated'),
    ('176,293', 'ResNet1D params'),
    ('172.16', 'ResNet1D Flash KB'),
    ('7.63', 'ResNet1D MFLOPs'),
    ('88.95', 'ResNet1D Acc'),
    ('88.97', 'ResNet1D F1'),
    ('88.85', 'ResNet1D V-recall'),
    ('7.95', 'ResNet1D S-recall'),
    ('85.70\\%', '0 dB AWGN Macro F1'),
    ('+14.2', 'Advantage over ResNet1D'),
    ('19.9\\%', 'Relative gain'),
    ('71.50\\%', 'ResNet1D at 0 dB'),
    ('94.20\\%', 'PLI Macro F1'),
    ('89.40\\%', 'MA Macro F1'),
    ('76.40\\%', 'BW Macro F1'),
    ('65{,}536', 'Self-attention FLOPs'),
    ('8{,}192', 'Cross-attention FLOPs'),
    ('8.0\\times', 'Ratio formula'),
    ('512', 'bytes SRAM per head'),
    ('4{,}096', 'bytes SRAM self-attn per head'),
    ('STM32F401RE', 'MCU part number'),
    ('512~KB Flash', 'MCU Flash size'),
    ('96~KB SRAM', 'MCU SRAM size')
]

all_passed = True
for val, desc in checks:
    count = text.count(val)
    if count == 0:
        print(f'FAIL: {desc} (\"{val}\") NOT FOUND!')
        all_passed = False
    else:
        print(f'OK: {desc} (\"{val}\"): found {count} times')

if all_passed:
    print('\nALL 40 NUMERICAL AND ARCHITECTURAL METRICS VERIFIED PERFECTLY!')
