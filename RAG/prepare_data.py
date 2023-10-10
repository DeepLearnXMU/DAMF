import jsonlines

for split in ['train', 'dev', 'test']:
    with jsonlines.open(f'../saved_data/KdConv/{split}.json', 'r') as reader:
        raw_data = [line for line in reader]

    writer = jsonlines.open(f'../saved_data/data_kdconv/{split}.json', 'w')
    example_num = 0
    for session in raw_data:
        dialogue = []
        for turn in session['dialogue']:
            text = turn['text']

            if dialogue:
                writer.write({'context': '\n'.join(dialogue), 'response': text})
                example_num += 1

            dialogue.append(text)

    print(f"{split}: {example_num}")




