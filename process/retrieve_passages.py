import os
from sys import argv
import jsonlines
from search import *
from tqdm import tqdm
import threading
import time

_, data_path, output_path = argv

collected_queries = set()
with jsonlines.open(data_path, 'r') as reader:
    for line in reader:
        for turn in line:
            for entity in turn['parse']['entities']:
                if entity[3][0].split('.')[0] not in ['time', 'quality']:
                    collected_queries.add(entity[0])
            if turn['queries']:
                collected_queries.update(set(turn['queries']))

collected_queries = {query for query in collected_queries if query.strip()}
print('keyword num', len(collected_queries))

passage_fpath = output_path
retrieved_queries = set()
if os.path.exists(passage_fpath):
    saved_data = []
    with jsonlines.open(passage_fpath, 'r') as reader:
        for line in reader:
            if line['query'] and line['passages']:
                retrieved_queries.add(line['query'])
                saved_data.append(line)
    writer = jsonlines.open(passage_fpath, 'w')
    writer.write_all(saved_data)
else:
    writer = jsonlines.open(passage_fpath, 'w')
max_page_num = 10
left_queries = collected_queries - retrieved_queries
print(len(left_queries))
pbar = tqdm(total=len(left_queries))

left_queries = iter(left_queries)
def thread_fn(name, lock_in, lock_out):
    global pbar
    global left_queries
    global writer
    while True:
        with lock_in:
            try:
                keyword = next(left_queries)
            except StopIteration:
                break
        fail_cnt = 0
        while True:
            results = call_sogou_search(SOGOU_URL, keyword)
            if results:
                passages = []
                for page in results['result']['pages'][:max_page_num]:
                    pageno = page['pageno']
                    title = page['title']
                    abstract = page['abstract']
                    url = page['url']
                    # text = jionlp.clean_text(page['abstract']).strip()
                    passages.append({'pageno': pageno, 'title': title, 'abstract': abstract, 'url': url})
                with lock_out:
                    writer.write({'query': keyword, 'passages': passages})
                    pbar.update()
                    break
            else:
                fail_cnt += 1
                if fail_cnt >= 5:
                    break
                time.sleep(fail_cnt * 1)

thread_num = 10
threads = []
lock_in = threading.Lock()
lock_out = threading.Lock()
for i in range(thread_num):
    new_thread = threading.Thread(target=thread_fn, args=('Thread-{}'.format(i), lock_in, lock_out,))
    new_thread.start()
    threads.append(new_thread)
for thread in threads:
    thread.join()