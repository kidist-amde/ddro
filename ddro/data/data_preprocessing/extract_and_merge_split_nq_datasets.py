import argparse
import jsonlines
import re
from tqdm.auto import tqdm
import gzip
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import BertTokenizer

DEV_SET_SIZE = 7830
TRAIN_SET_SIZE = 307373

def process_nq_devdataset(input_file_path, sample_size=None):
    nq_dev = []
    with gzip.open(input_file_path, 'rt') as f:
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or DEV_SET_SIZE, desc="Processing Dev Dataset")):
            if sample_size and idx >= sample_size:
                break  # Stop after processing the specified number of samples
            arr = []
            ## question_text
            question_text = item['question_text']
            arr.append(question_text)

            tokens = []
            for i in item['document_tokens']:
                tokens.append(i['token'])
            document_text = ' '.join(tokens)
            
            ## example_id
            example_id = str(item['example_id'])
            arr.append(example_id)

            # document_text = item['document_text']
            ## long_answer
            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0

            long_answers = [
                a['long_answer']
                for a in item['annotations']
                if a['long_answer']['start_token'] >= 0 and has_long_answer
            ]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer) if has_long_answer else arr.append('')

            # short_answer
            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [
                a['short_answers']
                for a in item['annotations']
                if a['short_answers'] and has_short_answer
            ]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = []
                for i in short_answers[0]:
                    start_token_s = i['start_token']
                    end_token_s = i['end_token']
                    shorta = ' '.join(x[start_token_s:end_token_s])
                    sa.append(shorta)
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer) if has_short_answer else arr.append('')

            ## title
            arr.append(item['document_title'])

            ## abs
            if document_text.find('<P>') != -1:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start+3:abs_end]
            else:
                abs = ''
            arr.append(abs)

            ## content
            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                if document_text.rfind('</Ul>') != -1:
                    final = document_text.rindex('</Ul>')
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
                else:
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
            else:
                content = document_text[abs_end+4:]
                content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                content = re.sub(' +', ' ', content)
                arr.append(content)

            arr.append(item.get('document_url', ''))
            doc_tac = item['document_title'] + abs + content
            arr.append(doc_tac)
            language = 'en'
            arr.append(language)
            nq_dev.append(arr)

    nq_dev_df = pd.DataFrame(nq_dev, columns=['query', 'id', 'long_answer', \
                                                   'short_answer', 'title', \
                                                    'abstract', 'content', 'document_url', \
                                                    'doc_tac', 'language'])
    return nq_dev_df

def process_nq_traindataset(input_file_path, sample_size=None):
    nq_train = []
    with gzip.open(input_file_path, 'rt') as f:
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or TRAIN_SET_SIZE, desc="Processing Train Dataset")):
            if sample_size and idx >= sample_size:
                break  
             ## question_text
            arr = []
            question_text = item['question_text']
            arr.append(question_text)

            ## example_id
            example_id = str(item['example_id'])
            arr.append(example_id)
            
            document_text = item['document_text']
            
            ## long_answer
            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0

            long_answers = [
                a['long_answer']
                for a in item['annotations']
                if a['long_answer']['start_token'] >= 0 and has_long_answer
            ]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer) if has_long_answer else arr.append('')

            # short_answer
            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [
                a['short_answers']
                for a in item['annotations']
                if a['short_answers'] and has_short_answer
            ]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = []
                for i in short_answers[0]:
                    start_token_s = i['start_token']
                    end_token_s = i['end_token']
                    shorta = ' '.join(x[start_token_s:end_token_s])
                    sa.append(shorta)
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer) if has_short_answer else arr.append('')

            ## title
            if document_text.find('<H1>') != -1:
                title_start = document_text.index('<H1>')
                title_end = document_text.index('</H1>')
                title = document_text[title_start+4:title_end]
            else:
                title = ''
            arr.append(title)

            ## abs
            if document_text.find('<P>') != -1:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start+3:abs_end]
            else:
                abs = ''
            arr.append(abs)

            ## content
            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                if document_text.rfind('</Ul>') != -1:
                    final = document_text.rindex('</Ul>')
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
                else:
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
            else:
                content = document_text[abs_end+4:]
                content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                content = re.sub(' +', ' ', content)
                arr.append(content)

            arr.append(item.get('document_url', ''))
            doc_tac = title + abs + content
            arr.append(doc_tac)

            language = 'en'
            arr.append(language)
            nq_train.append(arr)

    nq_train_df = pd.DataFrame(nq_train, columns=['query', 'id', 'long_answer', \
                                                   'short_answer', 'title', \
                                                    'abstract', 'content', 'document_url', \
                                                    'doc_tac', 'language'])
    return nq_train_df

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def lower(x):
    text = tokenizer.tokenize(x)
    id_ = tokenizer.convert_tokens_to_ids(text)
    return tokenizer.decode(id_)

def create_document_mapping(nq_all_doc):
    """
    Create a mapping between document titles, document contents, document IDs, and URLs.
    
    Args:
        nq_all_doc (DataFrame): A pandas DataFrame containing all documents with the columns:
                                ['query', 'id', 'long_answer', 'short_answer', 'title', 
                                 'abstract', 'content', 'document_url', 'doc_tac', 'language']

    Returns:
        dict: title_doc -> Mapping from title to concatenated content.
        dict: title_doc_id -> Mapping from title to a unique document index.
        dict: id_doc -> Mapping from document index to concatenated content.
        dict: ran_id_old_id -> Mapping from document index to original document ID.
        dict: doc_id_url -> Mapping from document index to document URL.
    """
    
    title_doc = {}
    title_doc_id = {}
    id_doc = {}
    ran_id_old_id = {}
    doc_id_url = {}
    idx = 0
    
    for i in range(len(nq_all_doc)):
        title = nq_all_doc['title'][i]
        doc_tac = nq_all_doc['doc_tac'][i]
        doc_id = nq_all_doc['id'][i]
        doc_url = nq_all_doc['document_url'][i]
        
        # Create the mappings
        title_doc[title] = doc_tac
        title_doc_id[title] = idx
        id_doc[idx] = doc_tac
        ran_id_old_id[idx] = doc_id
        doc_id_url[idx] = doc_url
        
        idx += 1
    
    return title_doc, title_doc_id, id_doc, ran_id_old_id, doc_id_url

def main():
    
    parser = argparse.ArgumentParser(description="Extract, process, and merge Google NQ data")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the NQ development dataset (.jsonl.gz)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the NQ training dataset (.jsonl.gz)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final merged data (TSV format)")
    parser.add_argument("--output_train_file", type=str, required=True, help="Path to save the training data (TSV format)")
    parser.add_argument("--output_val_file", type=str, required=True, help="Path to save the validation data (TSV format)")
    # parser.add_argument("--doc_content_file", type=str, required=True, help="Path to save the NQ_doc_content.tsv")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to process (optional)")
    args = parser.parse_args()


    # Process both dev and train datasets
    nq_dev = process_nq_devdataset(args.dev_file, args.sample_size)
    nq_train = process_nq_traindataset(args.train_file, args.sample_size)

    # Apply tokenization and lowercasing to titles
    nq_dev['title'] = nq_dev['title'].map(lower)
    nq_train['title'] = nq_train['title'].map(lower)

    # Concatenate dev and train data, drop duplicates based on title
    nq_all_doc = pd.concat([nq_train, nq_dev], ignore_index=True)
    nq_all_doc.drop_duplicates('title', inplace=True)
    nq_all_doc.reset_index(drop=True, inplace=True)

    print("Total number of documents: ", len(nq_all_doc))

    # Save only the gzipped TSV file
    # with gzip.open(args.output_file + '.gz', 'wt') as f:
    #     nq_all_doc.to_csv(f, sep='\t', index=False, header=False)  
    # print(f"Final merged data saved to {args.output_file}.gz")


    # Convert the DataFrame to Hugging Face Dataset
    nq_dataset = Dataset.from_pandas(nq_all_doc)

    # Split the dataset into training and validation sets (80/20 split)
    split_dataset = nq_dataset.train_test_split(test_size=0.2, shuffle=True)

    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Save the datasets to gzipped TSV files
    train_df = train_dataset.to_pandas()
    val_df = val_dataset.to_pandas()

    with gzip.open(args.output_train_file + '.gz', 'wt') as train_f:
        train_df.to_csv(train_f, sep='\t', index=False, header=False)

    with gzip.open(args.output_val_file + '.gz', 'wt') as val_f:
        val_df.to_csv(val_f, sep='\t', index=False, header=False)

    print(f"Training data saved to {args.output_train_file}.gz")
    print(f"Validation data saved to {args.output_val_file}.gz")

if __name__ == "__main__":
    main()
