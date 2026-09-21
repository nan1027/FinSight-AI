# Financial PhraseBank (Sentiment Analysis)

## Dataset name
Financial PhraseBank

## Source
Official dataset repository on the Hugging Face Hub:
https://huggingface.co/datasets/takala/financial_phrasebank

The dataset contains sentences from financial news and analyst reports annotated for the semantic orientation of the statement relative to a financial/investment context.

## Original task
Sentence-level financial sentiment classification with three labels:
- positive
- neutral
- negative

## Sentiment labels
- positive
- neutral
- negative

## Expected dataset format
The project uses the fully agreed subset in the official archive:
- FinancialPhraseBank-v1.0/Sentences_AllAgree.txt

Each line contains:
- sentence text
- an @ delimiter
- one sentiment label

Example format:

According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .@neutral

The download script converts this file into a canonical CSV at:

data/raw/financial_phrasebank/financial_phrasebank.csv

with columns:
- sentence
- label

## Licensing / distribution considerations
The official dataset metadata identifies the repository as using the CC BY-NC-SA 3.0 license. Because the raw dataset is distributed under that license and the repo ignores raw-data redistribution in version control, the project stores the downloaded data under `data/raw/` locally but does not commit the raw files to Git. The project uses the official Hugging Face archive rather than scraping an unofficial mirror.

## Exact location expected by the project
The scripts in this folder expect:

data/raw/financial_phrasebank/financial_phrasebank.csv

and they also keep the downloaded archive at:

data/raw/financial_phrasebank/FinancialPhraseBank-v1.0.zip

## Manual download step if automatic download is unavailable
If the automatic `huggingface_hub` workflow fails in a specific environment, use the official Hugging Face dataset page and download the archive manually:

1. Visit: https://huggingface.co/datasets/takala/financial_phrasebank
2. Download `data/FinancialPhraseBank-v1.0.zip`
3. Save the archive under `data/raw/financial_phrasebank/`
4. Extract the files and ensure `Sentences_AllAgree.txt` is present
5. Run `ml/sentiment/download_dataset.py` or convert the extracted text file into a CSV with the same schema used by the project

This keeps the source official and reproducible without relying on unsupported or scraped mirrors.
