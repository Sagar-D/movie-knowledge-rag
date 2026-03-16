import kagglehub
import pandas as pd
from enum import StrEnum
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from cinerag.storage import s3_client
from cinerag import config

logger = logging.getLogger(__name__)

DATA_FILE_NAME = "wiki_movie_plots_deduped_with_summaries.csv"
KAGGLE_DATASET = "gabrieltardochi/wikipedia-movie-plots-with-plot-summaries"


class Columns(StrEnum):
    TITLE = "Title"
    YEAR = "Release Year"
    ORIGIN = "Origin/Ethnicity"
    GENRE = "Genre"
    DIRECTOR = "Director"
    CAST = "Cast"
    PLOT = "Plot"
    SUMMARY = "PlotSummary"
    WIKI_LINK = "Wiki Page"
    RAG_TEXT = "RAG Text"
    RAG_METADATA = "RAG Metadata"


def _format_rag_text(row: pd.Series) -> str:

    formatted_text = f"""\
Movie : {row[Columns.TITLE]}
Release Year : {row[Columns.YEAR]}
Origin : {row[Columns.ORIGIN]}
Genre : {row[Columns.GENRE]}
Directed by : {row[Columns.DIRECTOR]}
Cast : {row[Columns.CAST]}

Plot :
{row[Columns.PLOT]}

Summary :
{row[Columns.SUMMARY]}\
""".strip()
    return formatted_text


def _format_rag_metadata(row: pd.Series) -> Dict:

    metadata = {
        "title": row[Columns.TITLE],
        "year": None if row[Columns.YEAR] == 0 else row[Columns.YEAR],
        "origin": row[Columns.ORIGIN],
        "genre": row[Columns.GENRE],
        "director": [director.strip() for director in row[Columns.DIRECTOR].split(",")],
        "cast": [actor.strip() for actor in row[Columns.CAST].strip().split(",")],
        "wiki_link": row[Columns.WIKI_LINK],
    }

    return metadata


def load_movie_dataset() -> Path:

    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    dataset_path = dataset_path / DATA_FILE_NAME
    logger.info("Dataset path on local: %s", dataset_path)
    if not s3_client.file_exists(
        s3_file_name=f"{config.DEFAULT_FILE_NAME}.csv", type="raw"
    ):
        s3_client.upload_raw_file(
            file_path=dataset_path, s3_file_name=f"{config.DEFAULT_FILE_NAME}.csv"
        )
        logger.info("Dataset uploaded to S3 successfully: %s", dataset_path.name)
    return dataset_path


def build_movie_rag_documents(
    dataset_path: Path = None,
) -> Tuple[List[str], List[Dict]]:

    if dataset_path == None:
        dataset_path = load_movie_dataset()

    df = pd.read_csv(dataset_path)
    df.dropna(subset=[Columns.TITLE], inplace=True)
    text_cols = df.columns.difference([Columns.YEAR, Columns.WIKI_LINK])
    df[text_cols] = df[text_cols].fillna("Unknown").astype(str)
    df[Columns.YEAR] = df[Columns.YEAR].fillna(0).astype(int)
    df[Columns.WIKI_LINK] = df[Columns.WIKI_LINK].fillna("").astype(str)

    df[Columns.RAG_TEXT] = df.apply(_format_rag_text, axis=1)
    df[Columns.RAG_METADATA] = df.apply(_format_rag_metadata, axis=1)

    logger.info("RAG documents generated for %s movie dataset", len(df))

    jsonl_data = [
        {"text": text, "metadata": meta}
        for text, meta in zip(
            df[Columns.RAG_TEXT].to_list(), df[Columns.RAG_METADATA].to_list()
        )
    ]

    if not s3_client.file_exists(
        s3_file_name=f"{config.DEFAULT_FILE_NAME}.jsonl", type="processed"
    ):
        s3_client.upload_processed_jsonl(
            data=jsonl_data, s3_file_name=f"{config.DEFAULT_FILE_NAME}.jsonl"
        )
        logger.info(
            "RAG documents uploaded to S3 successfully: %s",
            f"{config.DEFAULT_FILE_NAME}.jsonl",
        )
    return jsonl_data
