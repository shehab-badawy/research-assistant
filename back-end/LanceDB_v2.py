# Setup
## Imports
import pyarrow.parquet as pq
import pyarrow as pa
import lancedb
import os

## Configuration
embedding_file_path = r"C:\Users\User\Downloads\arxiv_embeddings_fixed.parquet"
metadata_file_path = r"C:\Users\User\Downloads\arxiv_abstracts.parquet"
table_name = "reasearchgpt"
batch_size = 10_000

## Load Parquet files
embed_pf = pq.ParquetFile(embedding_file_path)
meta_pf = pq.ParquetFile(metadata_file_path)

## Connect to LanceDB
db = lancedb.connect(
    uri="db://researchgpt-17xoa0",
    api_key="",
    region="us-east-1"
)

## Define schema (optional: dynamic based on first batch)
first_embed = next(embed_pf.iter_batches(batch_size=1))
first_meta = next(meta_pf.iter_batches(batch_size=1))

sample_batch = pa.table([
    pa.Table.from_batches([first_meta]).column("title"),
    pa.Table.from_batches([first_meta]).column("authors"),
    pa.Table.from_batches([first_meta]).column("abstract"),
    pa.Table.from_batches([first_meta]).column("year"),
    pa.Table.from_batches([first_embed]).column("embeddings")
], names=["title", "authors", "abstract", "year", "embeddings"])

# Create table if not exists
if table_name not in db.table_names():
    print("⚙️ Creating new table...")
    db.create_table(table_name, data=sample_batch)
else:
    print("📁 Opening existing table...")
table = db.open_table(table_name)

## Upload in batches
embed_batches = embed_pf.iter_batches(batch_size=batch_size)
meta_batches = meta_pf.iter_batches(batch_size=batch_size)

for idx, (embed_rb, meta_rb) in enumerate(zip(embed_batches, meta_batches), start=1):
    embed_table = pa.Table.from_batches([embed_rb])
    meta_table = pa.Table.from_batches([meta_rb])

    # Combine columns
    batch = pa.table([
        meta_table.column("title"),
        meta_table.column("authors"),
        meta_table.column("abstract"),
        meta_table.column("year"),
        embed_table.column("embeddings")
    ], names=["title", "authors", "abstract", "year", "embeddings"])

    # Upload
    table.add(batch)
    print(f"✅ Uploaded batch {idx} ({batch.num_rows} rows)")

print("🎉 Upload Completed")
