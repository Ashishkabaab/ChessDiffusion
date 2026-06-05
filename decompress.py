import zstandard as zstd
import os

input_file = '/project/lichess_db_puzzle.csv.zst'
output_file = '/project/lichess_db_puzzle.csv'

print("Decompressing...")
with open(input_file, 'rb') as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_file, 'wb') as destination:
        dctx.copy_stream(compressed, destination)

print(f"Done. Output size: {os.path.getsize(output_file) / 1e9:.2f} GB")
