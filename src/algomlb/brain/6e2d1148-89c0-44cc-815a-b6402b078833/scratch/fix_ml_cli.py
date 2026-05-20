path = "src/algomlb/cli/ml.py"
with open(path, "r") as f:
    lines = f.readlines()

new_block = [
    '    if target == "pa_outcome":\n',
    "        X, y = pipeline.build_pa_matrix(\n",
    '            data["pas"],\n',
    '            data["pitcher_gold"],\n',
    '            data["batter_gold"],\n',
    '            lineups_df=data["lineups"] if not data["lineups"].empty else None,\n',
    '            elo_df=data["elo"],\n',
    "        )\n",
    "    else:\n",
    "        X, y = pipeline.build_uranium_matrix(\n",
    '            data["games"],\n',
    '            data["pitcher_gold"],\n',
    '            data["lineups"] if not data["lineups"].empty else None,\n',
    '            data["batter_gold"] if not data["batter_gold"].empty else None,\n',
    '            elo_df=data["elo"],\n',
    '            pythag_df=data["pythag"],\n',
    '            re24_df=data["re24"],\n',
    "        )\n",
]

# Find all occurrences of the build_uranium_matrix call block
target_start = "    X, y = pipeline.build_uranium_matrix(\n"
target_end = "    )\n"

i = 0
while i < len(lines):
    if lines[i] == target_start:
        start_idx = i
        # Find the next '    )\n'
        j = i + 1
        while j < len(lines) and lines[j] != target_end:
            j += 1
        if j < len(lines):
            # Replace lines[i:j+1] with new_block
            lines[i : j + 1] = new_block
            i += len(new_block)
            continue
    i += 1

with open(path, "w") as f:
    f.writelines(lines)
print("Successfully updated ml.py")
