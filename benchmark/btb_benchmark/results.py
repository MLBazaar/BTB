import os

import pandas as pd


def load_results(files):
    version_results = dict()
    for filename in files:
        version = os.path.basename(filename).replace('.csv', '')
        version_results[version] = pd.read_csv(filename, index_col=0).round(6)

    return version_results


def get_wins(scores):
    is_winner = scores.T.rank(method='min', ascending=False) == 1
    return is_winner.sum(axis=1)


def get_exclusive_wins(scores):
    is_winner = scores.T.rank(method='min', ascending=False) == 1
    num_winners = is_winner.sum()
    is_exclusive = num_winners == 1
    is_exclusive_winner = is_winner & is_exclusive
    num_exclusive_wins = is_exclusive_winner.sum(axis=1)

    return num_exclusive_wins


def get_z_scores(scores):
    mean = scores.mean(axis=1)
    std = scores.std(axis=1)

    return ((scores.T - mean) / std).fillna(0).T.mean()


def get_summary(results, summary_function):
    summary = {}
    for version, scores in results.items():
        summary[version] = summary_function(scores)

    summary_df = pd.DataFrame(summary)
    summary_df.index.name = 'tuner'
    columns = summary_df.columns.sort_values(ascending=False)
    return summary_df[columns]


def add_sheet(dfs, name, writer, cell_fmt, index_fmt, header_fmt):
    startrow = 0
    widths = [0]
    if not isinstance(dfs, dict):
        dfs = {None: dfs}

    for df_name, df in dfs.items():
        df = df.reset_index()
        startrow += bool(df_name)
        df.to_excel(writer, sheet_name=name, startrow=startrow + 1, index=False, header=False)

        worksheet = writer.sheets[name]

        if df_name:
            worksheet.write(startrow - 1, 0, df_name, index_fmt)
            widths[0] = max(widths[0], len(df_name))

        for idx, column in enumerate(df.columns):
            worksheet.write(startrow, idx, column, header_fmt)
            width = max(len(column), *df[column].astype(str).str.len()) + 1
            if len(widths) > idx:
                widths[idx] = max(widths[idx], width)
            else:
                widths.append(width)

        startrow += len(df) + 2

    for idx, width in enumerate(widths):
        fmt = cell_fmt if idx else index_fmt
        worksheet.set_column(idx, idx, width + 1, fmt)


def write_results(results, output):
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    cell_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10"
    })
    index_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
    })
    header_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
        "bottom": 1
    })

    wins = get_summary(results, get_wins)
    summary = {
        'Number of Wins (with ties)': wins,
        'Number of Wins (exclusive)': get_summary(results, get_exclusive_wins)
    }
    add_sheet(summary, 'Number of Wins', writer, cell_fmt, index_fmt, header_fmt)

    for version in wins.columns:
        add_sheet(results[version], version, writer, cell_fmt, index_fmt, header_fmt)

    writer.save()
