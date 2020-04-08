import argparse
import os

import pandas as pd


def load_results(files):
    version_results = dict()
    for filename in files:
        version = os.path.basename(filename).replace('.csv', '')
        version_results[version] = pd.read_csv(filename, index_col=0)

    return version_results


def get_wins(scores):
    return (scores.T.rank(method='min', ascending=False) == 1).sum(axis=1)


def get_summary(results):
    summary = {}
    for version, scores in results.items():
        summary[version] = get_wins(scores)

    summary_df = pd.DataFrame(summary)
    summary_df.index.name = 'tuner'
    columns = summary_df.columns.sort_values(ascending=False)
    return summary_df[columns]


def add_sheet(df, name, writer, cell_fmt, index_fmt, header_fmt):
    df = df.reset_index()
    df.to_excel(writer, sheet_name=name, startrow=1, index=False, header=False)

    worksheet = writer.sheets[name]

    for idx, column in enumerate(df.columns):
        width = max(len(column), *df[column].astype(str).str.len()) + 1
        worksheet.write(0, idx, column, header_fmt)
        if idx:
            worksheet.set_column(idx, idx, width, cell_fmt)
        else:
            worksheet.set_column(idx, idx, width, index_fmt)


def write_summary(summary, results, output):
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

    add_sheet(summary, 'summary', writer, cell_fmt, index_fmt, header_fmt)

    for version in summary.columns:
        add_sheet(results[version], version, writer, cell_fmt, index_fmt, header_fmt)

    writer.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='Input path with results.')
    parser.add_argument('output', help='Output file.')
    args = parser.parse_args()

    results = load_results(args.input)
    summary = get_summary(results)
    write_summary(summary, results, args.output)


if __name__ == '__main__':
    main()
